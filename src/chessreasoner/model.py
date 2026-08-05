"""ChessReasoner backbone and heads.

A pre-norm decoder-only transformer with grouped-query attention, SwiGLU feed
forward, RMSNorm and standard 1-D RoPE, plus three auxiliary heads that exist
only during training.

Two deliberate omissions, both from the adversarial review in
``docs/architecture.md``:

* **No dual-scheme (2-D board) RoPE.** The fixed a1->h8 raster already places
  file (+1), rank (+8) and diagonal (+9/+7) neighbours at constant 1-D offsets,
  which standard RoPE encodes well. The 2-D scheme only repairs file
  wrap-around (~10.9% of file adjacencies) and costs half the rotary
  dimensions. It is ablation A4, not the default (review item R1).
* **No FlashAttention.** FlashAttention-2 requires Ampere (sm_80+); the T4 is
  sm_75. ``F.scaled_dot_product_attention`` picks the memory-efficient backend,
  which does support Turing.

RMSNorm accumulates in fp32 because the T4 has no bf16 and fp16 training
diverges otherwise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vocab import MOVE_VOCAB_SIZE

IGNORE_INDEX = -100


@dataclass
class ChessReasonerConfig:
    vocab_size: int = 8192
    n_layers: int = 18
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: int = 4
    d_ff: int = 2048
    max_seq_len: int = 1024
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = True
    grad_checkpointing: bool = False
    loss_chunk_tokens: int = 0
    """Compute the LM loss in chunks of this many tokens instead of
    materializing (B, T, vocab) logits at once. At batch 16 x 1024 x 8192 the
    fp16 logits plus their fp32 copy are ~0.8 GB -- which the design's
    activation estimate omitted entirely. 0 disables chunking."""
    """Trades ~30% throughput for a large activation-memory saving. Review item
    R8 flags micro-batch 16 x 1024 as possibly too tight on a 16 GB T4 --
    scripts/calibrate_throughput.py decides, rather than assuming."""

    # auxiliary heads (training only)
    with_aux_heads: bool = True
    board_squares: int = 64
    board_classes: int = 13
    value_bins: int = 128
    policy_size: int = MOVE_VOCAB_SIZE

    # loss weights; annealed to zero over the second half of training
    lambda_board: float = 0.5
    lambda_value: float = 0.5
    lambda_policy: float = 0.5

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must divide evenly into n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be a multiple of n_kv_heads")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


def base_120m() -> ChessReasonerConfig:
    """The paper configuration: 119.6M backbone parameters, ~120 MB int8."""
    return ChessReasonerConfig()


def gate_20m() -> ChessReasonerConfig:
    """Go/no-go gate model. If this cannot learn to read a board from Tier-1
    data, nothing downstream will work -- and it costs ~4 T4-hours to find out."""
    return ChessReasonerConfig(n_layers=10, d_model=384, n_heads=6, n_kv_heads=2,
                               d_ff=1024)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()  # fp32 accumulation: the T4 has no bf16
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32 * self.weight.float()).to(dtype)


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device):
    """Always fp32 -- apply_rope casts to the activation dtype at the end."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, n_heads, T, head_dim); cos/sin: (T, head_dim/2), always fp32.

    The rotation is computed in fp32 and cast back explicitly. The previous
    version scattered into ``torch.empty_like(x)``, which happened to downcast
    correctly but only by accident -- and an in-place scatter is both slower and
    awkward under gradient checkpointing.
    """
    dtype = x.dtype
    x32 = x.float()
    x1, x2 = x32[..., 0::2], x32[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    rotated = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return rotated.flatten(-2).to(dtype)


class Attention(nn.Module):
    def __init__(self, cfg: ChessReasonerConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads, self.n_kv = cfg.n_heads, cfg.n_kv_heads
        self.hd = cfg.head_dim
        self.repeat = self.n_heads // self.n_kv
        self.wq = nn.Linear(cfg.d_model, self.n_heads * self.hd, bias=False)
        self.wk = nn.Linear(cfg.d_model, self.n_kv * self.hd, bias=False)
        self.wv = nn.Linear(cfg.d_model, self.n_kv * self.hd, bias=False)
        self.wo = nn.Linear(self.n_heads * self.hd, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, cache: dict | None = None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)

        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v

        if self.repeat > 1:  # GQA: broadcast each KV head to its query group
            k = k.repeat_interleave(self.repeat, dim=1)
            v = v.repeat_interleave(self.repeat, dim=1)

        # is_causal only means the right thing when the query and key lengths
        # match. With a cache and T > 1 (prefill continuation, or speculative
        # decoding) it silently lets a token attend to its own future, so build
        # the offset mask explicitly. Measured leak before this fix: 2.9e-01.
        n_keys = k.shape[2]
        if n_keys == T:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif T == 1:
            out = F.scaled_dot_product_attention(q, k, v)  # one query sees all past
        else:
            offset = n_keys - T
            causal = torch.ones(T, n_keys, dtype=torch.bool, device=q.device).tril(offset)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=causal)
        return self.wo(out.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, cfg: ChessReasonerConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: ChessReasonerConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.norm1(x), cos, sin, cache)
        return x + self.ffn(self.norm2(x))


def hl_gauss_targets(win_prob: torch.Tensor, n_bins: int, sigma_bins: float = 0.75):
    """Soft categorical targets over ``n_bins`` win-probability buckets.

    Regressing a scalar centipawn value is badly conditioned; a smoothed
    categorical target trains far more stably (Ruoss et al. 2024). ``win_prob``
    is in [0, 1].
    """
    device = win_prob.device
    centers = (torch.arange(n_bins, device=device).float() + 0.5) / n_bins
    sigma = sigma_bins / n_bins
    d = (centers[None, :] - win_prob[:, None]) / sigma
    logits = -0.5 * d.pow(2)
    return torch.softmax(logits, dim=-1)


class ChessReasoner(nn.Module):
    def __init__(self, cfg: ChessReasonerConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        if cfg.with_aux_heads:
            self.board_head = nn.Linear(cfg.d_model, cfg.board_squares * cfg.board_classes)
            self.value_head = nn.Linear(cfg.d_model, cfg.value_bins)
            self.policy_head = nn.Linear(cfg.d_model, cfg.policy_size)

        self._rope: tuple | None = None
        self.apply(self._init_weights)
        # scaled residual init: keep the residual stream variance flat with depth
        for name, p in self.named_parameters():
            if name.endswith(("wo.weight", "down.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # -- parameter accounting ---------------------------------------------

    def parameter_counts(self) -> dict:
        aux_names = ("board_head", "value_head", "policy_head")
        aux = sum(p.numel() for n, p in self.named_parameters()
                  if n.split(".")[0] in aux_names)
        total = sum(p.numel() for p in self.parameters())
        # tied lm_head shares storage with embed, so it is already counted once
        return {"backbone": total - aux, "aux_heads": aux, "total": total,
                "inference_only": total - aux}

    def kv_cache_bytes(self, seq_len: int | None = None, dtype_bytes: int = 2) -> int:
        seq_len = seq_len or self.cfg.max_seq_len
        c = self.cfg
        return c.n_layers * 2 * c.n_kv_heads * c.head_dim * seq_len * dtype_bytes

    def strip_aux_heads(self) -> "ChessReasoner":
        """Drop the training-only heads. Inference is FEN in, prose out."""
        for name in ("board_head", "value_head", "policy_head"):
            if hasattr(self, name):
                delattr(self, name)
        self.cfg.with_aux_heads = False
        return self

    def optimizer_groups(self, weight_decay: float = 0.1) -> list[dict]:
        decay = [p for p in self.parameters() if p.dim() >= 2 and p.requires_grad]
        no_decay = [p for p in self.parameters() if p.dim() < 2 and p.requires_grad]
        return [{"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0}]

    # -- forward -----------------------------------------------------------

    def backbone(self, input_ids: torch.Tensor, caches: list | None = None,
                 pos_offset: int = 0) -> torch.Tensor:
        B, T = input_ids.shape
        if T + pos_offset > self.cfg.max_seq_len:
            raise ValueError(f"sequence of {T + pos_offset} exceeds max_seq_len")
        x = self.embed(input_ids)
        if self._rope is None or self._rope[0].device != x.device:
            self._rope = build_rope_cache(self.cfg.max_seq_len, self.cfg.head_dim,
                                          self.cfg.rope_theta, x.device)
        cos, sin = (t[pos_offset:pos_offset + T] for t in self._rope)
        use_ckpt = self.cfg.grad_checkpointing and self.training and caches is None
        for i, block in enumerate(self.blocks):
            if use_ckpt:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin, None if caches is None else caches[i])
        return self.norm_f(x)

    def forward(self, input_ids: torch.Tensor,
                loss_weights: torch.Tensor | None = None,
                board_pos: torch.Tensor | None = None,
                board_targets: torch.Tensor | None = None,
                value_pos: torch.Tensor | None = None,
                value_targets: torch.Tensor | None = None,
                policy_pos: torch.Tensor | None = None,
                policy_targets: torch.Tensor | None = None,
                aux_scale: float = 1.0) -> dict:
        """One training step's losses.

        ``loss_weights`` is the per-token segment weight from the tokenizer:
        board plane 0.1, prompt 0.0, answer 1.0 by default. Every auxiliary
        target is optional -- Tier-1 data has no engine labels, so only the LM
        loss and the board head are active during the gate run.

        ``aux_scale`` implements the anneal: pass 1.0 early, 0.0 by the end so
        the model never leans on heads it will not have at inference.
        """
        h = self.backbone(input_ids)
        out: dict = {}

        hidden = h[:, :-1].reshape(-1, self.cfg.d_model)
        labels = input_ids[:, 1:].reshape(-1)
        if loss_weights is not None:
            w = loss_weights[:, 1:].reshape(-1).float()
        else:
            w = torch.ones_like(labels, dtype=torch.float32)
        denom = w.sum().clamp(min=1.0)

        chunk = self.cfg.loss_chunk_tokens or hidden.shape[0]
        lm_loss = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, hidden.shape[0], chunk):
            end = start + chunk
            logits = self.lm_head(hidden[start:end]).float()
            per_token = F.cross_entropy(logits, labels[start:end], reduction="none")
            lm_loss = lm_loss + (per_token * w[start:end]).sum()
        lm_loss = lm_loss / denom
        out["lm_loss"] = lm_loss
        total = lm_loss

        if self.cfg.with_aux_heads and board_pos is not None and board_targets is not None:
            hb = _gather(h, board_pos)                       # (N, d_model)
            logits_b = self.board_head(hb).view(-1, self.cfg.board_squares,
                                                self.cfg.board_classes)
            loss_b = F.cross_entropy(
                logits_b.reshape(-1, self.cfg.board_classes).float(),
                board_targets.reshape(-1), ignore_index=IGNORE_INDEX)
            out["board_loss"] = loss_b
            with torch.no_grad():
                mask = board_targets != IGNORE_INDEX
                correct = (logits_b.argmax(-1) == board_targets) & mask
                out["board_acc"] = correct.sum() / mask.sum().clamp(min=1)
            total = total + aux_scale * self.cfg.lambda_board * loss_b

        if self.cfg.with_aux_heads and value_pos is not None and value_targets is not None:
            hv = _gather(h, value_pos)
            logp = F.log_softmax(self.value_head(hv).float(), dim=-1)
            loss_v = -(value_targets * logp).sum(-1).mean()
            out["value_loss"] = loss_v
            total = total + aux_scale * self.cfg.lambda_value * loss_v

        if self.cfg.with_aux_heads and policy_pos is not None and policy_targets is not None:
            hp = _gather(h, policy_pos)
            loss_p = F.cross_entropy(self.policy_head(hp).float(), policy_targets,
                                     ignore_index=IGNORE_INDEX)
            out["policy_loss"] = loss_p
            total = total + aux_scale * self.cfg.lambda_policy * loss_p

        out["loss"] = total
        return out

    # -- sampling ----------------------------------------------------------

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 64,
                 temperature: float = 0.0, eos_id: int | None = None) -> torch.Tensor:
        """Greedy (temperature 0) or sampled continuation, with a KV cache."""
        self.eval()
        caches: list[dict] = [{} for _ in self.blocks]
        h = self.backbone(input_ids, caches=caches)
        out = input_ids
        for _ in range(max_new_tokens):
            logits = self.lm_head(h[:, -1]).float()
            if temperature > 0:
                nxt = torch.multinomial(torch.softmax(logits / temperature, -1), 1)
            else:
                nxt = logits.argmax(-1, keepdim=True)
            out = torch.cat([out, nxt], dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
            h = self.backbone(nxt, caches=caches, pos_offset=out.shape[1] - 1)
        return out


def _gather(h: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Select hidden states at ``pos`` -- an (N, 2) tensor of (batch, time)."""
    return h[pos[:, 0], pos[:, 1]]
