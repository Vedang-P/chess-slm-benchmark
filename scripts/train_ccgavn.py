"""Train Candidate-Conditioned GAVN (CC-GAVN) with resumable HF checkpoints.

CC-GAVN is a compact action-value model for ChessBench.  Unlike the earlier
GAVN, which encodes the board independently and introduces the candidate move
only in its final MLP, CC-GAVN turns each candidate move into a token that
participates in every board-attention layer.  This lets the model condition
piece interactions on the hypothetical move while retaining one forward pass
per legal action, as required by the searchless action-value protocol.

The only augmentation is an exact horizontal chess-board symmetry.  It remaps
board squares, castling rights, en-passant files, and the candidate UCI action
together; return distributions are unchanged.  No frozen MATE or puzzle data
is read by this trainer.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.kaggle_checkpoint import (  # noqa: E402
    UploadTimer, api as make_hf_api, download_latest, upload_checkpoint,
    write_status,
)
from scripts.shard_data import ShardManager  # noqa: E402
from scripts.train_gavn import action_tables, relation_types  # noqa: E402


def candidate_relation_types() -> np.ndarray:
    """Square geometry plus a distinct relation for the candidate action token."""
    square_rel = relation_types()
    out = np.full((65, 65), 7, dtype=np.int64)
    out[:64, :64] = square_rel
    return out


def horizontal_action_map(utils) -> np.ndarray:
    """Action-id map for the exact a-file <-> h-file board reflection."""
    import chess

    mapping = np.empty(utils.NUM_ACTIONS, dtype=np.int64)
    for action in range(utils.NUM_ACTIONS):
        uci = utils.ACTION_TO_MOVE[action]
        def reflect(square: str) -> str:
            return chr(ord("h") - (ord(square[0]) - ord("a"))) + square[1]
        reflected = reflect(uci[:2]) + reflect(uci[2:4]) + uci[4:]
        mapping[action] = utils.MOVE_TO_ACTION[reflected]
    return mapping


def reflect_horizontal(tokens: np.ndarray, actions: np.ndarray,
                       action_map: np.ndarray, selected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the horizontal chess symmetry to selected token/action rows."""
    if not np.any(selected):
        return tokens, actions
    out = tokens.copy()
    mapped_actions = actions.copy()
    rows = np.flatnonzero(selected)
    # Board ordering is rank 8 to rank 1.  Reversing each rank reflects files.
    board = out[rows, 1:65].reshape(-1, 8, 8).copy()
    out[rows, 1:65] = board[:, :, ::-1].reshape(-1, 64)
    # FEN castling slots: K/Q/k/q swap under horizontal reflection.
    castling = out[rows, 65:69].copy()
    swap = {28: 27, 27: 28, 21: 22, 22: 21}  # tokenizer IDs K/Q/k/q
    original_castling = castling.copy()
    for old, new in swap.items():
        castling[original_castling == old] = new
    out[rows, 65:69] = castling
    # En-passant file is the first of its two metadata characters (a..h=10..17).
    ep = out[rows, 69]
    file_mask = (ep >= 10) & (ep <= 17)
    ep[file_mask] = 10 + (7 - (ep[file_mask] - 10))
    out[rows, 69] = ep
    mapped_actions[rows] = action_map[mapped_actions[rows]]
    return out, mapped_actions


def development_mask(tokens: np.ndarray, modulus: int, fold: int) -> np.ndarray:
    """Stable position-level split, independent of candidate action and order."""
    if modulus < 2 or not 0 <= fold < modulus:
        raise ValueError("development split requires --dev-mod >= 2 and a valid --dev-fold")
    # FNV-1a over all FEN tokens.  Using only position tokens prevents actions
    # from the same position crossing the train/development boundary.
    hashed = np.full(len(tokens), np.uint64(1469598103934665603), dtype=np.uint64)
    for column in range(77):
        hashed ^= tokens[:, column].astype(np.uint64)
        hashed *= np.uint64(1099511628211)
    return (hashed % np.uint64(modulus)) == fold


class CandidateConditionedBlock:
    def __init__(self, torch, dim: int, heads: int, relation_count: int,
                 sequence_length: int, dropout: float):
        nn = torch.nn
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff1 = nn.Linear(dim, dim * 2)
        self.ff2 = nn.Linear(dim * 2, dim)
        self.rel = nn.Parameter(torch.zeros(heads, relation_count))
        # Position-specific dynamic bias, conditioned on the board plus move.
        self.dynamic = nn.Linear(dim, heads * 2 * sequence_length)
        self.dropout = nn.Dropout(dropout)


class CCGAVN:
    """Candidate-conditioned geometric action-value network factory."""
    def __new__(cls, torch, dim: int, layers: int, heads: int,
                action_src: np.ndarray, action_dst: np.ndarray,
                action_promo: np.ndarray, relation_index: np.ndarray,
                dropout: float = 0.0):
        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dim, self.heads = dim, heads
                self.board_embed = torch.nn.Embedding(32, dim)
                self.square_embed = torch.nn.Parameter(torch.zeros(64, dim))
                self.global_embed = torch.nn.Embedding(32, dim)
                self.global_pos = torch.nn.Parameter(torch.zeros(13, dim))
                self.promo_embed = torch.nn.Embedding(5, 24)
                self.candidate = torch.nn.Sequential(
                    torch.nn.Linear(dim * 3 + 24, dim), torch.nn.GELU(),
                    torch.nn.LayerNorm(dim), torch.nn.Linear(dim, dim),
                )
                self.blocks = torch.nn.ModuleList()
                relation_count = int(np.max(relation_index)) + 1
                for i in range(layers):
                    block = CandidateConditionedBlock(
                        torch, dim, heads, relation_count, 65, dropout)
                    self.blocks.append(torch.nn.ModuleDict({
                        "norm1": block.norm1, "qkv": block.qkv,
                        "proj": block.proj, "norm2": block.norm2,
                        "ff1": block.ff1, "ff2": block.ff2,
                        "dynamic": block.dynamic, "dropout": block.dropout,
                    }))
                    self.register_parameter(f"rel_{i}", block.rel)
                self.dist_head = torch.nn.Sequential(
                    torch.nn.LayerNorm(dim), torch.nn.Linear(dim, 128))
                self.register_buffer("action_src", torch.tensor(action_src, dtype=torch.long))
                self.register_buffer("action_dst", torch.tensor(action_dst, dtype=torch.long))
                self.register_buffer("action_promo", torch.tensor(action_promo, dtype=torch.long))
                self.register_buffer("relation_index", torch.tensor(relation_index, dtype=torch.long))

            def forward(self, tokens, actions):
                bsz = tokens.size(0)
                x = self.board_embed(tokens[:, 1:65].clamp(0, 31)) + self.square_embed[None]
                context_ids = torch.cat((tokens[:, :1], tokens[:, 65:77]), dim=1).clamp(0, 31)
                context = self.global_embed(context_ids) + self.global_pos[None]
                context = context.mean(dim=1)
                x = x + context[:, None]
                src, dst = self.action_src[actions], self.action_dst[actions]
                batch = torch.arange(bsz, device=tokens.device)
                src_h, dst_h = x[batch, src], x[batch, dst]
                move = self.candidate(torch.cat((
                    src_h, dst_h, src_h * dst_h,
                    self.promo_embed(self.action_promo[actions]),
                ), dim=-1)) + context
                x = torch.cat((x, move[:, None]), dim=1)
                head_dim = dim // heads
                for i, block in enumerate(self.blocks):
                    h = torch.nn.functional.layer_norm(
                        x, (dim,), block["norm1"].weight,
                        block["norm1"].bias, block["norm1"].eps)
                    q, k, v = block["qkv"](h).chunk(3, dim=-1)
                    q = q.view(bsz, 65, heads, head_dim).transpose(1, 2)
                    k = k.view(bsz, 65, heads, head_dim).transpose(1, 2)
                    v = v.view(bsz, 65, heads, head_dim).transpose(1, 2)
                    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
                    scores = scores + getattr(self, f"rel_{i}")[:, self.relation_index].unsqueeze(0)
                    dynamic = block["dynamic"](h.mean(1)).view(bsz, heads, 2, 65)
                    scores = scores + (dynamic[:, :, 0, :, None] + dynamic[:, :, 1, None, :]) / math.sqrt(dim)
                    attn = torch.softmax(scores, dim=-1)
                    y = torch.matmul(block["dropout"](attn), v)
                    x = x + block["dropout"](block["proj"](
                        y.transpose(1, 2).contiguous().view(bsz, 65, dim)))
                    z = torch.nn.functional.layer_norm(
                        x, (dim,), block["norm2"].weight,
                        block["norm2"].bias, block["norm2"].eps)
                    x = x + block["dropout"](block["ff2"](torch.nn.functional.gelu(block["ff1"](z))))
                return self.dist_head(x[:, 64])
        return _Model()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf-shards", required=True)
    p.add_argument("--hf-repo", default="vedangfake/chess-slm-benchmark")
    p.add_argument("--hf-run", default="ccgavn-5m")
    p.add_argument("--outdir", required=True)
    p.add_argument("--sl-repo", default=os.environ.get("SL_REPO", "/kaggle/working/searchless_chess"))
    p.add_argument("--dim", type=int, default=208)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--steps", type=int, default=160000)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--w-dist", type=float, default=1.0)
    p.add_argument("--w-ce", type=float, default=0.25)
    p.add_argument("--reflect-prob", type=float, default=0.5)
    p.add_argument("--dev-mod", type=int, default=100,
                   help="Position-hash denominator for the held-out development split.")
    p.add_argument("--dev-fold", type=int, default=0,
                   help="Held-out position-hash fold; never sampled for updates.")
    p.add_argument("--dev-batch", type=int, default=8192)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-records", type=int, default=0)
    p.add_argument("--ckpt-every", type=int, default=5000)
    p.add_argument("--hf-upload-every", type=float, default=1800)
    p.add_argument("--resume-from-hf", action="store_true")
    return p.parse_args()


def main():
    import torch

    args = parse_args()
    if not (0.0 <= args.reflect_prob <= 1.0):
        raise ValueError("--reflect-prob must be in [0, 1]")
    if args.dim % args.heads:
        raise ValueError("--dim must be divisible by --heads")
    src, dst, promo, utils = action_tables(Path(args.sl_repo))
    action_map = horizontal_action_map(utils)
    bucket_values = np.asarray(utils.get_uniform_buckets_edges_values(128)[1], dtype=np.float32)
    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # A trainer must authenticate before doing work: this guarantees that every
    # checkpoint (including a failure status) can satisfy the HF persistence
    # contract instead of discovering missing credentials after GPU time spent.
    hf_client = make_hf_api(ROOT)
    resume_dir = (download_latest(hf_client, args.hf_repo, args.hf_run, outdir / "hf-resume")
                  if args.resume_from_hf else None)
    if resume_dir is not None:
        config_path = resume_dir / "config.json"
        if not config_path.exists():
            raise ValueError(f"remote checkpoint is missing {config_path.name}: {resume_dir}")
        resume_config = json.loads(config_path.read_text(encoding="utf-8"))
        if resume_config.get("architecture") != "cc-gavn-v1":
            raise ValueError("remote checkpoint is not a CC-GAVN v1 run")
        scientific_fields = ("dim", "layers", "heads", "batch", "lr", "warmup",
                             "temperature", "w_dist", "w_ce", "reflect_prob",
                             "dev_mod", "dev_fold", "seed")
        mismatches = [
            f"{key}: checkpoint={resume_config.get(key)!r}, requested={getattr(args, key)!r}"
            for key in scientific_fields
            if key in resume_config and resume_config.get(key) != getattr(args, key)
        ]
        if mismatches:
            raise ValueError(
                "Refusing to resume with a changed model, data split, or optimization configuration: "
                + "; ".join(mismatches)
                + ". Start a fresh --hf-run for a new experiment.")

    token = hf_client.token
    manager = ShardManager(args.hf_repo, args.hf_shards,
                           Path(os.environ.get("SHARD_CACHE", "/kaggle/tmp/shards")), token=token)
    manager.ensure_downloaded(max_records=args.max_records)
    manager.count_rows(max_records=args.max_records)
    schedule_rng = np.random.default_rng(args.seed)
    schedule = manager.schedule(args.steps, schedule_rng, args.max_records)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_model = CCGAVN(torch, args.dim, args.layers, args.heads, src, dst, promo,
                        candidate_relation_types()).to(device)
    params = sum(p.numel() for p in raw_model.parameters())
    if params > 5_000_000:
        raise ValueError(f"CC-GAVN has {params:,} parameters; exceeds the strict 5M budget")
    model = raw_model
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(raw_model)
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_step = 0
    if resume_dir is not None and (resume_dir / "state.pt").exists():
        state = torch.load(resume_dir / "state.pt", map_location=device, weights_only=False)
        raw_model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state.get("scaler", {}))
        start_step = int(state["step"])
        np_rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(torch.ByteTensor(torch.frombuffer(state["torch_rng"], dtype=torch.uint8)))
        print(f"[resume] step={start_step}", flush=True)
    print(f"[train] device={device} rows={manager.total:,} params={params:,}", flush=True)

    current_tag = None
    tokens = actions = winprob = teacher = train_mask = dev_indices = None
    timer, started = UploadTimer(args.hf_upload_every), time.time()
    timer.mark()
    for step in range(start_step, args.steps):
        tag = schedule[step]
        if tag != current_tag:
            tokens, actions, winprob, teacher = manager.load(tag, args.max_records)
            current_tag = tag
            if teacher.shape != (len(tokens), 128):
                raise ValueError(f"shard {tag}: invalid teacher shape {teacher.shape}")
            dev_mask = development_mask(tokens, args.dev_mod, args.dev_fold)
            train_mask = ~dev_mask
            dev_indices = np.flatnonzero(dev_mask)
            if not np.any(train_mask) or len(dev_indices) == 0:
                raise ValueError(f"shard {tag}: development split left an empty partition")
            print(f"[split] shard={tag} train={train_mask.mean():.3%} dev={dev_mask.mean():.3%}", flush=True)
        idx = np_rng.integers(0, len(tokens), size=args.batch)
        # Rejection sampling preserves uniform sampling over the train split
        # without materializing a potentially multi-gigabyte index array.
        while np.any(train_mask[idx] == 0):
            rejected = train_mask[idx] == 0
            idx[rejected] = np_rng.integers(0, len(tokens), size=int(rejected.sum()))
        bt_np = np.asarray(tokens[idx], dtype=np.int64)
        ba_np = np.asarray(actions[idx], dtype=np.int64)
        reflected = np_rng.random(args.batch) < args.reflect_prob
        bt_np, ba_np = reflect_horizontal(bt_np, ba_np, action_map, reflected)
        teacher_batch = np.asarray(teacher[idx], dtype=np.float32)
        wp_np = np.asarray(winprob[idx], dtype=np.float32)
        bt = torch.as_tensor(bt_np, dtype=torch.long, device=device)
        ba = torch.as_tensor(ba_np, dtype=torch.long, device=device)
        teacher_logp = torch.as_tensor(teacher_batch, dtype=torch.float32, device=device)
        wp = torch.as_tensor(wp_np, dtype=torch.float32, device=device)
        frac = max(0.0, (step - args.warmup) / max(1, args.steps - args.warmup))
        lr = args.lr * min(1.0, (step + 1) / max(1, args.warmup))
        if step >= args.warmup:
            lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * frac))
        optimizer.param_groups[0]["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(bt, ba)
            student_logp = torch.log_softmax(logits / args.temperature, dim=-1)
            teacher_probs = torch.softmax(teacher_logp / args.temperature, dim=-1)
            dist = -(teacher_probs * student_logp).sum(-1).mean() * args.temperature**2
            bucket = torch.clamp(torch.ceil(wp * 128).long() - 1, 0, 127)
            bins = torch.arange(128, device=device, dtype=torch.float32)[None]
            hard = torch.exp(-(bins - bucket[:, None]) ** 2 / (2 * 0.75 ** 2))
            hard = hard / hard.sum(-1, keepdim=True)
            ce = -(hard * student_logp).sum(-1).mean()
            loss = args.w_dist * dist + args.w_ce * ce
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if (step + 1) % 100 == 0:
            print(f"[train] step={step+1}/{args.steps} loss={loss.item():.4f} dist={dist.item():.4f} "
                  f"ce={ce.item():.4f} reflected={reflected.mean():.2f} elapsed={(time.time()-started)/60:.1f}m", flush=True)
        # The wall-clock interval is itself a checkpoint trigger: otherwise a
        # 5,000-step checkpoint can exceed the recoverability window.
        checkpoint_due = ((step + 1) % args.ckpt_every == 0
                          or step + 1 == args.steps or timer.due())
        if checkpoint_due:
            # This is a held-out, position-disjoint development diagnostic. It
            # is never used as a source of gradients and frozen MATE/puzzles
            # remain untouched until a configuration is selected.
            dev_idx = np_rng.choice(dev_indices, size=args.dev_batch, replace=len(dev_indices) < args.dev_batch)
            dev_tokens = torch.as_tensor(np.asarray(tokens[dev_idx], dtype=np.int64), dtype=torch.long, device=device)
            dev_actions = torch.as_tensor(np.asarray(actions[dev_idx], dtype=np.int64), dtype=torch.long, device=device)
            dev_teacher = torch.as_tensor(np.asarray(teacher[dev_idx], dtype=np.float32), dtype=torch.float32, device=device)
            dev_wp = torch.as_tensor(np.asarray(winprob[dev_idx], dtype=np.float32), dtype=torch.float32, device=device)
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                dev_logp = torch.log_softmax(model(dev_tokens, dev_actions) / args.temperature, dim=-1)
                dev_dist = -(torch.softmax(dev_teacher / args.temperature, dim=-1) * dev_logp).sum(-1).mean() * args.temperature**2
                dev_bucket = torch.clamp(torch.ceil(dev_wp * 128).long() - 1, 0, 127)
                dev_bins = torch.arange(128, device=device, dtype=torch.float32)[None]
                dev_hard = torch.exp(-(dev_bins - dev_bucket[:, None]) ** 2 / (2 * 0.75 ** 2))
                dev_hard = dev_hard / dev_hard.sum(-1, keepdim=True)
                dev_ce = -(dev_hard * dev_logp).sum(-1).mean()
                dev_loss = args.w_dist * dev_dist + args.w_ce * dev_ce
            checkpoint = outdir / f"checkpoint-{step+1}"
            checkpoint.mkdir(parents=True, exist_ok=True)
            torch.save({"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(), "step": step + 1,
                        "numpy_rng": np_rng.bit_generator.state,
                        "torch_rng": torch.get_rng_state().cpu().numpy().tobytes()}, checkpoint / "state.pt")
            config = vars(args) | {
                "architecture": "cc-gavn-v1",
                "relation_schema": "v2-live-knight-king-rank-file-diagonal-other+candidate",
                "canonical_decision_head": "distribution_expectation",
                "parameter_count": params,
            }
            (checkpoint / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
            metrics = {"step": step + 1, "train_loss": float(loss.detach()),
                       "dev_loss": float(dev_loss), "dev_dist": float(dev_dist), "dev_ce": float(dev_ce),
                       "dev_tag": str(current_tag)}
            (checkpoint / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            print(f"[dev] step={step+1} loss={dev_loss:.4f} dist={dev_dist:.4f} ce={dev_ce:.4f}", flush=True)
            upload_checkpoint(hf_client, args.hf_repo, outdir, args.hf_run, checkpoint.name)
            timer.mark()
    print(f"[train] done in {(time.time()-started)/3600:.2f}h", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        try:
            import traceback
            def flag(name, default):
                key = f"--{name}"
                return sys.argv[sys.argv.index(key) + 1] if key in sys.argv else default
            write_status(make_hf_api(ROOT), flag("hf-repo", "vedangfake/chess-slm-benchmark"),
                         flag("hf-run", "ccgavn-5m"),
                         type(exc).__name__ + ": " + str(exc) + "\n" + traceback.format_exc()[-4000:])
        except Exception as status_exc:
            print(f"[hf] could not upload failure status: {status_exc}", flush=True)
        raise
