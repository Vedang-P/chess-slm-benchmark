"""Train the square-token Geometric Action-Value Network on Kaggle.

This is intentionally a PyTorch implementation so it can use Kaggle's modern
CUDA stack independently of the old JAX/Orbax stack needed to read the
released Ruoss checkpoints. The input contract is the existing student set:
``tokens`` [N,77], ``actions`` [N], optional ``winprob`` [N], and a teacher
matrix [N,128] containing normalized log-probabilities.
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


def relation_types() -> np.ndarray:
    """Return a fixed 64x64 chess relation category matrix."""
    out = np.zeros((64, 64), dtype=np.int64)
    for a in range(64):
        ar, af = divmod(a, 8)
        for b in range(64):
            br, bf = divmod(b, 8)
            dr, df = abs(ar - br), abs(af - bf)
            if a == b:
                rel = 0
            elif ar == br:
                rel = 1
            elif af == bf:
                rel = 2
            elif dr == df:
                rel = 3
            elif (dr, df) in ((1, 2), (2, 1)):
                rel = 4
            elif max(dr, df) == 1:
                rel = 5
            elif dr == 0 or df == 0 or dr == df:
                rel = 6
            else:
                rel = 7
            out[a, b] = rel
    return out


def action_tables(sl_repo: Path):
    sys.path.insert(0, str(sl_repo.parent))
    from searchless_chess.src import utils  # type: ignore
    import chess

    src, dst, promo = [], [], []
    promo_map = {"q": 1, "r": 2, "b": 3, "n": 4}
    for action in range(utils.NUM_ACTIONS):
        uci = utils.ACTION_TO_MOVE[action]
        src_sq = chess.parse_square(uci[:2])
        dst_sq = chess.parse_square(uci[2:4])
        src.append((7 - src_sq // 8) * 8 + src_sq % 8)
        dst.append((7 - dst_sq // 8) * 8 + dst_sq % 8)
        promo.append(promo_map.get(uci[4:].lower(), 0))
    return np.asarray(src), np.asarray(dst), np.asarray(promo), utils


class GeometricBlock:
    def __init__(self, torch, dim: int, heads: int, relation_count: int,
                 dropout: float = 0.0, bias_mode: str = "both"):
        self.nn = torch.nn
        self.norm1 = torch.nn.LayerNorm(dim)
        self.qkv = torch.nn.Linear(dim, dim * 3)
        self.proj = torch.nn.Linear(dim, dim)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.ff1 = torch.nn.Linear(dim, dim * 2)
        self.ff2 = torch.nn.Linear(dim * 2, dim)
        self.rel = torch.nn.Parameter(torch.zeros(heads, relation_count))
        self.dynamic = torch.nn.Linear(dim, heads * 128)
        self.heads = heads
        self.dim = dim
        self.dropout = torch.nn.Dropout(dropout)
        self.bias_mode = bias_mode

    def parameters(self):
        for module in (self.norm1, self.qkv, self.proj, self.norm2,
                       self.ff1, self.ff2, self.dynamic, self.dropout):
            yield from module.parameters()
        yield self.rel

    def __call__(self, x, rel_index):
        import torch
        bsz, n, dim = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        head_dim = dim // self.heads
        q = q.view(bsz, n, self.heads, head_dim).transpose(1, 2)
        k = k.view(bsz, n, self.heads, head_dim).transpose(1, 2)
        v = v.view(bsz, n, self.heads, head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        if self.bias_mode in ("both", "fixed"):
            static = self.rel[:, rel_index].unsqueeze(0)
            scores = scores + static
        if self.bias_mode in ("both", "dynamic"):
            pooled = h.mean(dim=1)
            dynamic = self.dynamic(pooled).view(bsz, self.heads, 2, 64)
            dynamic_bias = dynamic[:, :, 0, :, None] + dynamic[:, :, 1, None, :]
            scores = scores + dynamic_bias / math.sqrt(dim)
        attn = torch.softmax(scores, dim=-1)
        y = torch.matmul(self.dropout(attn), v)
        y = y.transpose(1, 2).contiguous().view(bsz, n, dim)
        x = x + self.dropout(self.proj(y))
        x = x + self.dropout(self.ff2(torch.nn.functional.gelu(self.ff1(self.norm2(x)))))
        return x


class GAVN:
    """nn.Module wrapper kept as a class factory for clean torch imports."""
    def __new__(cls, torch, dim: int, layers: int, heads: int,
                action_src: np.ndarray, action_dst: np.ndarray,
                action_promo: np.ndarray, relation_index: np.ndarray,
                bias_mode: str = "both"):
        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dim = dim
                self.bias_mode = bias_mode
                self.board_embed = torch.nn.Embedding(32, dim)
                self.square_embed = torch.nn.Parameter(torch.zeros(64, dim))
                self.global_embed = torch.nn.Embedding(32, dim)
                self.global_pos = torch.nn.Parameter(torch.zeros(13, dim))
                self.blocks = torch.nn.ModuleList()
                for _ in range(layers):
                    block = GeometricBlock(torch, dim, heads, 8, bias_mode=bias_mode)
                    self.blocks.append(torch.nn.ModuleDict({
                        "norm1": block.norm1, "qkv": block.qkv,
                        "proj": block.proj, "norm2": block.norm2,
                        "ff1": block.ff1, "ff2": block.ff2,
                        "dynamic": block.dynamic, "dropout": block.dropout,
                    }))
                    self.register_parameter(f"rel_{_}", block.rel)
                self.register_buffer("action_src", torch.tensor(action_src, dtype=torch.long))
                self.register_buffer("action_dst", torch.tensor(action_dst, dtype=torch.long))
                self.register_buffer("action_promo", torch.tensor(action_promo, dtype=torch.long))
                self.promo_embed = torch.nn.Embedding(5, 32)
                self.action = torch.nn.Sequential(
                    torch.nn.Linear(dim * 3 + 32, dim), torch.nn.GELU(),
                    torch.nn.LayerNorm(dim), torch.nn.Linear(dim, dim // 2),
                    torch.nn.GELU())
                self.dist_head = torch.nn.Linear(dim // 2, 128)
                self.q_head = torch.nn.Linear(dim // 2, 1)
                self.register_buffer("relation_index", torch.tensor(relation_index))

            def forward(self, tokens, actions):
                x = self.board_embed(tokens[:, 1:65].clamp(0, 31)) + self.square_embed[None]
                context_tokens = torch.cat((tokens[:, :1], tokens[:, 65:77]), dim=1).clamp(0, 31)
                context = self.global_embed(context_tokens) + self.global_pos[None, :13]
                x = x + context.mean(dim=1, keepdim=True)
                for i, block in enumerate(self.blocks):
                    h = torch.nn.functional.layer_norm(x, (dim,),
                                                       block["norm1"].weight,
                                                       block["norm1"].bias,
                                                       block["norm1"].eps)
                    q, k, v = block["qkv"](h).chunk(3, dim=-1)
                    hd = dim // heads
                    q = q.view(-1, 64, heads, hd).transpose(1, 2)
                    k = k.view(-1, 64, heads, hd).transpose(1, 2)
                    v = v.view(-1, 64, heads, hd).transpose(1, 2)
                    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hd)
                    rel = getattr(self, f"rel_{i}")[:, self.relation_index]
                    if self.bias_mode in ("both", "fixed"):
                        scores = scores + rel.unsqueeze(0)
                    if self.bias_mode in ("both", "dynamic"):
                        dyn = block["dynamic"](h.mean(1)).view(-1, heads, 2, 64)
                        scores = scores + (
                            dyn[:, :, 0, :, None] + dyn[:, :, 1, None, :]) / math.sqrt(dim)
                    attn = torch.softmax(scores, -1)
                    y = torch.matmul(block["dropout"](attn), v)
                    x = x + block["dropout"](block["proj"](
                        y.transpose(1, 2).contiguous().view(-1, 64, dim)))
                    z = torch.nn.functional.layer_norm(x, (dim,),
                                                        block["norm2"].weight,
                                                        block["norm2"].bias,
                                                        block["norm2"].eps)
                    x = x + block["dropout"](block["ff2"](
                        torch.nn.functional.gelu(block["ff1"](z))))
                src = self.action_src[actions]
                dst = self.action_dst[actions]
                feat = torch.cat((x[torch.arange(x.size(0)), src],
                                  x[torch.arange(x.size(0)), dst],
                                  x[torch.arange(x.size(0)), src] *
                                  x[torch.arange(x.size(0)), dst],
                                  self.promo_embed(self.action_promo[actions])), dim=-1)
                a = self.action(feat)
                return self.dist_head(a), self.q_head(a).squeeze(-1)
        return _Model()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--teacher", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--sl-repo", default=os.environ.get("SL_REPO", "/kaggle/working/searchless_chess"))
    p.add_argument("--dim", type=int, default=192)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--bias-mode", default="both",
                   choices=["both", "fixed", "dynamic", "none"],
                   help="attention relation bias: static rel + dynamic, or ablations")
    p.add_argument("--w-dist", type=float, default=1.0)
    p.add_argument("--w-q", type=float, default=0.5)
    p.add_argument("--w-ce", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-records", type=int, default=0)
    p.add_argument("--ckpt-every", type=int, default=2000)
    p.add_argument("--hf-repo", default="vedangfake/chess-slm-benchmark")
    p.add_argument("--hf-run", default="gavn-3m")
    p.add_argument("--hf-upload-every", type=float, default=1800)
    p.add_argument("--resume-from-hf", action="store_true")
    return p.parse_args()


def main():
    import torch
    args = parse_args()
    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    hf_client = None
    if args.resume_from_hf:
        hf_client = make_hf_api(ROOT)
        resume_dir = download_latest(hf_client, args.hf_repo, args.hf_run,
                                     outdir / "hf-resume")
    else:
        resume_dir = None

    data = np.load(args.data, mmap_mode="r")
    tokens = data["tokens"]
    actions = data["actions"]
    winprob = np.asarray(data["winprob"], dtype=np.float32) if "winprob" in data else None
    if args.max_records:
        tokens, actions = tokens[:args.max_records], actions[:args.max_records]
        if winprob is not None:
            winprob = winprob[:args.max_records]
    teacher_data = np.load(args.teacher, mmap_mode="r")
    teacher = teacher_data["teacher_logp"] if isinstance(teacher_data, np.lib.npyio.NpzFile) else teacher_data
    teacher = teacher[:len(tokens)]
    if teacher.shape != (len(tokens), 128):
        raise ValueError(f"teacher shape {teacher.shape} != {(len(tokens), 128)}")
    log_norm = np.logaddexp.reduce(np.asarray(teacher[: min(1024, len(teacher))]), axis=1)
    if not np.allclose(log_norm, 0.0, atol=2e-3):
        raise ValueError("teacher matrix is not normalized log-probabilities")

    src, dst, promo, utils = action_tables(Path(args.sl_repo))
    bucket_values = np.asarray(utils.get_uniform_buckets_edges_values(128)[1], dtype=np.float32)
    if winprob is None:
        if len(tokens) > 5_000_000:
            raise ValueError("winprob absent and dataset too large to derive teacher Q eagerly; "
                             "provide an npz with a winprob column")
        teacher_probs = np.exp(np.asarray(teacher))
        winprob = (teacher_probs @ bucket_values).astype(np.float32)
    if len(src) <= actions.max():
        raise ValueError("action id exceeds official action table")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GAVN(torch, args.dim, args.layers, args.heads, src, dst, promo,
                 relation_types(), bias_mode=args.bias_mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=0.01)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    start_step = 0
    if resume_dir is not None and (resume_dir / "state.pt").exists():
        state = torch.load(resume_dir / "state.pt", map_location=device,
                           weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state.get("scaler", {}))
        start_step = int(state["step"])
        np_rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(state["torch_rng"])
        print(f"[resume] step={start_step}", flush=True)
    print(f"[train] device={device} N={len(tokens)} params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    timer = UploadTimer(args.hf_upload_every)
    t0 = time.time()
    for step in range(start_step, args.steps):
        idx = np_rng.integers(0, len(tokens), size=args.batch)
        bt = torch.as_tensor(np.asarray(tokens[idx], dtype=np.int64), dtype=torch.long, device=device)
        ba = torch.as_tensor(np.asarray(actions[idx], dtype=np.int64), dtype=torch.long, device=device)
        tp_batch = np.asarray(teacher[idx], dtype=np.float32)
        tlogp = torch.as_tensor(tp_batch, dtype=torch.float32, device=device)
        tq = torch.as_tensor(np.exp(tp_batch) @ bucket_values, dtype=torch.float32, device=device)
        wp = torch.as_tensor(winprob[idx], dtype=torch.float32, device=device)
        lr = args.lr * min(1.0, (step + 1) / max(1, args.warmup))
        if step >= args.warmup:
            frac = (step - args.warmup) / max(1, args.steps - args.warmup)
            lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * frac))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=device.type == "cuda"):
            logits, q = model(bt, ba)
            temp = args.temperature
            student_logp = torch.log_softmax(logits / temp, dim=-1)
            dist = -(torch.exp(tlogp / temp) * student_logp).sum(-1) * temp * temp
            q_loss = torch.nn.functional.smooth_l1_loss(q, tq)
            idx128 = torch.clamp(torch.round(wp * 128).long(), 0, 127)
            bins = torch.arange(128, device=device, dtype=torch.float32)[None]
            hard = torch.exp(-(bins - idx128[:, None]) ** 2 / (2 * 0.75 ** 2))
            hard = hard / hard.sum(-1, keepdim=True)
            ce = -(hard * student_logp).sum(-1).mean()
            loss = args.w_dist * dist.mean() + args.w_q * q_loss + args.w_ce * ce
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if (step + 1) % 100 == 0:
            print(f"[train] step={step+1}/{args.steps} loss={loss.item():.4f} "
                  f"dist={dist.mean().item():.4f} q={q_loss.item():.4f} "
                  f"elapsed={(time.time()-t0)/60:.1f}m", flush=True)
        if (step + 1) % args.ckpt_every == 0 or step + 1 == args.steps:
            cp = outdir / f"checkpoint-{step+1}"
            cp.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(), "step": step + 1,
                        "numpy_rng": np_rng.bit_generator.state,
                        "torch_rng": torch.get_rng_state()}, cp / "state.pt")
            (cp / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
            (cp / "metrics.json").write_text(json.dumps({"step": step + 1, "loss": float(loss.detach())}, indent=2), encoding="utf-8")
            if hf_client is None and os.environ.get("HF_WRITE_TOKEN"):
                hf_client = make_hf_api(ROOT)
            if hf_client is not None and (timer.due() or step + 1 == args.steps):
                try:
                    upload_checkpoint(hf_client, args.hf_repo, outdir, args.hf_run, cp.name)
                    timer.mark()
                except Exception as exc:
                    print(f"[hf] upload failed; local copy retained: {exc}", flush=True)
    print(f"[train] done in {(time.time()-t0)/3600:.2f}h", flush=True)


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
                         flag("hf-run", "gavn-3m"), type(exc).__name__ + ": " + str(exc) + "\n" + traceback.format_exc()[-4000:])
        except Exception as status_exc:
            print(f"[hf] could not upload failure status: {status_exc}", flush=True)
        raise
