"""Step 6 of the build order: measure real throughput, don't assume it.

    python scripts/calibrate_throughput.py --config base_120m
    python scripts/calibrate_throughput.py --config gate_20m --tokens 2.0e8

Review item R8: the design assumed 25-30% MFU on a T4, which is optimistic for
a model this small -- small models are memory-bandwidth bound and the T4 has
320 GB/s. If real MFU is 15-20%, the 35 h pretraining estimate becomes 50-60 h.
This script sweeps micro-batch sizes, reports achieved TFLOP/s and MFU, finds
what fits in memory, and projects the wall-clock for the full token budget.

Runs on CUDA for the real number; falls back to MPS/CPU so the harness can be
smoke-tested off-GPU (the projection is then meaningless and is labelled so).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chessreasoner import model as M  # noqa: E402

# fp16 tensor-core peak, TFLOP/s
DEVICE_PEAK_TFLOPS = {
    "Tesla T4": 65.0, "T4": 65.0,
    "Tesla P100": 18.7, "P100": 18.7,
    "NVIDIA A100": 312.0, "A100": 312.0,
    "NVIDIA L4": 121.0, "L4": 121.0,
}


def flops_per_token(cfg: M.ChessReasonerConfig, n_params: int, seq_len: int) -> float:
    """Training FLOPs per token: 6N for the parameters, plus attention.

    The attention term matters here -- at 1024 context and d_model 768 it is a
    non-trivial share, and ignoring it inflates the apparent MFU.
    """
    return 6 * n_params + 12 * cfg.n_layers * cfg.d_model * seq_len


def device_peak(device: torch.device) -> tuple[float | None, str]:
    if device.type != "cuda":
        return None, device.type
    name = torch.cuda.get_device_name(device)
    for key, peak in DEVICE_PEAK_TFLOPS.items():
        if key.lower() in name.lower():
            return peak, name
    return None, name


def bench(cfg: M.ChessReasonerConfig, batch: int, seq_len: int, device: torch.device,
          dtype: torch.dtype, steps: int, warmup: int) -> dict | None:
    torch.manual_seed(0)
    net = M.ChessReasoner(cfg).to(device)
    net.train()
    opt = torch.optim.AdamW(net.optimizer_groups(), lr=1e-4, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler(device.type) if dtype == torch.float16 else None

    ids = torch.randint(0, cfg.vocab_size, (batch, seq_len), device=device)
    weights = torch.ones(batch, seq_len, device=device)
    # one board target per sequence, as the packed Tier-1 corpus produces
    board_pos = torch.stack([torch.arange(batch, device=device),
                             torch.full((batch,), 100, device=device)], dim=1)
    board_targets = torch.randint(0, 13, (batch, 64), device=device)

    def step() -> None:
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
            out = net(ids, loss_weights=weights,
                      board_pos=board_pos, board_targets=board_targets)
        if scaler is not None:
            scaler.scale(out["loss"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

    try:
        for _ in range(warmup):
            step()
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(steps):
            step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    except (torch.OutOfMemoryError, RuntimeError) as exc:
        if "out of memory" not in str(exc).lower():
            raise
        return None
    finally:
        peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else float("nan")
        del net, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()

    tokens = steps * batch * seq_len
    n_params = M.ChessReasoner(cfg).parameter_counts()["total"]
    achieved = flops_per_token(cfg, n_params, seq_len) * tokens / elapsed / 1e12
    return {"batch": batch, "s_per_step": elapsed / steps,
            "tokens_per_s": tokens / elapsed, "tflops": achieved, "peak_gb": peak}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="base_120m", choices=["base_120m", "gate_20m"])
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-sizes", type=int, nargs="*", default=[4, 8, 16, 24, 32])
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--tokens", type=float, default=3.2e9,
                    help="total training tokens to project wall-clock for")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.precision]

    cfg = getattr(M, args.config)()
    cfg.max_seq_len = args.seq_len
    cfg.grad_checkpointing = args.grad_checkpointing
    counts = M.ChessReasoner(cfg).parameter_counts()
    peak, dev_name = device_peak(device)

    print(f"config         {args.config}  ({counts['backbone']:,} backbone "
          f"+ {counts['aux_heads']:,} aux)")
    print(f"device         {dev_name}" + (f"   peak {peak} TFLOP/s fp16" if peak else ""))
    print(f"precision      {args.precision}"
          f"{'   grad checkpointing ON' if args.grad_checkpointing else ''}")
    print(f"seq_len        {args.seq_len}")
    if device.type == "cuda" and args.precision == "bf16" \
            and torch.cuda.get_device_capability(device)[0] < 8:
        print("\nWARNING: bf16 requires Ampere (sm_80+). The T4 is sm_75 -- use fp16.")
    print()

    header = f"{'batch':>6} {'tok/step':>9} {'s/step':>8} {'tok/s':>10} {'TFLOP/s':>9}"
    if peak:
        header += f" {'MFU':>7}"
    header += f" {'peak GB':>9}"
    print(header)
    print("-" * len(header))

    best = None
    for batch in args.batch_sizes:
        result = bench(cfg, batch, args.seq_len, device, dtype, args.steps, args.warmup)
        if result is None:
            print(f"{batch:>6} {'':>9} {'':>8} {'':>10} {'':>9}"
                  + (f" {'':>7}" if peak else "") + f" {'OOM':>9}")
            continue
        line = (f"{batch:>6} {batch * args.seq_len:>9,} {result['s_per_step']:>8.3f} "
                f"{result['tokens_per_s']:>10,.0f} {result['tflops']:>9.2f}")
        if peak:
            line += f" {result['tflops'] / peak:>6.1%}"
        line += f" {result['peak_gb']:>9.2f}"
        print(line)
        if best is None or result["tokens_per_s"] > best["tokens_per_s"]:
            best = result

    if best is None:
        print("\nEvery batch size ran out of memory. Try --grad-checkpointing "
              "or a shorter --seq-len.")
        return

    hours = args.tokens / best["tokens_per_s"] / 3600
    print(f"\nbest: batch {best['batch']}  ->  {best['tokens_per_s']:,.0f} tok/s"
          + (f"  ({best['tflops'] / peak:.1%} MFU)" if peak else ""))
    print(f"projected wall-clock for {args.tokens:.2e} tokens: "
          f"{hours:.1f} h  ({hours / 9:.1f} Kaggle sessions at 9 h)")
    if device.type != "cuda":
        print("\nNOTE: not measured on the target GPU -- this projection is a "
              "harness smoke test, not the R8 answer. Re-run on a T4.")
    else:
        print(f"\nDesign doc sec 7.3 estimates 40-60 h. Measured: {hours:.1f} h. "
              "Update the doc with this number.")


if __name__ == "__main__":
    main()
