"""Assemble the final chessbench-full Kaggle dataset from HF shard pieces.

Merges the per-shard artifacts uploaded by build_full_dataset.py into the
runbook contract:

  train_set.npz      keys: tokens [N,77] uint8, actions [N] uint16,
                           winprob [N] float32
  teacher_logp.npy   fp16 [N,128] (9M teacher, normalized log-probs)

Runs on a Kaggle kernel (1TB disk) or on a laptop with >= 135GB free.
Streams via memmap: never holds more than one shard in RAM.

Usage:
    python3 scripts/assemble_full_dataset.py \
        --n-shards 8 \
        --hf-repo vedangfake/chess-slm-benchmark \
        --hf-run chessbench-full-build \
        --out /kaggle/working/chessbench-full

Then publish /kaggle/working/chessbench-full as the Kaggle Dataset
`chessbench-full` (public so all three accounts can mount it).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.kaggle_checkpoint import api as make_hf_api  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-shards", type=int, default=8)
    p.add_argument("--hf-repo", default="vedangfake/chess-slm-benchmark")
    p.add_argument("--hf-run", default="chessbench-full-build")
    p.add_argument("--out", default="/kaggle/working/chessbench-full")
    p.add_argument("--keep-shards", action="store_true",
                   help="do not delete downloaded shard pieces")
    return p.parse_args()


def write_array_stream(member, arr):
    """Write an ndarray into an open binary member without materializing it."""
    np.lib.format.write_array(member, arr)


def main():
    import zipfile
    import io
    from huggingface_hub import hf_hub_download

    args = parse_args()
    client = make_hf_api(ROOT)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stage = out / ".shard-stage"
    stage.mkdir(parents=True, exist_ok=True)

    prefix = args.hf_run.strip("/")
    n_shards = args.n_shards

    # 1. download per-shard pieces from HF (skip existing)
    rows_total = 0
    for i in range(n_shards):
        tag = f"{i:05d}"
        shard_dir = stage / f"shard-{tag}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        for fname in ("train_set.npz", "teacher_logp.npy"):
            dest = shard_dir / fname
            if dest.exists() and dest.stat().st_size > 0:
                continue
            hf_hub_download(
                repo_id=args.hf_repo, repo_type="dataset", token=client.token,
                filename=f"{prefix}/shard-{tag}/{fname}",
                local_dir=str(shard_dir))
            # hf_hub_download writes into a cache; move the file into place
            cached = shard_dir / f"{prefix}" / f"shard-{tag}" / fname
            if cached.exists():
                cached.replace(dest)
        d = np.load(shard_dir / "train_set.npz")
        rows_total += int(d["tokens"].shape[0])
        print(f"[assemble] shard {tag}: {d['tokens'].shape[0]} rows", flush=True)

    # 2. memmap merge
    tokens = np.lib.format.open_memmap(str(out / "tokens.mem"), mode="w+",
                                       dtype=np.uint8, shape=(rows_total, 77))
    actions = np.lib.format.open_memmap(str(out / "actions.mem"), mode="w+",
                                        dtype=np.uint16, shape=(rows_total,))
    winprob = np.lib.format.open_memmap(str(out / "winprob.mem"), mode="w+",
                                        dtype=np.float32, shape=(rows_total,))
    teacher = np.lib.format.open_memmap(str(out / "teacher.mem"), mode="w+",
                                        dtype=np.float16, shape=(rows_total, 128))
    pos = 0
    for i in range(n_shards):
        tag = f"{i:05d}"
        shard_dir = stage / f"shard-{tag}"
        d = np.load(shard_dir / "train_set.npz")
        n = int(d["tokens"].shape[0])
        tokens[pos:pos + n] = d["tokens"]
        actions[pos:pos + n] = d["actions"]
        winprob[pos:pos + n] = d["winprob"]
        teacher[pos:pos + n] = np.load(shard_dir / "teacher_logp.npy", mmap_mode="r")
        pos += n
        print(f"[assemble] merged shard {tag} -> {pos}/{rows_total}", flush=True)

    # 3. pack into the runbook contract files (streamed, no big RAM copies)
    with zipfile.ZipFile(out / "train_set.npz", "w",
                         compression=zipfile.ZIP_STORED) as zf:
        for key, arr in (("tokens", tokens), ("actions", actions),
                         ("winprob", winprob)):
            with zf.open(f"{key}.npy", "w", force_zip64=True) as member:
                write_array_stream(member, arr)
    with open(out / "teacher_logp.npy", "wb") as f:
        np.lib.format.write_array(f, teacher)

    # 4. verify
    d = np.load(out / "train_set.npz")
    t = np.load(out / "teacher_logp.npy", mmap_mode="r")
    assert d["tokens"].shape == (rows_total, 77)
    assert t.shape == (rows_total, 128)
    norm = np.logaddexp.reduce(np.asarray(t[:1024]), axis=1)
    assert np.allclose(norm, 0, atol=2e-3), "teacher not normalized"
    sizes = {p.name: p.stat().st_size / 2**30 for p in out.iterdir() if p.is_file()}
    print(f"[assemble] DONE: {rows_total:,} rows; {json.dumps(sizes, indent=2)}",
          flush=True)
    print(f"[assemble] publish {out} as the Kaggle dataset 'chessbench-full'",
          flush=True)

    if not args.keep_shards:
        for p in stage.rglob("*"):
            if p.is_file():
                p.unlink()
        for p in (out / "tokens.mem", out / "actions.mem", out / "winprob.mem",
                  out / "teacher.mem"):
            p.unlink(missing_ok=True)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"[assemble] wall {(time.time()-t0)/60:.1f}m", flush=True)