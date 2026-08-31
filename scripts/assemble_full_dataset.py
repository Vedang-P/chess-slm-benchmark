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

    # 1. get total rows via HF file list (no download)
    from huggingface_hub import HfApi
    api = HfApi(token=client.token)
    files = set(api.list_repo_files(args.hf_repo, repo_type="dataset"))
    rows_total = 0
    # Use manifest if available for row counts, else estimate via file exists
    # For disk-efficiency, we first collect row counts by downloading only train_set.npz headers
    shard_rows = {}
    for i in range(n_shards):
        tag = f"{i:05d}"
        # quick check if shard exists on HF
        if f"{prefix}/shard-{tag}/train_set.npz" not in files:
            raise RuntimeError(f"shard {tag} missing on HF")
        # download only the train_set.npz to get row count, then keep for merge
        dest = shard_dir / "train_set.npz"
        if not dest.exists():
            from huggingface_hub import hf_hub_url
            import urllib.request
            url = hf_hub_url(args.hf_repo, f"{prefix}/shard-{tag}/train_set.npz", repo_type="dataset")
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {client.token}"} if client.token else {})
            with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
                import shutil
                shutil.copyfileobj(r, f)
        d = np.load(dest)
        n = int(d["tokens"].shape[0])
        shard_rows[tag] = n
        rows_total += n
        print(f"[assemble] shard {tag}: {n} rows (counted, {rows_total} total)", flush=True)
    print(f"[assemble] total rows {rows_total:,}", flush=True)
    # 2. create memmaps
    tokens = np.lib.format.open_memmap(str(out / "tokens.mem"), mode="w+", dtype=np.uint8, shape=(rows_total, 77))
    actions = np.lib.format.open_memmap(str(out / "actions.mem"), mode="w+", dtype=np.uint16, shape=(rows_total,))
    winprob = np.lib.format.open_memmap(str(out / "winprob.mem"), mode="w+", dtype=np.float32, shape=(rows_total,))
    teacher = np.lib.format.open_memmap(str(out / "teacher.mem"), mode="w+", dtype=np.float16, shape=(rows_total, 128))
    pos = 0
    for i in range(n_shards):
        tag = f"{i:05d}"
        shard_dir = stage / f"shard-{tag}"
        # ensure teacher present (direct download, no HF cache)
        dest = shard_dir / "teacher_logp.npy"
        if not dest.exists():
            from huggingface_hub import hf_hub_url
            import urllib.request
            url = hf_hub_url(args.hf_repo, f"{prefix}/shard-{tag}/teacher_logp.npy", repo_type="dataset")
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {client.token}"} if client.token else {})
            with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
                import shutil
                shutil.copyfileobj(r, f)
        d = np.load(shard_dir / "train_set.npz")
        n = int(d["tokens"].shape[0])
        tokens[pos:pos+n] = d["tokens"]
        actions[pos:pos+n] = d["actions"]
        winprob[pos:pos+n] = d["winprob"]
        teacher[pos:pos+n] = np.load(shard_dir / "teacher_logp.npy", mmap_mode="r")
        pos += n
        print(f"[assemble] merged shard {tag} -> {pos}/{rows_total}", flush=True)
        # free disk: delete shard files after merge
        import shutil
        shutil.rmtree(shard_dir, ignore_errors=True)
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

    # 4b. upload the assembled files to HF so training kernels can fetch them
    # without a Kaggle dataset dependency
    for fname in ("train_set.npz", "teacher_logp.npy"):
        client.upload_file(
            path_or_fileobj=str(out / fname),
            path_in_repo=f"{prefix}/assembled/{fname}",
            repo_id=args.hf_repo, repo_type="dataset")
        print(f"[assemble] uploaded {fname} to HF", flush=True)
    print(f"[assemble] publish {out} as the Kaggle dataset 'chessbench-full'",
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