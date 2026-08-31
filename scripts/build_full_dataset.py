"""Build the chessbench-full training dataset across multiple Kaggle sessions.

Processes ChessBench train shards one at a time, so a killed kernel (Kaggle
dies at ~12h) never loses more than one shard:

  shard i: download raw bag -> tokenize/parse (tokens/actions/winprob npz)
           -> 9M teacher label (fp16 [N,128]) -> upload both to HF
           -> manifest update (upload after EVERY shard = progress saved)

Resume: the manifest lives in the HF dataset repo under `--hf-run`; a new
session skips every completed shard. Run the same command again after a
kernel death.

Usage (Kaggle kernel, era stack installed by the notebook):
    python3 scripts/build_full_dataset.py \
        --n-shards 8 \
        --sl-repo /kaggle/working/searchless_chess \
        --workdir /kaggle/working/chessbench-build \
        --teacher-checkpoint /kaggle/working/checkpoints/9M/6400000/params_ema \
        --teacher-dim 256 --teacher-layers 8 --teacher-heads 8 \
        --hf-repo vedangfake/chess-slm-benchmark \
        --hf-run chessbench-full-build \
        --resume-from-hf
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.kaggle_checkpoint import (  # noqa: E402
    api as make_hf_api, write_status,
)

GCS_BASE = "https://storage.googleapis.com/searchless_chess/data/train"
TOTAL_SHARDS = 2148


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-shards", type=int, default=8,
                   help="how many train shards to process (00000 .. n-1)")
    p.add_argument("--shard-start", type=int, default=0,
                   help="first shard index to consider")
    p.add_argument("--shard-end", type=int, default=-1,
                   help="last shard index to consider (exclusive; -1 = n-shards)")
    p.add_argument("--sl-repo", default=os.environ.get("SL_REPO", "/kaggle/working/searchless_chess"))
    p.add_argument("--workdir", default="/kaggle/working/chessbench-build")
    p.add_argument("--teacher-checkpoint", required=True,
                   help="9M orbax dir (.../9M/6400000/params_ema)")
    p.add_argument("--teacher-dim", type=int, default=256)
    p.add_argument("--teacher-layers", type=int, default=8)
    p.add_argument("--teacher-heads", type=int, default=8)
    p.add_argument("--teacher-batch", type=int, default=128)
    p.add_argument("--max-shard-records", type=int, default=0,
                   help="cap rows per shard (smoke tests)")
    p.add_argument("--hf-repo", default="vedangfake/chess-slm-benchmark")
    p.add_argument("--hf-run", default="chessbench-full-build")
    p.add_argument("--resume-from-hf", action="store_true")
    p.add_argument("--force-shard", type=int, default=-1,
                   help="process exactly this shard index even if done (repair)")
    return p.parse_args()


def shard_name(i: int) -> str:
    return f"action_value-{i:05d}-of-{TOTAL_SHARDS:05d}_data.bag"


def manifest_path(args):
    return f"{args.hf_run.strip('/')}/manifest.json"


def read_manifest(client, args) -> dict:
    try:
        import huggingface_hub
        data = huggingface_hub.hf_hub_download(
            repo_id=args.hf_repo, filename=manifest_path(args),
            repo_type="dataset", token=client.token)
        return json.loads(Path(data).read_text(encoding="utf-8"))
    except Exception:
        return {"shards": {}}


def write_manifest(client, args, manifest) -> None:
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        tmp = f.name
    try:
        client.upload_file(path_or_fileobj=tmp, path_in_repo=manifest_path(args),
                           repo_id=args.hf_repo, repo_type="dataset")
        print("[manifest] uploaded", flush=True)
    finally:
        os.unlink(tmp)


def download(url: str, dest: Path) -> None:
    """Download with --fail and resume; a truncated bag is rejected and retried."""
    if dest.exists() and validate_bag(dest) > 0:
        print(f"[dl] {dest.name} already present and valid", flush=True)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        dest.unlink(missing_ok=True)
        print(f"[dl] attempt {attempt + 1}: {url}", flush=True)
        r = subprocess.run(["curl", "-sL", "--fail", "--retry", "5", "-C", "-",
                            "-o", str(dest), url])
        if r.returncode == 0 and validate_bag(dest) > 0:
            print(f"[dl] valid ({dest.stat().st_size} bytes)", flush=True)
            return
        print(f"[dl] attempt {attempt + 1} invalid/partial; retrying", flush=True)
    raise RuntimeError(f"download failed for {url}")


def validate_bag(path: Path) -> int:
    """Return the record count from the bagz index tail; 0 if the file is not
    a complete bag. Shard sizes vary a lot (20MB..1.7GB), so there is no fixed
    size threshold: a truncated download fails the index sanity checks."""
    try:
        import struct
        size = path.stat().st_size
        if size < 1024:
            return 0
        with open(path, "rb") as f:
            f.seek(size - 8)
            (index_start,) = struct.unpack("<Q", f.read(8))
        index_size = size - index_start
        if index_size <= 0 or index_size % 8 != 0:
            return 0
        return index_size // 8
    except Exception:
        return 0


def shard_done_on_hf(client, args, tag: str) -> bool:
    """Source of truth: both shard artifacts exist on HF (race-safe for
    parallel kernels; the manifest is advisory only)."""
    try:
        files = set(client.list_repo_files(args.hf_repo, repo_type="dataset"))
    except Exception:
        return False
    pre = f"{args.hf_run.strip('/')}/shard-{tag}/"
    return f"{pre}train_set.npz" in files and f"{pre}teacher_logp.npy" in files


def main() -> None:
    args = parse_args()
    client = make_hf_api(ROOT)
    workdir = Path(args.workdir)
    raw_dir = workdir / "raw"
    shard_root = workdir / "shards"
    manifest = read_manifest(client, args)
    done = set(manifest.get("shards", {}))
    print(f"[build] {len(done)}/{args.n_shards} shards in manifest, resume={args.resume_from_hf}",
          flush=True)

    end = args.shard_end if args.shard_end >= 0 else args.n_shards
    targets = ([args.force_shard] if args.force_shard >= 0
               else range(args.shard_start, min(end, args.n_shards)))
    for i in targets:
        name = shard_name(i)
        tag = f"{i:05d}"
        if args.force_shard < 0 and (tag in done or shard_done_on_hf(client, args, tag)):
            print(f"[build] shard {tag} done (manifest/HF), skip", flush=True)
            continue

        t0 = time.time()
        shard_dir = shard_root / f"shard-{tag}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        raw_bag = raw_dir / name
        npz_out = shard_dir / "train_set.npz"
        teacher_out = shard_dir / "teacher_logp.npy"

        # 1. download raw shard (validated: truncated bags are rejected)
        download(f"{GCS_BASE}/{name}", raw_bag)
        expected_rows = validate_bag(raw_bag)
        print(f"[build] shard {tag}: bag valid, {expected_rows:,} records", flush=True)

        # 2. parse -> tokens/actions/winprob
        cmd = [sys.executable, "scripts/build_student_train_set.py",
               "--bag", str(raw_bag), "--sl-repo", str(args.sl_repo),
               "--out", str(npz_out)]
        if args.max_shard_records:
            cmd += ["--max-records", str(args.max_shard_records)]
        print(f"[build] shard {tag}: parsing", flush=True)
        subprocess.run(cmd, check=True)
        n_parsed = int(np.load(npz_out)["tokens"].shape[0])
        if args.max_shard_records:
            expected_rows = args.max_shard_records
        if n_parsed < expected_rows:
            raise RuntimeError(
                f"shard {tag}: parsed {n_parsed} rows but bag has "
                f"{expected_rows} (truncated or corrupt); aborting shard")
        print(f"[build] shard {tag}: parsed {n_parsed} rows", flush=True)

        # 3. teacher label with the 9M
        cmd = [sys.executable, "scripts/teacher_label.py",
               "--npz", str(npz_out), "--checkpoint", args.teacher_checkpoint,
               "--out", str(teacher_out), "--batch", str(args.teacher_batch),
               "--dim", str(args.teacher_dim),
               "--layers", str(args.teacher_layers),
               "--heads", str(args.teacher_heads),
               "--sl-repo", str(args.sl_repo)]
        if args.max_shard_records:
            cmd += ["--max-records", str(args.max_shard_records)]
        print(f"[build] shard {tag}: teacher labeling (9M)", flush=True)
        subprocess.run(cmd, check=True)

        # 4. validate before upload
        d = np.load(npz_out)
        t = np.load(teacher_out, mmap_mode="r")
        n = d["tokens"].shape[0]
        assert t.shape == (n, 128), (t.shape, n)
        norm = np.logaddexp.reduce(np.asarray(t[:1024]), axis=1)
        assert np.allclose(norm, 0, atol=2e-3), "teacher not normalized"
        rows = int(n)
        print(f"[build] shard {tag}: validated {rows} rows", flush=True)

        # 5. upload shard artifacts + manifest (persistence)
        for fname in ("train_set.npz", "teacher_logp.npy"):
            client.upload_file(
                path_or_fileobj=str(shard_dir / fname),
                path_in_repo=f"{args.hf_run.strip('/')}/shard-{tag}/{fname}",
                repo_id=args.hf_repo, repo_type="dataset")
        manifest.setdefault("shards", {})[tag] = {
            "rows": rows, "elapsed_s": round(time.time() - t0),
        }
        write_manifest(client, args, manifest)
        print(f"[build] shard {tag} DONE ({rows} rows, "
              f"{(time.time()-t0)/60:.1f}m) -> HF", flush=True)

        # 6. disk hygiene: drop the raw bag and local copies
        raw_bag.unlink(missing_ok=True)
        for fname in ("train_set.npz", "teacher_logp.npy"):
            (shard_dir / fname).unlink(missing_ok=True)

    print(f"[build] all target shards complete: {len(done)}/{args.n_shards}",
          flush=True)


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
                         flag("hf-run", "chessbench-full-build"),
                         type(exc).__name__ + ": " + str(exc) + "\n" + traceback.format_exc()[-4000:])
        except Exception as status_exc:
            print(f"[hf] could not upload failure status: {status_exc}", flush=True)
        raise