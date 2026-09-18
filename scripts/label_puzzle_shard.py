"""Label puzzle-curriculum rows with the 9M teacher and publish a shard to HF.

Runs inside the era venv (jax 0.4.35/orbax 0.5.5); called by the Kaggle
labeling kernels after scripts/teacher_label.py produces teacher log-probs.

Steps:
  1. teacher_label.py  -> teacher_logp.npy (fp16 [N,128])
  2. winprob = exp(logp) @ bucket_values  (fp32), saved into train_set.npz
  3. upload train_set.npz + teacher_logp.npy to HF under <prefix>/<name>/
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True, help="npz with tokens/actions")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--teacher-checkpoint", required=True)
    ap.add_argument("--sl-repo", required=True)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--hf-repo", default="vedangfake/chess-slm-benchmark")
    ap.add_argument("--hf-prefix", required=True, help="e.g. chessbench-full-build/shard-P000")
    ap.add_argument("--token-file", default="", help="file containing the HF write token")
    args = ap.parse_args()

    import glob
    import json

    import numpy as np

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    teacher_path = out / "teacher_logp.npy"
    if not teacher_path.exists():
        cmd = [sys.executable, str(ROOT / "scripts" / "teacher_label.py"),
               "--npz", args.rows, "--checkpoint", args.teacher_checkpoint,
               "--out", str(teacher_path), "--batch", str(args.batch),
               "--sl-repo", args.sl_repo, "--dim", str(args.dim),
               "--layers", str(args.layers), "--heads", str(args.heads)]
        print("[label] " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    token = None
    if args.token_file and Path(args.token_file).exists():
        token = Path(args.token_file).read_text().strip()
    else:
        hits = sorted(glob.glob("/kaggle/input/**/hf_token.txt", recursive=True))
        if hits:
            token = Path(hits[0]).read_text().strip()
    if not token:
        token = __import__("os").environ.get("HF_WRITE_TOKEN", "")
    if not token:
        raise RuntimeError("no HF token found")

    sys.path.insert(0, str(Path(args.sl_repo).parent))
    from searchless_chess.src import utils  # type: ignore
    bucket_values = np.asarray(utils.get_uniform_buckets_edges_values(128)[1], dtype=np.float32)

    from huggingface_hub import HfApi
    api = HfApi(token=token)

    d = np.load(args.rows)
    tokens, actions = d["tokens"], d["actions"]
    teacher = np.load(teacher_path, mmap_mode="r")
    if teacher.shape != (len(tokens), 128):
        raise ValueError(f"teacher shape {teacher.shape} != {(len(tokens), 128)}")
    norm = np.logaddexp.reduce(np.asarray(teacher[:1024], dtype=np.float32), axis=1)
    if not np.allclose(norm, 0.0, atol=2e-3):
        raise ValueError("teacher labels are not normalized")
    winprob = (np.exp(np.asarray(teacher, dtype=np.float32)) @ bucket_values).astype(np.float32)
    np.savez_compressed(out / "train_set.npz", tokens=tokens, actions=actions, winprob=winprob)
    print(f"[label] wrote train_set.npz ({len(tokens)} rows), winprob mean {winprob.mean():.4f}", flush=True)

    meta = {"rows": int(len(tokens)), "dim": args.dim, "layers": args.layers,
            "heads": args.heads, "teacher": args.teacher_checkpoint, "time": time.time()}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))

    for name in ("train_set.npz", "teacher_logp.npy", "meta.json"):
        api.upload_file(path_or_fileobj=str(out / name),
                        path_in_repo=f"{args.hf_prefix}/{name}",
                        repo_id=args.hf_repo, repo_type="dataset")
        print(f"[label] uploaded {args.hf_prefix}/{name}", flush=True)


if __name__ == "__main__":
    main()
