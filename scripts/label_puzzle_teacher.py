"""Attach the original 9M teacher distributions to a frozen puzzle shard."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.kaggle_checkpoint import api as make_hf_api, write_status  # noqa: E402
from scripts.shard_data import npz_member_shape  # noqa: E402
from scripts.freeze_puzzle_corpus import file_sha256  # noqa: E402


def label(args) -> dict:
    client = make_hf_api(ROOT)
    remote = f"{args.hf_prefix.strip('/')}/shard-{args.shard:05d}"
    files = set(client.list_repo_files(args.hf_repo, repo_type="dataset"))
    if f"{remote}/teacher_logp.npy" in files:
        print(f"[teacher] {remote} already complete", flush=True)
        return {"complete": True, "skipped": True}
    needed = [f"{remote}/train_set.npz", f"{remote}/meta.json"]
    if any(name not in files for name in needed):
        raise ValueError(f"{remote} lacks completed Stockfish labels")
    args.workdir.mkdir(parents=True, exist_ok=True)
    raw_path = Path(hf_hub_download(args.hf_repo, needed[0], repo_type="dataset",
                                    local_dir=str(args.workdir), token=client.token))
    meta_path = Path(hf_hub_download(args.hf_repo, needed[1], repo_type="dataset",
                                     local_dir=str(args.workdir), token=client.token))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not meta.get("complete") or meta.get("identity", {}).get("engine") != "Stockfish 16":
        raise ValueError(f"{remote}: invalid source label identity")
    rows = npz_member_shape(raw_path, "tokens")[0]
    if rows != meta.get("rows") or rows <= 0:
        raise ValueError(f"{remote}: label row count mismatch")
    if file_sha256(raw_path) != meta.get("train_set_sha256"):
        raise ValueError(f"{remote}: raw label SHA256 mismatch")
    output = args.workdir / "teacher_logp.npy"
    command = [sys.executable, str(ROOT / "scripts" / "teacher_label.py"),
               "--npz", str(raw_path), "--checkpoint", str(args.teacher_checkpoint),
               "--out", str(output), "--batch", str(args.batch),
               "--sl-repo", str(args.sl_repo), "--dim", "256",
               "--layers", "8", "--heads", "8"]
    subprocess.run(command, cwd=ROOT, check=True)
    teacher = np.load(output, mmap_mode="r")
    if teacher.shape != (rows, 128) or teacher.dtype != np.float16:
        raise ValueError(f"{remote}: invalid teacher shape/dtype {teacher.shape}/{teacher.dtype}")
    sample = np.asarray(teacher[:min(rows, 1024)], dtype=np.float32)
    if not np.isfinite(sample).all():
        raise ValueError(f"{remote}: non-finite teacher values")
    normalizer = np.logaddexp.reduce(sample, axis=1)
    if not np.allclose(normalizer, 0.0, atol=2e-3):
        raise ValueError(f"{remote}: unnormalized teacher log probabilities")
    client.upload_file(path_or_fileobj=str(output),
                       path_in_repo=f"{remote}/teacher_logp.npy",
                       repo_id=args.hf_repo, repo_type="dataset",
                       commit_message=f"Complete {remote} 9M teacher labels")
    return {"complete": True, "rows": rows, "remote": remote}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--hf-prefix", required=True)
    parser.add_argument("--hf-repo", default="vedangfake/chess-slm-benchmark")
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--sl-repo", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=1024)
    args = parser.parse_args()
    try:
        print(json.dumps(label(args), sort_keys=True), flush=True)
    except Exception as exc:
        try:
            client = make_hf_api(ROOT)
            remote = f"{args.hf_prefix.strip('/')}/shard-{args.shard:05d}"
            write_status(client, args.hf_repo, remote,
                         f"FAILED {type(exc).__name__}: {exc}")
        finally:
            raise


if __name__ == "__main__":
    main()
