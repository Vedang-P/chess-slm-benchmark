#!/usr/bin/env python3
"""Push CPU eval kernels to Kaggle (1 per trained run, no GPU quota needed).

Each kernel pulls the run's LATEST HF checkpoint and runs the full frozen
protocol: all 4 MATE sets + official 10K puzzles. CPU-only (enable_gpu=False)
so the exhausted GPU quota doesn't block it.

Usage:
  python3 scripts/launch_evals.py --status
  python3 scripts/launch_evals.py
  python3 scripts/launch_evals.py --only eval-gavn-3m-seed0
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from launch_trainers import env_for_account  # noqa: E402

EVAL_CONFIGS = [
    # (owner, slug, hf RUN_ID prefix)
    ("vedanggggg", "eval-gavn-3m-seed0", "account1-gavn-3m-seed0"),
    ("vedanggggg", "eval-baseline-5m-seed0", "account1-baseline-5m-seed0"),
    ("vedangpandeyyy", "eval-gavn-5m-seed0", "account2-gavn-5m-seed0"),
    ("vedangpandeyyy", "eval-gavn-3m-seed1", "account2-gavn-3m-seed1"),
    ("softmaxsimp", "eval-gavn-5m-geometry", "account3-gavn-5m-geometry"),
    ("softmaxsimp", "eval-gavn-5m-loss", "account3-gavn-5m-loss"),
]

TEMPLATE = ROOT / "notebooks/03_kaggle_eval_frontier.ipynb"
RUN_ID_LINE = "RUN_ID = 'account1-gavn-3m-seed0'"


def kaggle(args: list[str], account: str, timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "kaggle", *args],
                          capture_output=True, text=True,
                          env=env_for_account(account), timeout=timeout)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    configs = [c for c in EVAL_CONFIGS if not args.only or c[1] == args.only]
    if args.status:
        for owner, slug, _ in configs:
            r = kaggle(["kernels", "status", f"{owner}/{slug}"], owner)
            print(f"{owner}/{slug:26s} {r.stdout.strip() or r.stderr.strip()}")
        return

    failures = []
    for owner, slug, run_id in configs:
        push_dir = Path(tempfile.mkdtemp(prefix=f"kaggle_push_{slug}_"))
        txt = TEMPLATE.read_text()
        old = RUN_ID_LINE
        assert old in txt, f"template missing {old!r}"
        txt = txt.replace(old, f"RUN_ID = '{run_id}'")
        (push_dir / f"{slug}.ipynb").write_text(txt)
        meta = {
            "id": f"{owner}/{slug}",
            "title": slug,
            "code_file": f"{slug}.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": False,   # CPU eval: no GPU quota consumed
            "enable_tpu": False,
            "enable_internet": True,
            "dataset_sources": [f"{owner}/chess-creds"],
            "competition_sources": [],
            "kernel_sources": [],
        }
        (push_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        print(f"[push] {owner}/{slug} (eval {run_id}) ...", flush=True)
        r = kaggle(["kernels", "push", "-p", str(push_dir)], owner)
        out = (r.stdout + r.stderr).strip()
        if r.returncode == 0:
            print(f"[push] OK  {owner}/{slug}: {out.splitlines()[-1] if out else ''}")
        else:
            print(f"[push] FAIL {owner}/{slug}: {out[:600]}")
            failures.append(slug)
        shutil.rmtree(push_dir, ignore_errors=True)
        time.sleep(15)
    if failures:
        sys.exit(f"push failures: {failures}")
    print("all eval kernels pushed")


if __name__ == "__main__":
    main()
