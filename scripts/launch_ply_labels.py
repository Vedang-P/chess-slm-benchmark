"""Smoke-test or start the later-ply labeling fleet by hand.

The kernels check out the pinned commit from GitHub, so the commit must already be
pushed. In normal operation the cont1 supervisor tick (scripts/watch_cont.py) starts
and restarts the workers; ``--fleet`` runs that same tick once from this machine.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.cont_config import ACCOUNTS, HF_REPO  # noqa: E402
from scripts.kaggle_checkpoint import api as make_hf_api  # noqa: E402
from scripts.kaggle_pool import push_kernel  # noqa: E402
from scripts.watch_ply_labels import MARKER, TEMPLATE, tick, worker_config  # noqa: E402

SMOKE_PREFIX = "puzzle-ply-smoke-v1"
SMOKE_SLUG = "ccgavn-ply-label-smoke"


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def pushed_head() -> str:
    head = git("rev-parse", "HEAD")
    subprocess.run(["git", "-C", str(ROOT), "fetch", "-q", "origin"], check=True)
    if not git("branch", "-r", "--contains", head):
        raise SystemExit(f"commit {head[:12]} is not on GitHub; push it first")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("working tree has uncommitted changes; the kernel would not see them")
    return head


def main() -> None:
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true",
                      help="one kernel: engine timing, 1-vs-4 engine label gap, a few roots")
    mode.add_argument("--fleet", action="store_true", help="run one supervisor pass for the fleet")
    p.add_argument("--account", default=ACCOUNTS[0], choices=ACCOUNTS)
    p.add_argument("--smoke-roots", type=int, default=40)
    p.add_argument("--bench-roots", type=int, default=60)
    p.add_argument("--check-only", action="store_true", help="--fleet: decide, launch nothing")
    args = p.parse_args()
    commit = pushed_head()
    if args.smoke:
        config = worker_config(0, commit, phase="smoke") | {
            "data_prefix": SMOKE_PREFIX, "smoke_roots": args.smoke_roots,
            "bench_roots": args.bench_roots, "worker_count": 1}
        print(f"[launch] {args.account}/{SMOKE_SLUG} commit={commit[:12]}", flush=True)
        push_kernel(args.account, SMOKE_SLUG, TEMPLATE, MARKER, config, gpu=False)
        return
    client = make_hf_api(ROOT)
    files = set(client.list_repo_files(HF_REPO, repo_type="dataset"))
    print(json.dumps(tick(client, files, commit, check_only=args.check_only), indent=2,
                     sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
