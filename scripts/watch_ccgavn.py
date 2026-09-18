#!/usr/bin/env python3
"""Keep the CC-GAVN run alive across Kaggle session deaths.

User decision (2026-09-17): one model only, CC-GAVN (seed 0 on
  vedangpandeyyy). seed 1 on shoumikmitra was stopped the same day: the
  account was reassigned to space-ablation by user decision.
Kaggle kernels die at ~12h or when the weekly GPU quota is exhausted. This
watcher polls each kernel and re-pushes notebook 07 through launch_trainers.py
whenever a session has ended but its run has not finished. It exits when all
runs have written checkpoint-160000 to HF.

Usage:
  nohup python3 scripts/watch_ccgavn.py >/dev/null 2>&1 &   # local loop
  python3 scripts/watch_ccgavn.py --once                    # one pass (CI)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from launch_trainers import env_for_account  # noqa: E402
from quota_relaunch import gpu_remaining, kernel_status  # noqa: E402

WATCHERS = [
    ("vedangpandeyyy", "ccgavn-5m-seed0"),
]
FINAL_STEP = 320000
LOG = ROOT / "logs" / "watch_ccgavn.log"
POLL_S = 600
MIN_REPUSH_GAP_S = 900
ACTIVE = ("RUNNING", "QUEUED", "PENDING", "STARTING")


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass  # CI checkouts may not carry the log directory


def final_checkpoint_on_hf(slug: str) -> bool:
    from kaggle_checkpoint import hf_token
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token(ROOT))
    files = api.list_repo_files(repo_id="vedangfake/chess-slm-benchmark",
                                repo_type="dataset")
    return f"{slug}/checkpoint-{FINAL_STEP}/config.json" in files


def push(slug: str) -> bool:
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "launch_trainers.py"), "--only", slug],
        capture_output=True, text=True, timeout=600)
    out = (r.stdout + r.stderr).strip()
    if r.returncode == 0:
        log(f"[push] OK {slug}: {out.splitlines()[-1] if out else ''}")
        return True
    log(f"[push] FAIL {slug}: {out[-400:]}")
    return False


def iterate(last_push: dict) -> tuple[bool, dict]:
    all_done = True
    for owner, slug in WATCHERS:
        ref = f"{owner}/{slug}"
        try:
            if final_checkpoint_on_hf(slug):
                log(f"DONE: {slug}/checkpoint-{FINAL_STEP} exists on HF")
                continue
            all_done = False
            status = kernel_status(ref, owner)
            quota = gpu_remaining(owner)
            if any(k in status for k in ACTIVE):
                log(f"{ref}: status={status} quota={quota}h — running")
            elif quota is None or quota <= 0:
                log(f"{ref}: status={status} quota={quota}h — NEEDS_ACCOUNT_HOP "
                    f"(no quota on {owner}; awaiting weekly refresh)")
            elif time.time() - last_push.get(slug, 0.0) < MIN_REPUSH_GAP_S:
                log(f"{ref}: status={status} quota={quota}h — session ended; "
                    f"re-push throttled ({int(time.time()-last_push.get(slug, 0.0))}s)")
            else:
                log(f"{ref}: status={status} quota={quota}h — session ended; re-pushing")
                if push(slug):
                    last_push[slug] = time.time()
                else:
                    last_push[slug] = time.time() - MIN_REPUSH_GAP_S + 120
        except Exception as exc:  # transient API failures must not kill the watcher
            all_done = False
            log(f"{ref}: poll error: {type(exc).__name__}: {exc}")
    return all_done, last_push


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true",
                    help="run one pass and exit (GitHub Actions mode)")
    args = ap.parse_args()
    names = ", ".join(f"{o}/{s}" for o, s in WATCHERS)
    log(f"watching [{names}] -> checkpoint-{FINAL_STEP} "
        f"(mode={'once' if args.once else f'poll {POLL_S}s'})")
    last_push: dict = {}
    while True:
        all_done, last_push = iterate(last_push)
        if all_done or args.once:
            return
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
