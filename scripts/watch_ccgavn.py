#!/usr/bin/env python3
"""Keep the single CC-GAVN run alive across Kaggle session deaths.

User decision (2026-09-17): one model only, ccgavn-5m-seed0. Kaggle kernels
die at ~12h or when the weekly GPU quota is exhausted. This watcher polls the
kernel status and re-pushes notebook 07 through launch_trainers.py whenever
the session has ended but the run has not finished. It exits when the final
checkpoint (step 160000) is present on HF, or when quota is exhausted (it
then logs NEEDS_ACCOUNT_HOP and keeps waiting for the weekly refresh).

Usage:
  nohup python3 scripts/watch_ccgavn.py >/dev/null 2>&1 &   # local loop
  python3 scripts/watch_ccgavn.py --once                    # one poll (CI)
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

OWNER = "vedanggggg"
SLUG = "ccgavn-5m-seed0"
REF = f"{OWNER}/{SLUG}"
FINAL_STEP = 160000
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


def final_checkpoint_on_hf() -> bool:
    from kaggle_checkpoint import hf_token
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token(ROOT))
    files = api.list_repo_files(repo_id="vedangfake/chess-slm-benchmark",
                                repo_type="dataset")
    return f"{SLUG}/checkpoint-{FINAL_STEP}/config.json" in files


def push() -> bool:
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "launch_trainers.py"), "--only", SLUG],
        capture_output=True, text=True, timeout=600)
    out = (r.stdout + r.stderr).strip()
    if r.returncode == 0:
        log(f"[push] OK {REF}: {out.splitlines()[-1] if out else ''}")
        return True
    log(f"[push] FAIL {REF}: {out[-400:]}")
    return False


def iterate(last_push: float) -> tuple[bool, float]:
    try:
        if final_checkpoint_on_hf():
            log(f"DONE: {SLUG}/checkpoint-{FINAL_STEP} exists on HF")
            return True, last_push
        status = kernel_status(REF, OWNER)
        quota = gpu_remaining(OWNER)
        if any(k in status for k in ACTIVE):
            log(f"status={status} quota={quota}h — running")
        elif quota is None or quota <= 0:
            log(f"status={status} quota={quota}h — NEEDS_ACCOUNT_HOP "
                f"(no quota on {OWNER}; awaiting weekly refresh)")
        elif time.time() - last_push < MIN_REPUSH_GAP_S:
            log(f"status={status} quota={quota}h — session ended; "
                f"re-push throttled ({int(time.time()-last_push)}s since last)")
        else:
            log(f"status={status} quota={quota}h — session ended; re-pushing")
            if push():
                last_push = time.time()
            else:
                last_push = time.time() - MIN_REPUSH_GAP_S + 120
    except Exception as exc:  # transient API failures must not kill the watcher
        log(f"poll error: {type(exc).__name__}: {exc}")
    return False, last_push


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true",
                    help="run one poll and exit (GitHub Actions mode)")
    args = ap.parse_args()
    log(f"watching {REF} -> HF {SLUG}/checkpoint-{FINAL_STEP} "
        f"(mode={'once' if args.once else f'poll {POLL_S}s'})")
    last_push = 0.0
    while True:
        done, last_push = iterate(last_push)
        if done or args.once:
            return
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
