#!/usr/bin/env python3
"""Wait for Kaggle GPU quota reset, then relaunch the 6 training kernels.

Polls `kaggle quota` on all 3 accounts every 10 min. When every account has
remaining GPU > 0, pushes all 6 kernels via scripts/launch_trainers.py (which
resumes each run from its latest HF checkpoint) and exits 0.

Progress is logged to logs/quota_relaunch.log. Designed to run detached with
nohup; safe against double-launch (checks kernel RUNNING status first).

Usage:
  nohup python3 scripts/quota_relaunch.py >/dev/null 2>&1 &
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from launch_trainers import env_for_account, TRAINER_CONFIGS  # noqa: E402

ACCOUNTS = ["vedanggggg", "vedangpandeyyy", "softmaxsimp"]
LOG = ROOT / "logs" / "quota_relaunch.log"
POLL_S = 600


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def gpu_remaining(account: str) -> float | None:
    r = subprocess.run([sys.executable, "-m", "kaggle", "quota"],
                       env=env_for_account(account),
                       capture_output=True, text=True, timeout=60)
    for line in (r.stdout or "").splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] == "GPU":
            try:
                return float(parts[2].rstrip("h"))
            except ValueError:
                return None
    return None


def kernel_status(ref: str, account: str) -> str:
    r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "status", ref],
                       env=env_for_account(account),
                       capture_output=True, text=True, timeout=60)
    return (r.stdout or r.stderr).strip()


def main() -> None:
    log("quota_relaunch armed: waiting for GPU quota reset")
    while True:
        remaining = {a: gpu_remaining(a) for a in ACCOUNTS}
        log(f"quota remaining: {remaining}")
        if all(v is not None and v > 0 for v in remaining.values()):
            # safety: don't double-launch if kernels are already running
            running = []
            for owner, slug, *_ in TRAINER_CONFIGS:
                st = kernel_status(f"{owner}/{slug}", owner)
                if "RUNNING" in st:
                    running.append(f"{owner}/{slug}")
            if running:
                log(f"kernels already RUNNING ({running}); nothing to push. exiting.")
                return
            log("quota available on all accounts — relaunching wave")
            r = subprocess.run([sys.executable, str(ROOT / "scripts" / "launch_trainers.py")],
                               capture_output=True, text=True, timeout=600)
            log(f"launch_trainers rc={r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
            if r.returncode == 0:
                log("relaunch done. exiting.")
                return
            log("relaunch failed; will retry next poll")
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
