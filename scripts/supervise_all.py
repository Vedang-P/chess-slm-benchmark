"""Local supervisor: run every keep-alive check every 15 minutes.

Each check is status-gated and idempotent (they only push when a kernel is
needed and not active). Complements the GitHub Actions crons, which fire
unreliably. Safe to leave running or kill at any time.
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "supervise_all.log"
CHECKS = [
    ("build-2b", ROOT / "scripts" / "ensure_build_2b.py"),
    ("1b", ROOT / "scripts" / "watch_1b.py"),
    ("eval-320k", ROOT / "scripts" / "ensure_eval_320k.py"),
    ("monitor", ROOT / "scripts" / "push_monitor.py"),
]


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def main() -> None:
    log("supervisor started (15 min interval)")
    while True:
        for name, script in CHECKS:
            try:
                r = subprocess.run([sys.executable, str(script)], capture_output=True,
                                   text=True, timeout=900)
                out = (r.stdout + r.stderr).strip().splitlines()
                log(f"{name}: {(out[-1] if out else 'no output')[:200]}")
            except Exception as exc:
                log(f"{name}: error {type(exc).__name__}: {exc}")
        time.sleep(900)


if __name__ == "__main__":
    main()
