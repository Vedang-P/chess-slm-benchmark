#!/usr/bin/env python3
"""Permanent watchdog for the local benchmark run.

Runs detached; every 60s it records the run's health to
/tmp/bench_watch.json so it can be inspected at any time, and appends
ALERT lines to /tmp/bench_watch.log when something is wrong (process
died, no progress for N minutes, live feed stale).

    bash scripts/detach.sh python3 scripts/watch_run.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RESULTS = Path(sys.argv[1] if len(sys.argv) > 1 else "results/chess")
TARGET = int(sys.argv[2]) if len(sys.argv) > 2 else 20
PROC_PATTERN = sys.argv[3] if len(sys.argv) > 3 else "run_suite.py --output_dir results/chess"
WATCH = Path(f"/tmp/bench_watch_{RESULTS.name}.json")
LOG = Path(f"/tmp/bench_watch_{RESULTS.name}.log")
STALL_MIN = 10  # no new positions in this many minutes -> alert

def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")

def positions_done() -> int:
    """Count completed samples (one per line in the samples JSONLs) — this
    updates every position, not just at cell end (cells run 15-30 min)."""
    total = 0
    if RESULTS.exists():
        for p in RESULTS.rglob("*.samples.jsonl"):
            try:
                total += sum(1 for _ in p.open())
            except Exception:
                pass
    return total

def live_age_s() -> float:
    live_path = Path("monitor/live.json")
    if not live_path.exists():
        return float("inf")
    try:
        from datetime import datetime

        ts = json.loads(LIVE.read_text()).get("updated_at", "")
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return time.time() - dt.timestamp()
    except Exception:
        return float("inf")

def process_alive() -> bool:
    try:
        out = subprocess.run(["pgrep", "-f", PROC_PATTERN],
                             capture_output=True, text=True)
        return out.returncode == 0
    except Exception:
        return False

def main() -> None:
    last_done = -1
    last_change = time.time()
    while True:
        done = positions_done()
        alive = process_alive()
        age = live_age_s()
        now = time.time()
        if done != last_done:
            last_done = done
            last_change = now
        state = {
            "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "alive": alive,
            "positions_done": done,
            "target_positions": TARGET,
            "live_age_s": round(age, 1) if age != float("inf") else None,
            "stalled_min": round((now - last_change) / 60, 1) if done > 0 else 0,
        }
        WATCH.write_text(json.dumps(state, indent=1))
        if not alive and done < 20:
            log(f"ALERT: run process is DEAD at {done}/20 positions")
        elif done >= 20:
            log(f"RUN COMPLETE: {done} positions")
            sys.exit(0)
        elif done > 0 and (now - last_change) / 60 > STALL_MIN:
            log(f"ALERT: stalled — no new position for {STALL_MIN}+ min (at {done}/20)")
        time.sleep(60)

if __name__ == "__main__":
    main()
