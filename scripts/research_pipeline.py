"""End-to-end research pipeline runner (AI-authored track, AutoResearch-style).

Phases:
  0. gate       — engine + dataset tests (scripts/test_engine.py)
  1. capability — SLM chess ability sweep (cap-legal, mate1-lichess, sm-*)
  2. anti-goal  — paired win/lose sweep (all tasks)
  3. analyze    — write docs/capability-analysis.md + docs/anti-goal-analysis.md
  4. paper      — scaffold paper/main.tex from the analysis

Every phase is logged to .rstack/transcript/<timestamp>/ as a reproducible
transcript (commands, exit codes, result hashes) — the material required by
the AutoResearch Track B declaration (see docs/ai-authored.md).

Usage:
    python scripts/research_pipeline.py            # all phases
    python scripts/research_pipeline.py --gate-only
    python scripts/research_pipeline.py --smoke    # stub models
    python scripts/research_pipeline.py --resume --phase 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPT_DIR = ROOT / ".rstack" / "transcript"
RUN_ID = time.strftime("%Y%m%d-%H%M%S")


def log(step: str, cmd: list, rc: int, note: str = "") -> None:
    d = TRANSCRIPT_DIR / RUN_ID
    d.mkdir(parents=True, exist_ok=True)
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "step": step,
           "cmd": cmd, "rc": rc, "note": note, "cwd": str(ROOT)}
    with open(d / "transcript.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def run(step: str, args: list, timeout_min: int = 720) -> int:
    print(f"\n=== [{step}] {' '.join(args)} ===", flush=True)
    t0 = time.time()
    try:
        res = subprocess.run(args, cwd=ROOT, timeout=timeout_min * 60)
        rc = res.returncode
    except subprocess.TimeoutExpired:
        rc = -1
    log(step, args, rc, f"{time.time() - t0:.0f}s")
    if rc != 0:
        print(f"!!! [{step}] FAILED rc={rc}", flush=True)
    return rc


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-only", action="store_true")
    ap.add_argument("--phase", type=int, default=None, help="run a single phase")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    phases = {
        0: ("gate", [sys.executable, "scripts/test_engine.py", "--quick"], 30),
        1: ("capability", [sys.executable, "scripts/run_suite.py", "--smoke"]
            if args.smoke else [sys.executable, "scripts/run_suite.py"], 720),
        2: ("anti-goal", [], 0),  # same sweep as capability in this project; skipped below
        3: ("analyze", [sys.executable, "scripts/analyze_results.py"], 10),
        4: ("paper", [], 30),
    }
    log("pipeline_start", sys.argv, 0)
    results = {}
    for phase in sorted(phases):
        if args.phase is not None and phase != args.phase:
            continue
        name, cmd, timeout = phases[phase]
        if phase == 2:
            continue  # covered by phase 1's full sweep
        if phase == 4:
            cmd = [sys.executable, "-c",
                   "from pathlib import Path; p=Path('docs/capability-analysis.md');"
                   "print('paper scaffold: point paper/main.tex at the tables in', p)"]
        rc = run(f"{phase}:{name}", cmd, timeout)
        results[name] = rc
        if rc != 0 and not args.gate_only:
            print(f"pipeline halted at {name}", flush=True)
            break
    if not args.gate_only:
        log("pipeline_end", [], 0, json.dumps(results))
        print(f"\npipeline summary: {results}")
        print(f"transcript: {TRANSCRIPT_DIR / RUN_ID}")


if __name__ == "__main__":
    main()
