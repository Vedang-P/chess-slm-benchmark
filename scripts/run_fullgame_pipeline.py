"""Master pipeline runner for the fullgame commentary track.

Chains prehoch -> reconcile -> emit -> train in one resumable process
so the whole 40h+ job runs unattended. Each stage skips if its output
already exists (idempotent), so a crash resumes at the next stage.

    OPENCODE_API_KEY=... nohup python scripts/run_fullgame_pipeline.py \
        > /tmp/opencode/fullgame.log 2>&1 &
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

COMMENTARY = ROOT / "data" / "raw" / "commentary"
LOG_DIR = Path(os.environ.get("FULLGAME_LOG_DIR", "/tmp/opencode"))


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_stage(name: str, cmd: list[str], cwd: Path) -> int:
    _log(f"=== STAGE {name}: {' '.join(cmd)} ===")
    with open(LOG_DIR / f"fullgame-{name}.log", "a") as log:
        res = subprocess.run(cmd, cwd=cwd, stderr=subprocess.STDOUT,
                             stdout=log)
    _log(f"=== STAGE {name} rc={res.returncode} ===")
    return res.returncode


def main() -> None:
    stages = [
        ("prehoch", [
            sys.executable, "-u", "scripts/build_commentary_data.py",
            "prehoch",
            "--model", "deepseek-v4-flash",
            "--temperature", "0.3",
            "--max-tokens", "2000",
            "--max-attempts", "3"]),
        ("reconcile", [
            sys.executable, "-u", "scripts/build_commentary_data.py",
            "reconcile",
            "--model", "deepseek-v4-flash",
            "--temperature", "0.3",
            "--max-tokens", "2000",
            "--max-attempts", "3",
            "--stockfish", str(ROOT / "venv/bin/stockfish")]),
        ("emit", [
            sys.executable, "scripts/build_commentary_data.py",
            "emit", "--agree-ratio", "2"]),
        ("train", [
            sys.executable, "scripts/train_mate_lora.py",
            "--train", str(COMMENTARY / "train.jsonl"),
            "--eval", str(COMMENTARY / "eval.jsonl"),
            "--out", str(ROOT / "results/fullgame-lora-adapter"),
            "--base", str(ROOT / "data/models/gemma-4-E2B-it-text"),
            "--game-mode",
            "--max-seq-len", "3072",
            "--batch", "1",
            "--grad-accum", "16",
            "--epochs", "1"]),
    ]
    for name, cmd in stages:
        t0 = time.time()
        rc = run_stage(name, cmd, ROOT)
        _log(f"stage {name} finished in {(time.time()-t0)/3600:.2f}h rc={rc}")
        if rc != 0 and name != "prehoch":
            _log(f"FATAL: stage {name} failed; aborting")
            sys.exit(1)
        if name == "prehoch" and rc != 0:
            _log("WARN: prehoch rc!=0 but continuing (resume next run)")
    _log("ALL STAGES DONE")


if __name__ == "__main__":
    main()
