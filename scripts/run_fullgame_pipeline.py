"""Master pipeline runner for the fullgame lucid-commentary track.

Chains games -> commentate -> emit -> train in one resumable process so
the whole multi-hour job runs unattended. Each stage skips if its output
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
        ("commentate", [
            sys.executable, "-u", "scripts/build_commentary_data.py",
            "commentate",
            "--model", "deepseek-v4-flash",
            "--temperature", "0.3",
            "--max-tokens", "800",
            "--max-attempts", "3"]),
        ("emit", [
            sys.executable, "scripts/build_commentary_data.py",
            "emit"]),
        # training runs on Kaggle T4 (16GB, the campaign model) — see
        # notebooks/build_commentary_kernel_notebook.py. Locally the model
        # does not fit this GPU's 6GB, so the train stage is not chained here.
    ]
    for name, cmd in stages:
        t0 = time.time()
        rc = run_stage(name, cmd, ROOT)
        _log(f"stage {name} finished in {(time.time()-t0)/3600:.2f}h rc={rc}")
        if rc != 0:
            _log(f"FATAL: stage {name} failed; aborting")
            sys.exit(1)
    _log("ALL STAGES DONE")


if __name__ == "__main__":
    main()
