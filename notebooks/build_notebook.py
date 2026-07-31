"""Single source of truth for the Kaggle notebooks.

    python notebooks/build_notebook.py          # -> kaggle_run.ipynb (full)
    python notebooks/build_notebook.py --check  # -> kaggle_check.ipynb (tiny)

Both notebooks are generated from the same cell list; the check notebook
uses tiny n, short timeouts, and a verdict cell that raises if any stage
fails. NEVER hand-edit the .ipynb files -- regenerate them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = Path(__file__).resolve().parent


def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def _code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


def _notebook(cells) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


STAGE_HELPER = '''
import json, time, shutil
from pathlib import Path

STAGE_LOG = Path("{R}/stage_log.json")
def run_stage(name, args, timeout_min):
    Path("{R}").mkdir(parents=True, exist_ok=True)
    rec = {{"stage": name, "status": "running", "elapsed_min": None}}
    t0 = time.time()
    try:
        res = subprocess.run(args, timeout=timeout_min * 60)
        rec["status"] = "ok" if res.returncode == 0 else "failed"
        rec["returncode"] = res.returncode
    except subprocess.TimeoutExpired:
        rec["status"] = "timeout"
    except Exception as e:
        rec["status"] = "error"
        rec["error"] = str(e)[:200]
    rec["elapsed_min"] = round((time.time() - t0) / 60, 1)
    entries = json.loads(STAGE_LOG.read_text()) if STAGE_LOG.exists() else []
    entries.append(rec)
    STAGE_LOG.write_text(json.dumps(entries, indent=1))
    print(f"stage {{name}}: {{rec['status']}} ({{rec['elapsed_min']}}min)", flush=True)
    return rec["status"]
'''.strip()


def build_cells(check: bool) -> list:
    R = "results_check" if check else "results"
    MODE_TAG = "CHECK MODE (tiny, raises on failure)" if check else "FULL RUN (paper data)"
    SWEEP_FLAG = "--check" if check else ""
    T_ENGINE = 10 if check else 60
    T_SWEEP = 25 if check else 12 * 60  # 12h cap for the full sweep

    cells = [
        _md(f"# Anti-Goal Chess Benchmark @ NeurIPS 2026 — {MODE_TAG}\n\n"
            "Paired win/lose small-model chess study (see README). "
            f"Results land in `{R}/` and are zipped for download.\n"
            "- Positions + exact oracles: committed (`data/positions/`), generated once by `scripts/generate_positions.py`.\n"
            "- Engine + dataset tests gate every run: `scripts/test_engine.py`.\n"
            "- Sweep: `scripts/run_suite.py` (models x tasks x {{win,lose}})."),
        _md("## 1. Clone the repo\n\n"
            "The repo is **private**, so this cell needs a GitHub PAT. On Kaggle, a secret "
            "only reaches the notebook if you ATTACH it: open this notebook -> right-hand "
            "**Secrets** panel -> enable `GITHUB_TOKEN` for this notebook. The token must "
            "have `repo` scope (classic PAT) or Contents:Read (fine-grained)."),
        _code("""import os, subprocess, sys
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "neuro-symbolic-pathfinding"
if not REPO.exists():
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
    print("GITHUB_TOKEN found in environment:", bool(token), flush=True)
    url = "https://github.com/Vedang-P/neuro-symbolic-pathfinding.git"
    if token:
        url = url.replace("https://", f"https://x-access-token:{token}@")
    res = subprocess.run(["git", "clone", "--quiet", url, str(REPO)],
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            "git clone failed. If the repo is private this means the secret never "
            "reached the notebook. Fix: notebook right-hand panel -> Secrets -> "
            "enable GITHUB_TOKEN for THIS notebook (it must contain a PAT with the "
            "repo scope). Stderr: " + res.stderr[-300:]
        )
os.chdir(REPO)
print("cwd:", Path.cwd())"""),
        _md("## 2. Submodule + dependencies"),
        _code("""subprocess.run(["git", "submodule", "update", "--init", "--depth", "1"], check=True, capture_output=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-r", "requirements.txt"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "python-chess"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "--quiet", "-y", "wandb"], check=True)
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")"""),
        _md("## 3. Stage runner (never raises; the verdict cell checks results)"),
        _code(STAGE_HELPER.format(R=R)),
        _md("## 4. Gate: engine + dataset tests"),
        _code(f"""status = run_stage("engine_tests", [sys.executable, "scripts/test_engine.py", "--quick"], {T_ENGINE})
if status != "ok":
    raise RuntimeError("engine tests failed -- see output above")"""),
        _md("## 5. Data validation"
            if not check else
            "## 5. Position generation sanity (tiny, exercises the oracle path)"),
        (_code(f"""status = run_stage("gen_check", [sys.executable, "scripts/generate_positions.py", "--check", "--out", "{R}/positions"], {T_ENGINE})
if status != "ok":
    raise RuntimeError("position generation check failed")""")
         if check else
         _code("""import json
for name in ["cap-legal-8x8", "sm-3x3-win", "sm-3x3-draw", "sm-5x5-win",
             "sm-5x5-draw", "mate1-8x8", "mate1-lichess", "mob-8x8"]:
    recs = json.loads(Path(f"data/positions/{name}.json").read_text())
    assert len(recs) >= 40, f"{name}: expected >=40 positions, got {len(recs)}"
    assert all("win_moves" in r and "lose_moves" in r for r in recs)
print("committed position data OK (8 task sets, oracle fields present)")""")),
        _md("## 6. The chess sweep (models x tasks, paired win/lose)"),
        _code(f"""status = run_stage(
    "chess_sweep",
    [sys.executable, "scripts/run_suite.py", "--output_dir", "{R}/chess"{("", " --check")[check]}],
    {T_SWEEP},
)
print("sweep:", status)"""),
        _md("## 7. Results table"),
        _code("""import pandas as pd
csv_path = Path("{R}/chess/comparison_table.csv")
if csv_path.exists():
    df = pd.read_csv(csv_path)
    display(df)
    print("rows:", len(df))
else:
    print("no comparison table -- sweep did not complete")""".replace("{R}", R)),
        _md("## 8. Verdict (check mode: fail loudly)") if check else _md("## 8. Zip results"),
        (_code(f"""entries = json.loads(STAGE_LOG.read_text()) if STAGE_LOG.exists() else []
fails = [e for e in entries if e["status"] != "ok"]
if fails:
    raise RuntimeError(f"check mode: {{len(fails)}} failed stages: {{[e['stage'] for e in fails]}}")
print("ALL CHECK STAGES PASSED")""")
         if check else
         _code(f"""shutil.make_archive("/kaggle/working/{R}", "zip", root_dir=Path("{R}").resolve())
print("zipped {R}.zip")""")),
        _md("## Notes\n"
            "- **Resume after a died session:** re-run the notebook with a trimmed sweep, e.g. "
            "`run_suite.py --models <remaining> --tasks <remaining> --output_dir results/chess`; "
            "per-run JSONs under `results/chess/*.summary.json` are the source of truth; the CSV is rebuilt at the end.\n"
            "- **Gemma models** need the `HF_TOKEN` Kaggle secret (gated access).\n"
            "- **Timeouts:** full-mode sweep is capped at 12h; typical T4 estimate ~1-2 min/position-cell, "
            "well under a single Kaggle session."),
    ]
    return cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="emit kaggle_check.ipynb")
    args = ap.parse_args()
    out = NB_DIR / ("kaggle_check.ipynb" if args.check else "kaggle_run.ipynb")
    cells = build_cells(check=args.check)
    out.write_text(json.dumps(_notebook(cells), indent=1))
    print(f"wrote {out} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
