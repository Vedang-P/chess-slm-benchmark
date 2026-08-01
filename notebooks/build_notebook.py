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


def build_cells(check: bool, pilot: bool = False) -> list:
    R = "results_check" if check else "results"
    MODE_TAG = "CHECK MODE (tiny, raises on failure)" if check else (
        "REPRESENTATION PILOT (pick the best prompt format)" if pilot else "FULL RUN (paper data)")
    SWEEP_FLAG = "--check" if check else ""
    T_ENGINE = 10 if check else 60
    T_SWEEP = 50 if check else 12 * 60  # check: 6 models x 6 cells incl. model loads

    cells = [
        _md(f"# SLM Chess Capability Benchmark @ NeurIPS 2026 — {MODE_TAG}\n\n"
            "Paired win/lose small-model chess study (see README). "
            f"Results land in `{R}/` and are zipped for download.\n"
            "- Positions + exact oracles: committed (`data/positions/`), generated once by `scripts/generate_positions.py`.\n"
            "- Engine + dataset tests gate every run: `scripts/test_engine.py`.\n"
            "- Sweep: `scripts/run_suite.py` (models x tasks x {{win,lose}})."),
        _md("## 1. Get the repo (GitHub secret method)\n\n"
            "The repo is **private**. On Kaggle, a secret reaches the notebook ONLY if it is "
            "**attached to this notebook** and the kernel is started AFTER attaching:\n\n"
            "1. Notebook editor -> **+ Add** (top-right) -> **Add secret** -> select `GITHUB_TOKEN` "
            "(it must exist under Account settings -> Secrets; value = a classic PAT with `repo` scope).\n"
            "2. **Save** the notebook (Ctrl+S).\n"
            "3. **Kernel -> Restart & Run All** (env vars are injected at kernel start; plain "
            "\"Run All\" does NOT pick up newly attached secrets).\n\n"
            "This cell reads the token from the env var, and falls back to Kaggle's own "
            "`kaggle_secrets` API if the env var is missing."),
        _code("""import os, shutil, subprocess, sys
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "neuro-symbolic-pathfinding"
# ALWAYS start from a fresh clone: re-runs in the same Kaggle session keep the
# old /kaggle/working repo, and stale code has bitten us more than once.
if REPO.exists():
    shutil.rmtree(REPO)

def find_token():
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):
        if os.environ.get(name):
            return os.environ[name]
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("GITHUB_TOKEN")
    except Exception:
        return None

# diagnostic: what token-ish env vars are actually present?
present = sorted(k for k in os.environ if "TOKEN" in k.upper() or "SECRET" in k.upper())
print("token-ish env vars present:", present, flush=True)
token = find_token()
print("GITHUB_TOKEN resolved:", bool(token), flush=True)
url = "https://github.com/Vedang-P/neuro-symbolic-pathfinding.git"
if token:
    url = url.replace("https://", f"https://x-access-token:{token}@")
res = subprocess.run(["git", "clone", "--quiet", url, str(REPO)],
                     capture_output=True, text=True)
if res.returncode != 0:
    raise RuntimeError(
        "git clone failed. The token did not reach this run. Fix order: "
        "(1) + Add -> Add secret -> GITHUB_TOKEN; (2) SAVE the notebook; "
        "(3) Kernel -> Restart & Run All. Then check the diagnostic line above: "
        "if 'token-ish env vars present' is empty, the secret is not attached to "
        "THIS notebook. Stderr: " + res.stderr[-300:]
    )
os.chdir(REPO)
print("cwd:", Path.cwd())"""),
        _md("## 2. Dependencies (forced upgrade; transformers must be >= 5.13 for Gemma 4)"),
        _code("""subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-U", "-r", "requirements.txt"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "--quiet", "-y", "wandb"], check=True)
import transformers
if int(transformers.__version__.split(".")[0]) < 5:
    raise RuntimeError(
        f"transformers {transformers.__version__} is too old for Gemma 4 "
        "(needs >= 5.13). The upgrade failed — check the pip install output above."
    )
print("transformers", transformers.__version__, "(gemma4 support OK)")
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
for name in ["cap-legal-8x8", "bestmove-8x8", "mate1-lichess", "mate2-lichess",
             "sm-3x3-win", "sm-5x5-win", "sm-5x5-draw", "mate1-8x8", "mob-8x8"]:
    recs = json.loads(Path(f"data/positions/{name}.json").read_text())
    assert len(recs) >= 40, f"{name}: expected >=40 positions, got {len(recs)}"
    assert all("win_moves" in r and "lose_moves" in r for r in recs)
print("committed position data OK (9 task sets, oracle fields present)")""")),
        (_md("## 6. Recover completed results (after a died session)\n\n"
             "Every completed cell's summary is backed up to the public live repo by "
             "`--monitor` (check results and full results live in separate namespaces, "
             "so recovery can never mix them). If this is a fresh session, pull them "
             "back so `--resume` can skip what already ran."),
         _code(f"""import json, urllib.request
from pathlib import Path

out = Path("{R}/chess")
out.mkdir(parents=True, exist_ok=True)
base = "https://raw.githubusercontent.com/Vedang-P/chess-bench-live/main"
idx_url = f"{{base}}/{R}/index.json"
if list(out.glob("*.summary.json")):
    print("results already present locally")
else:
    try:
        idx = json.load(urllib.request.urlopen(idx_url, timeout=20))
        for name in idx["files"]:
            url = f"{{base}}/{R}/chess/{{name}}"
            (out / name).write_bytes(urllib.request.urlopen(url, timeout=20).read())
        print(f"recovered {{len(idx['files'])}} completed summaries from the live repo")
    except Exception as e:
        print("nothing to recover (first run or no backup yet):", e)""")),
        _md("## 7. The chess sweep (models x tasks, paired win/lose)\n\n"
            "`--monitor` publishes live progress + per-cell result backups to the public "
            "dashboard repo (monitor/state.json, results/*). `--resume` skips cells whose "
            "summary already exists (recovered in the previous cell)."),
        _code(f"""sweep_args = [sys.executable, "scripts/run_suite.py", "--output_dir", "{R}/chess",
              "--monitor", "--monitor-interval", "120"{", '--config', 'configs/pilot.yaml'" if pilot else ""}]
if {"True" if check else "False"}:
    sweep_args.append("--check")
sweep_args.append("--resume")   # skip cells whose summaries were recovered
status = run_stage("chess_sweep", sweep_args, {T_SWEEP})
print("sweep:", status)"""),
        _md("## 8. Results table"),
        _code("""import pandas as pd
csv_path = Path("{R}/chess/comparison_table.csv")
if csv_path.exists():
    df = pd.read_csv(csv_path)
    display(df)
    print("rows:", len(df))
else:
    print("no comparison table -- sweep did not complete")""".replace("{R}", R)),
        _md("## 9. Verdict (check mode: fail loudly)") if check else _md("## 9. Zip results"),
        (_code(f"""entries = json.loads(STAGE_LOG.read_text()) if STAGE_LOG.exists() else []
fails = [e for e in entries if e["status"] != "ok"]
if fails:
    raise RuntimeError(f"check mode: {{len(fails)}} failed stages: {{[e['stage'] for e in fails]}}")
print("ALL CHECK STAGES PASSED")""")
         if check else
         _code(f"""shutil.make_archive("/kaggle/working/{R}", "zip", root_dir=Path("{R}").resolve())
print("zipped {R}.zip")""")),
        _md("## Notes\n"
            "- **Getting the repo (secret method):** the `GITHUB_TOKEN` secret must be attached "
            "to this notebook (+ Add -> Add secret), the notebook SAVED, and the kernel "
            "RESTARTED -- env vars are injected at kernel start only.\n"
            "- **Resume after a died session:** re-run the notebook with a trimmed sweep, e.g. "
            "`run_suite.py --models <remaining> --tasks <remaining> --output_dir results/chess`; "
            "per-run JSONs under `results/chess/*.summary.json` are the source of truth; the CSV is rebuilt at the end.\n"
            "- **Gemma models** need the `HF_TOKEN` Kaggle secret (gated access).\n"
            "- **Timeouts:** full-mode sweep is capped at 12h; typical T4 estimate ~1-2 min/position-cell, "
            "well under a single Kaggle session."),
    ]
    # flatten any (md, code) pairs added as single elements
    flat = []
    for c in cells:
        if isinstance(c, tuple):
            flat.extend(c)
        else:
            flat.append(c)
    return flat


def build_debug_cells() -> list:
    """kaggle_debug.ipynb: run ONE gemma cell with full visibility — the exact
    rendered prompt, then the model's output streamed live token by token."""
    cells = [
        _md("# Debug: Gemma 4 prompt + live chain of thought\n\n"
            "Runs a single gemma4-e2b mate1-lichess position with:\n"
            "- the EXACT rendered prompt printed (special tokens visible)\n"
            "- the model's output **streamed live** as tokens are generated\n"
            "- the parsed verdict after generation\n\n"
            "Watch: does the model answer directly (thinking disabled worked), or does it "
            "still emit `<|channel>thought` reasoning and get cut off at the token budget?"),
        _md("## 1. Fresh clone + deps"),
        _code("""import os, shutil, subprocess, sys
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "neuro-symbolic-pathfinding"
if REPO.exists():
    shutil.rmtree(REPO)
token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
url = "https://github.com/Vedang-P/neuro-symbolic-pathfinding.git"
if token:
    url = url.replace("https://", f"https://x-access-token:{token}@")
subprocess.run(["git", "clone", "--quiet", url, str(REPO)], check=True)
os.chdir(REPO)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-U", "-r", "requirements.txt"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "--quiet", "-y", "wandb"], check=True)
import transformers
print("transformers", transformers.__version__)"""),
        _md("## 2. One gemma cell, fully visible"),
        _code("""import subprocess, sys
subprocess.run([sys.executable, "scripts/run_chess.py",
                "--model", "gemma4-e2b", "--task", "mate1-lichess",
                "--prompt-variant", "grid", "--n", "1",
                "--max_new_tokens", "1024",
                "--output_dir", "results_debug/chess",
                "--verbose", "--stream"], check=False)"""),
        _md("## What to look for\n"
            "- **Prompt block**: starts `<bos><|turn>user...<turn|>\\n<|turn>model\\n` — "
            "no thinking tokens (`<|think>`, `<|channel>`).\n"
            "- **Streaming output**: if the first tokens are plain text answering with "
            "`MOVE:`, the fix worked. If you see `<|channel>thought` or long reasoning "
            "followed by truncation at ~1024 tokens, paste the output here — we'll raise "
            "the budget and/or tighten the prompt."),
    ]
    return cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="emit kaggle_check.ipynb")
    ap.add_argument("--pilot", action="store_true", help="emit kaggle_pilot.ipynb")
    ap.add_argument("--debug", action="store_true", help="emit kaggle_debug.ipynb")
    args = ap.parse_args()
    if args.debug:
        out = NB_DIR / "kaggle_debug.ipynb"
        cells = build_debug_cells()
    elif args.pilot:
        out = NB_DIR / "kaggle_pilot.ipynb"
        cells = build_cells(check=False, pilot=True)
    else:
        out = NB_DIR / ("kaggle_check.ipynb" if args.check else "kaggle_run.ipynb")
        cells = build_cells(check=args.check)
    out.write_text(json.dumps(_notebook(cells), indent=1))
    print(f"wrote {out} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
