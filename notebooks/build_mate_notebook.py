"""Generate kaggle_mate.ipynb — the autonomous MATE thinking run.

    python notebooks/build_mate_notebook.py

Config (measured 2026-08-04): thinking ON, 32768 total / 16384 thinking
budget, forcing prompt, 100 positions (same first-100 as the direct run),
~6h. No fallbacks, no retries; no-answer reasons are classified.
"""
from __future__ import annotations

import json
from pathlib import Path

from build_notebook import _code, _md, _notebook

NB_DIR = Path(__file__).resolve().parent

CLONE_CELL = r'''
import os, shutil, subprocess, sys
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "neuro-symbolic-pathfinding"
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

token = find_token()
url = "https://github.com/Vedang-P/neuro-symbolic-pathfinding.git"
if token:
    url = url.replace("https://", f"https://x-access-token:{token}@")
res = subprocess.run(["git", "clone", "--quiet", url, str(REPO)],
                     capture_output=True, text=True)
if res.returncode != 0:
    raise RuntimeError("clone failed (token not attached?): " + res.stderr[-300:])
os.chdir(REPO)
print("cwd:", Path.cwd())
'''.strip()

DEPS_CELL = r'''
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-U", "-r", "requirements.txt"], check=True)
import torch, transformers
print("transformers", transformers.__version__, "| cuda", torch.cuda.is_available())
'''.strip()

GATE_CELL = r'''
status = subprocess.run([sys.executable, "scripts/test_engine.py"], capture_output=True, text=True)
if status.returncode != 0:
    print(status.stdout[-2000:]); print(status.stderr[-2000:])
    raise RuntimeError("test_engine failed")
print("ALL TESTS PASSED")
'''.strip()

RUN_CELL = r'''
import json, time
from pathlib import Path

out = Path("results/mate-selection")
out.mkdir(parents=True, exist_ok=True)
cmd = [sys.executable, "scripts/run_mate_eval.py",
       "--model", "deepseek-v4-flash",
       "--n", "100",
       "--thinking-budget", "16384",
       "--max_new_tokens", "32768",
       "--force-answer-prompt",
       "--output_dir", "results/mate-selection",
       "--live-push",
       "--resume"]
t0 = time.time()
res = subprocess.run(cmd)
print(f"run exited rc={res.returncode} after {(time.time()-t0)/3600:.1f}h")
'''.strip()

SUMMARY_CELL = r'''
import json, glob, collections
rows = []
for f in glob.glob("results/mate-selection/*.samples.jsonl"):
    for line in open(f):
        if line.strip(): rows.append(json.loads(line))
n = len(rows)
answered = [r for r in rows if r["status"] in ("correct", "wrong")]
na = [r for r in rows if r["status"] == "no_answer"]
print(f"samples: {n}/100 | answered: {len(answered)} | "
      f"accuracy: {sum(r['compliance'] for r in answered)}/{len(answered)}")
print("no-answer reasons:", dict(collections.Counter(r.get("no_answer_reason") for r in na)))
if rows:
    s = json.loads(open("results/mate-selection/deepseek-v4-flash_mate-selection-test_strategy.summary.json").read())
    print(json.dumps(s.get("metrics", {}).get("accuracy", {}), indent=1))
'''.strip()


def build() -> list:
    cells = [
        _md("# MATE Move-Selection: DeepSeek Thinking Run (100 positions)\n\n"
            "Same 100 positions as the direct-mode run, thinking ENABLED at a 32768\n"
            "budget with the forcing prompt. Results land in `results/mate-selection/`\n"
            "and upload to Hugging Face + the live dashboard. ~6h on a Kaggle session.\n\n"
            "Secrets needed: `GITHUB_TOKEN` (clone + live push), `HF_TOKEN` (results\n"
            "upload), `OPENCODE_API_KEY` (deepseek gateway). Attach, SAVE, Restart.\n\n"
            "Honesty contract: no fallbacks, no retries. Empty answers are recorded\n"
            "as no_answer with a reason (truncated / gave_up / unparseable)."),
        _md("## 1. Get the repo"),
        _code(CLONE_CELL),
        _md("## 2. Dependencies"),
        _code(DEPS_CELL),
        _md("## 3. Engine/dataset gate"),
        _code(GATE_CELL),
        _md("## 4. The thinking run (100 positions, ~6h)\n\n"
            "`--resume` skips already-scored positions if the session dies. "
            "`--live-push` streams to the dashboard."),
        _code(RUN_CELL),
        _md("## 5. Results summary"),
        _code(SUMMARY_CELL),
        _md("## Notes\n"
            "- Config: thinking ON, 32768 total / 16384 thinking budget, forcing prompt (measured 4/5 complete at ~210s/pos; answers ~100% correct when they come).\n"
            "- ~6h expected; Kaggle sessions run up to ~9-12h. If it dies, re-run: --resume skips done positions.\n"
            "- Results auto-upload to HF (vedangfake/chess-bench-results) and the live dashboard."),
    ]
    return cells


if __name__ == "__main__":
    out = NB_DIR / "kaggle_mate.ipynb"
    out.write_text(json.dumps(_notebook(build()), indent=1))
    print(f"wrote {out}")
