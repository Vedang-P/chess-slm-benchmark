"""Generate the gemma variants aggregator notebook (CPU kernel).

Combines the 6 gemma GPU workers' live state
(monitor/gemma/workers/{noexplain,tactic,both}-w{1,2}.*) into the canonical
monitor/gemma/state.json + history.jsonl + live.json the gemma dashboard
page reads (chess-bench-live.pages.dev/gemma.html).

Runs every 45s for the whole campaign. CPU-only -- does not count against
the 2-concurrent-GPU limit. If this session dies, relaunch it; nothing is
lost (workers keep publishing independently).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from build_notebook import _code, _md, _notebook

NB_DIR = Path(__file__).resolve().parent
ROOT = NB_DIR.parent

CLONE_CELL = r'''
import os, shutil, subprocess, sys
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "chess-slm-benchmark"
if REPO.exists():
    shutil.rmtree(REPO)

def find_token():
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):
        if os.environ.get(name):
            return os.environ[name]
    raise SystemExit("GITHUB_TOKEN missing")

url = "https://github.com/Vedang-P/chess-slm-benchmark.git"
url = url.replace("https://", f"https://x-access-token:{find_token()}@")
res = subprocess.run(["git", "clone", "--quiet", "-b", "main", url, str(REPO)],
                     capture_output=True, text=True)
if res.returncode != 0:
    raise RuntimeError("clone failed (bad/missing GITHUB_TOKEN?): " + res.stderr[-300:])
os.chdir(REPO)
print("cwd:", Path.cwd())
'''.strip()

AGG_CELL = '''
import subprocess, sys
# the 6 gemma workers: noexplain/tactic/both x w1,w2
workers = ",".join(f"{v}-w{n}" for v in ("noexplain", "tactic", "both") for n in (1, 2))
cmd = [sys.executable, "scripts/aggregate_live_state.py",
       "--namespace", "gemma",
       "--run-id", "gemma-variants-campaign",
       "--workers", workers,
       "--interval", "45"]
print("running:", " ".join(cmd))
res = subprocess.run(cmd, stderr=subprocess.STDOUT)
if res.returncode != 0:
    raise RuntimeError(f"aggregator exited rc={res.returncode} -- see output above")
'''.strip()

CELLS = [
    _md("# Gemma variants: Aggregator (CPU)\n\n"
        "Combines the six GPU workers' live state "
        "(monitor/gemma/workers/noexplain-w1/2, tactic-w1/2, both-w1/2) into "
        "the canonical monitor/gemma/state.json + history.jsonl + live.json "
        "the gemma dashboard page reads. Runs every 45s for the whole "
        "campaign. CPU-only. If this session dies, relaunch it; nothing is lost."),
    _md("## 1. Secrets (hardcoded env vars)"),
    _code("import os\n"
          'print("secrets are injected at build time; this cell is a placeholder")'),
    _md("## 2. Get the repo"),
    _code(CLONE_CELL),
    _md("## 3. Run the aggregator (forever)"),
    _code(AGG_CELL),
    _md("## Notes\n"
        "- If a worker stops publishing, the aggregator keeps going with the "
        "remaining workers and shows stale ones in its report.\n"
        "- The dashboard: chess-bench-live.pages.dev/gemma.html"),
]


def main() -> None:
    env = {}
    for line in (ROOT / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k] = v.strip().strip('"').strip("'")

    nb = _notebook(CELLS)
    # inject GITHUB_TOKEN into the placeholder secrets cell
    for cell in nb["cells"]:
        src = "".join(cell.get("source") or [])
        if "secrets are injected at build time" in src:
            cell["source"] = ["import os\n",
                              f'os.environ["GITHUB_TOKEN"] = {env.get("GITHUB_TOKEN", "")!r}\n',
                              "print('secrets set:', bool(os.environ['GITHUB_TOKEN']))\n"]
            break

    push_dir = NB_DIR / "push_gemma_agg"
    push_dir.mkdir(parents=True, exist_ok=True)
    (push_dir / "kaggle_gemma_agg.ipynb").write_text(json.dumps(nb, indent=1))
    (push_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": "softmaxsimp/gemma-variants-aggregator",
        "title": "Gemma variants -- aggregator (CPU)",
        "code_file": "kaggle_gemma_agg.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": [],
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }, indent=1))
    print(f"wrote {push_dir}/kernel-metadata.json (CPU, secrets injected)")
    print("push with: kaggle-softmaxsimp kernels push -p notebooks/push_gemma_agg")


if __name__ == "__main__":
    main()
