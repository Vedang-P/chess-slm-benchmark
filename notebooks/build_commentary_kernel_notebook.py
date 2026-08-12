"""Generate kaggle_commentary_prehoch.ipynb — full-game commentary
distillation data generation on a Kaggle kernel (persists across
sessions; the ~40h pre-hoc API loop runs here).

    python notebooks/build_commentary_kernel_notebook.py
    kaggle kernels push -p notebooks/push_commentary

The kernel clones main (which carries scripts/build_commentary_data.py +
data/positions/fullgame/games.jsonl), installs python-chess, then runs:

    prehoch -> reconcile -> emit

and uploads train.jsonl + eval.jsonl + the raw prehoch/reconciled rows
to the HF dataset repo so the local trainer can pull them.

Secrets injected at build time (never committed): OPENCODE_API_KEY,
HF_WRITE_TOKEN, GITHUB_TOKEN.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from build_notebook import _code, _md, _notebook

NB_DIR = Path(__file__).resolve().parent
OWNER = os.environ.get("COMMENTARY_OWNER", "softmaxsimp")
SLUG = "commentary-prehoch"
HF_DATASET = "vedangfake/chess-bench-results"

CLONE_CELL = r'''
import os, shutil, subprocess, sys
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "chess-slm-benchmark"
if REPO.exists():
    shutil.rmtree(REPO)

def find_token():
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):
        v = os.environ.get(name)
        if v:
            return v
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("GITHUB_TOKEN")
    except Exception:
        return None

url = "https://github.com/Vedang-P/chess-slm-benchmark.git"
url = url.replace("https://", f"https://x-access-token:{find_token()}@")
res = subprocess.run(["git", "clone", "--quiet", "-b", "main", url, str(REPO)],
                     capture_output=True, text=True)
if res.returncode != 0:
    raise RuntimeError("clone failed: " + res.stderr[-300:])
os.chdir(REPO)
print("cwd:", Path.cwd())
'''.strip()

DEPS_CELL = r'''
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "python-chess"], check=True)
import chess
print("python-chess", chess.__version__)
'''.strip()

PREHOCH_CELL = r'''
import os, subprocess, sys, time
from pathlib import Path

# resumable: skip if a previous wave already wrote rows
out = Path("data/raw/commentary/prehoch.jsonl")
if out.exists():
    n = sum(1 for _ in out.open())
    print(f"resuming prehoch from {n} existing rows", flush=True)
cmd = [sys.executable, "-u", "scripts/build_commentary_data.py",
       "prehoch",
       "--model", "deepseek-v4-flash",
       "--temperature", "0.3",
       "--max-tokens", "2000",
       "--max-attempts", "3"]
print("running:", " ".join(cmd))
t0 = time.time()
res = subprocess.run(cmd, stderr=subprocess.STDOUT)
print(f"prehoch exited rc={res.returncode} after {(time.time()-t0)/3600:.2f}h",
      flush=True)
if res.returncode != 0:
    raise RuntimeError("prehoch failed -- see output above")
'''.strip()

RECONCILE_CELL = r'''
import os, subprocess, sys, time
from pathlib import Path

cmd = [sys.executable, "-u", "scripts/build_commentary_data.py",
       "reconcile",
       "--model", "deepseek-v4-flash",
       "--temperature", "0.3",
       "--max-tokens", "2000",
       "--max-attempts", "3"]
print("running:", " ".join(cmd))
t0 = time.time()
res = subprocess.run(cmd, stderr=subprocess.STDOUT)
print(f"reconcile exited rc={res.returncode} after {(time.time()-t0)/3600:.2f}h",
      flush=True)
if res.returncode != 0:
    raise RuntimeError("reconcile failed -- see output above")
'''.strip()

EMIT_CELL = r'''
import os, subprocess, sys
from pathlib import Path

cmd = [sys.executable, "scripts/build_commentary_data.py", "emit",
       "--agree-ratio", "2"]
print("running:", " ".join(cmd))
res = subprocess.run(cmd, stderr=subprocess.STDOUT)
if res.returncode != 0:
    raise RuntimeError("emit failed -- see output above")
for f in ("train.jsonl", "eval.jsonl"):
    p = Path("data/raw/commentary") / f
    if not p.exists():
        raise RuntimeError(f"emit did not produce {f}")
    print(f"{f}: {sum(1 for _ in p.open())} rows", flush=True)
'''.strip()

UPLOAD_CELL = r'''
import os, sys
from pathlib import Path

from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_WRITE_TOKEN", ""))
src = Path("data/raw/commentary")
for f in ("prehoch.jsonl", "reconciled.jsonl", "train.jsonl", "eval.jsonl"):
    p = src / f
    if not p.exists():
        print(f"skip {f} (missing)", flush=True)
        continue
    api.upload_file(path_or_fileobj=p.read_bytes(),
                    path_in_repo=f"fullgame-commentary/{f}",
                    repo_id="%(hf_dataset)s",
                    repo_type="dataset",
                    commit_message=f"fullgame-commentary {f}")
    print(f"uploaded {f} ({p.stat().st_size/1e6:.1f}MB)", flush=True)
'''.strip()


def load_env() -> dict:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    env = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def inject_secrets(nb: dict, env: dict, names: list[str]) -> None:
    lines = []
    for name in names:
        if name not in env or not env[name]:
            raise RuntimeError(f"missing secret {name} in .env")
        lines.append(f'os.environ[{name!r}] = {env[name]!r}\n')
    lines.append("print('secrets set:', "
                 + ", ".join(f"bool(os.environ.get({n!r}))" for n in names)
                 + ")\n")
    cell = {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": lines,
    }
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source") or [])
        if "secrets are injected at build time" in src:
            nb["cells"][i] = cell
            return
    raise RuntimeError("placeholder secrets cell not found")


def main() -> None:
    cells = [
        _md("# Full-game commentary distillation — data generation\n\n"
            "deepseek-v4-flash comments on Lichess/TWIC master games "
            "move-by-move (pre-hoc + reconcile), emitting the Track-2 "
            "game-format training rows. Runs on a Kaggle CPU kernel so "
            "the long API loop persists across sessions."),
        _md("## 1. Secrets (injected at build time)"),
        _code("print('secrets are injected at build time')"),
        _md("## 2. Get the repo"),
        _code(CLONE_CELL),
        _md("## 3. Dependencies"),
        _code(DEPS_CELL),
        _md("## 4. Pre-hoc commentary (per-move, long)"),
        _code(PREHOCH_CELL),
        _md("## 5. Reconcile disagreements"),
        _code(RECONCILE_CELL),
        _md("## 6. Emit train/eval rows"),
        _code(EMIT_CELL),
        _md("## 7. Upload to HF"),
        _code(UPLOAD_CELL % {"hf_dataset": HF_DATASET}),
    ]
    nb = _notebook(cells)
    env = load_env()
    inject_secrets(nb, env, ["OPENCODE_API_KEY", "HF_WRITE_TOKEN",
                             "GITHUB_TOKEN"])

    push_dir = NB_DIR / "push_commentary"
    push_dir.mkdir(parents=True, exist_ok=True)
    code_file = "kaggle_commentary_prehoch.ipynb"
    (push_dir / code_file).write_text(json.dumps(nb, indent=1))
    (push_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{OWNER}/{SLUG}",
        "title": "Commentary prehoch",
        "code_file": code_file,
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
    print(f"wrote {push_dir}/{code_file}")
    print("push with: kaggle kernels push -p notebooks/push_commentary")


if __name__ == "__main__":
    main()
