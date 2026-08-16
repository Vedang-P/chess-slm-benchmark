"""Generate kaggle_gemma_legality.ipynb — CPU kernel that downloads the raw
gemma MATE eval samples from the HF archive, classifies every answer
(correct / wrong-illegal / wrong-legal-candidate / wrong-legal-other), and
uploads the report to HF.

    python notebooks/build_gemma_legality_kernel_notebook.py
    kaggle kernels push -p notebooks/push_gemma_legality
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from build_notebook import _code, _md, _notebook

NB_DIR = Path(__file__).resolve().parent
OWNER = os.environ.get("LEGALITY_OWNER", "vedanggggg")
SLUG = "gemma-legality-analysis"

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
                "-r", "requirements.txt"], check=True)
print("deps ok")
'''.strip()

ANALYZE_CELL = r'''
import os, subprocess, sys, time, json
from pathlib import Path

cmd = [sys.executable, "scripts/analyze_gemma_legality.py",
       "--workers", "w1,w2",
       "--out", "results/gemma-legality",
       "--upload"]
print("running:", " ".join(cmd))
t0 = time.time()
res = subprocess.run(cmd, stderr=subprocess.STDOUT)
print(f"analysis exited rc={res.returncode} after {(time.time()-t0)/60:.1f}min",
      flush=True)
if res.returncode != 0:
    raise RuntimeError("analysis failed -- see output above")
# print the report so it is visible in the kernel output
try:
    rep = json.loads(open("results/gemma-legality/legality_report.json").read())
    print("=== REPORT ===")
    print(json.dumps(rep, indent=1))
except Exception as e:
    print("report print failed:", e, flush=True)
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
    lines = ["import os\n"]
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
        _md("# Base gemma legality analysis\n\n"
            "Downloads the raw gemma1000 strategy eval samples from the HF "
            "archive and classifies every answer: correct / wrong-illegal "
            "(claimed move is ILLEGAL on the FEN) / wrong-legal-candidate / "
            "wrong-legal-other. Uploads the report to HF."),
        _md("## 1. Secrets (injected at build time)"),
        _code("print('secrets are injected at build time')"),
        _md("## 2. Get the repo"),
        _code(CLONE_CELL),
        _md("## 3. Dependencies"),
        _code(DEPS_CELL),
        _md("## 4. Analyze + upload"),
        _code(ANALYZE_CELL),
    ]
    nb = _notebook(cells)
    env = load_env()
    inject_secrets(nb, env, ["GITHUB_TOKEN", "HF_WRITE_TOKEN"])

    push_dir = NB_DIR / "push_gemma_legality"
    push_dir.mkdir(parents=True, exist_ok=True)
    code_file = "kaggle_gemma_legality.ipynb"
    (push_dir / code_file).write_text(json.dumps(nb, indent=1))
    (push_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{OWNER}/{SLUG}",
        "title": "Gemma legality analysis",
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
    print("push with: kaggle kernels push -p notebooks/push_gemma_legality")


if __name__ == "__main__":
    main()
