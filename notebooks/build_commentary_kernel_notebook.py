"""Generate kaggle_commentary_train.ipynb — QLoRA SFT of gemma-4-E2B on
the fullgame lucid-commentary corpus, run on a Kaggle T4 GPU kernel.

    python notebooks/build_commentary_kernel_notebook.py
    kaggle kernels push -p notebooks/push_commentary

The kernel clones main (which carries scripts/ + data/positions/fullgame
games; the emitted train/eval rows are pulled from the HF dataset repo),
installs the campaign deps, trains with the SAME model and load path as
every eval baseline (full multimodal google/gemma-4-E2B-it, 4-bit,
device_map {"":0}), and uploads the adapter + eval results to HF.

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
SLUG = "commentary-train"
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
                "torch==2.5.1", "--index-url",
                "https://download.pytorch.org/whl/cu121"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "bitsandbytes==0.44.1"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "-r", "requirements.txt"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "pillow", "torchvision"], check=True)
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
'''.strip()

FETCH_DATA_CELL = r'''
import os, shutil, sys
from pathlib import Path
from huggingface_hub import hf_hub_download

os.makedirs("data/raw/commentary", exist_ok=True)
for f in ("train.jsonl", "eval.jsonl"):
    p = hf_hub_download(repo_id="%(hf_dataset)s",
                        filename=f"fullgame-commentary/{f}",
                        repo_type="dataset",
                        token=os.environ.get("HF_WRITE_TOKEN"))
    shutil.copy(p, f"data/raw/commentary/{f}")
    print(f"{f}: {sum(1 for _ in open(f'data/raw/commentary/{f}'))} rows")
'''.strip()

TRAIN_CELL = r'''
import os, subprocess, sys, time
from pathlib import Path

cmd = [sys.executable, "scripts/train_mate_lora.py",
       "--train", "data/raw/commentary/train.jsonl",
       "--eval", "data/raw/commentary/eval.jsonl",
       "--out", "results/fullgame-lora-adapter",
       "--game-mode",
       "--max-seq-len", "3072",
       "--batch", "2",
       "--grad-accum", "8",
       "--epochs", "1",
       "%(smoke)s"]
cmd = [c for c in cmd if c]
print("running:", " ".join(cmd))
t0 = time.time()
res = subprocess.run(cmd, stderr=subprocess.STDOUT)
print(f"train exited rc={res.returncode} after {(time.time()-t0)/3600:.2f}h",
      flush=True)
if res.returncode != 0:
    raise RuntimeError("training failed -- see output above")
'''.strip()

UPLOAD_CELL = r'''
import os, sys
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi(token=os.environ.get("HF_WRITE_TOKEN", ""))
adapter = Path("results/fullgame-lora-adapter")
for f in adapter.iterdir():
    if f.is_file():
        api.upload_file(path_or_fileobj=f.read_bytes(),
                        path_in_repo=f"fullgame-commentary-adapter/{f.name}",
                        repo_id="%(hf_dataset)s",
                        repo_type="dataset",
                        commit_message=f"fullgame adapter {f.name}")
print("adapter uploaded to HF", flush=True)
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
    smoke = "--smoke" in os.sys.argv
    cells = [
        _md("# Full-game lucid-commentary LoRA — gemma-4-E2B\n\n"
            "QLoRA SFT on deepseek's lucid commentary over 100 master "
            "games. Same model + load path as every eval baseline."),
        _md("## 1. Secrets (injected at build time)"),
        _code("print('secrets are injected at build time')"),
        _md("## 2. Get the repo"),
        _code(CLONE_CELL),
        _md("## 3. Dependencies (campaign DEPS_CELL)"),
        _code(DEPS_CELL),
        _md("## 4. Pull emitted train/eval rows from HF"),
        _code(FETCH_DATA_CELL % {"hf_dataset": HF_DATASET}),
        _md("## 5. Train (campaign model, 4-bit QLoRA)"),
        _code(TRAIN_CELL % {"smoke": "--smoke" if smoke else ""}),
        _md("## 6. Upload adapter to HF"),
        _code(UPLOAD_CELL % {"hf_dataset": HF_DATASET}),
    ]
    nb = _notebook(cells)
    env = load_env()
    inject_secrets(nb, env, ["OPENCODE_API_KEY", "HF_WRITE_TOKEN",
                             "GITHUB_TOKEN"])

    push_dir = NB_DIR / "push_commentary"
    push_dir.mkdir(parents=True, exist_ok=True)
    code_file = "kaggle_commentary_train.ipynb"
    (push_dir / code_file).write_text(json.dumps(nb, indent=1))
    (push_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{OWNER}/{SLUG}",
        "title": "Commentary train",
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
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
