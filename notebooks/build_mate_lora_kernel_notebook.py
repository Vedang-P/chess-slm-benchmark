"""Generate kaggle_mate_lora.ipynb — QLoRA SFT of gemma-4-E2B on MATE
noexplain, run on a Kaggle T4 GPU kernel.

    python notebooks/build_mate_lora_kernel_notebook.py
    kaggle kernels push -p notebooks/push_mate_lora

The kernel clones main, installs deps, builds the LoRA data (already
committed under data/positions/mate-lora/), trains with wandb logging,
uploads the adapter to HF, and (optionally) runs the 1k-position
noexplain eval with the adapter.

Secrets injected at build time (never committed): GITHUB_TOKEN,
HF_WRITE_TOKEN, WANDB_API_KEY.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from build_notebook import _code, _md, _notebook

NB_DIR = Path(__file__).resolve().parent
OWNER = os.environ.get("MATE_LORA_OWNER", "softmaxsimp")
SLUG = "mate-lora-noexplain"
WANDB_PROJECT = "chess-slm-benchmark"

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
import torch
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "torch==2.5.1", "--index-url",
                "https://download.pytorch.org/whl/cu121"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "torchvision==0.20.1", "torchaudio==2.5.1", "--index-url",
                "https://download.pytorch.org/whl/cu121"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "bitsandbytes==0.44.1"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "-r", "requirements.txt"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-U", "wandb"], check=True)
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
'''.strip()

TRAIN_CELL = r'''
import os, subprocess, sys, time
from pathlib import Path

os.environ["BENCH_RUN_ID"] = "mate-lora-noexplain"
cmd = [sys.executable, "scripts/train_mate_lora.py",
       "--train", "data/positions/mate-lora/train.jsonl",
       "--eval", "data/positions/mate-lora/eval.jsonl",
       "--out", "results/mate-lora-adapter",
       "--wandb-project", "%(wandb_project)s",
       "%(smoke)s"]
cmd = [c for c in cmd if c]
print("running:", " ".join(cmd))
t0 = time.time()
res = subprocess.run(cmd, stderr=subprocess.STDOUT)
print(f"train exited rc={res.returncode} after {(time.time()-t0)/3600:.2f}h", flush=True)
if res.returncode != 0:
    raise RuntimeError("training failed -- see output above")
'''.strip()

UPLOAD_CELL = r'''
import os, sys
from pathlib import Path

# push adapter + processor to HF so any kernel can eval it
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_WRITE_TOKEN", ""))
adapter = Path("results/mate-lora-adapter")
for f in adapter.iterdir():
    if f.is_file():
        api.upload_file(path_or_fileobj=f.read_bytes(),
                        path_in_repo=f"mate-lora-noexplain/{f.name}",
                        repo_id="vedangfake/chess-bench-results",
                        repo_type="dataset",
                        commit_message=f"mate-lora {f.name}")
print("adapter uploaded to HF", flush=True)
'''.strip()

EVAL_CELL = r'''
import os, subprocess, sys, time
from pathlib import Path

# score the adapter on the exact 1k-position noexplain test set
cmd = [sys.executable, "scripts/run_mate_eval.py",
       "--model", "gemma4-e2b",
       "--adapter", "results/mate-lora-adapter",
       "--task-file", "mate-selection-test-noexplain.json",
       "--n", "1000",
       "--max_new_tokens", "128",
       "--output_dir", "results/mate-lora-eval",
       "--verbose"]
print("running:", " ".join(cmd))
t0 = time.time()
res = subprocess.run(cmd, stderr=subprocess.STDOUT)
print(f"eval exited rc={res.returncode} after {(time.time()-t0)/60:.1f}min", flush=True)
if res.returncode != 0:
    raise RuntimeError("eval failed -- see output above")
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
        _md("# MATE noexplain LoRA — gemma-4-E2B\n\n"
            "QLoRA SFT on 200k noexplain selection pairs; eval on the "
            "1k-position noexplain test set. Target: >90% (deepseek 92.2%)."),
        _md("## 1. Secrets (injected at build time)"),
        _code("print('secrets are injected at build time')"),
        _md("## 2. Get the repo"),
        _code(CLONE_CELL),
        _md("## 3. Dependencies"),
        _code(DEPS_CELL),
        _md("## 4. Train (wandb-logged)"),
        _code(TRAIN_CELL % {"wandb_project": WANDB_PROJECT,
                            "smoke": "--smoke" if smoke else ""}),
        _md("## 5. Upload adapter to HF"),
        _code(UPLOAD_CELL),
        _md("## 6. Eval on the 1k noexplain test set"),
        _code(EVAL_CELL),
    ]
    nb = _notebook(cells)
    env = load_env()
    inject_secrets(nb, env, ["GITHUB_TOKEN", "HF_WRITE_TOKEN", "WANDB_API_KEY"])

    push_dir = NB_DIR / "push_mate_lora"
    push_dir.mkdir(parents=True, exist_ok=True)
    code_file = "kaggle_mate_lora.ipynb"
    (push_dir / code_file).write_text(json.dumps(nb, indent=1))
    (push_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{OWNER}/{SLUG}",
        "title": "MATE LoRA noexplain",
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
    print(f"wrote {push_dir}/{code_file} ({'SMOKE' if smoke else 'FULL'})")
    print("push with: kaggle kernels push -p notebooks/push_mate_lora")


if __name__ == "__main__":
    main()
