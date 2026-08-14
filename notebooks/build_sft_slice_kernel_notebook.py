"""Generate kaggle_sft_slice.ipynb — the noexplain slice SFT run on a Kaggle
T4 GPU kernel, with wandb logging + HF checkpoint safety net + resume.

    python notebooks/build_sft_slice_kernel_notebook.py
    kaggle kernels push -p notebooks/push_sft_slice

Run 1 (11h): full 600k SFT. If the session dies at 12h, tomorrow's kernel
starts with --resume-from-hf and picks up the HF checkpoints.

Secrets injected at build time: GITHUB_TOKEN, HF_WRITE_TOKEN, WANDB_API_KEY.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from build_notebook import _code, _md, _notebook

NB_DIR = Path(__file__).resolve().parent
OWNER = os.environ.get("SFT_SLICE_OWNER", "vedanggggg")
SLUG = "sft-slice-noexplain"
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
# GPU matrix: Kaggle free tier hands out P100 (sm_60) OR T4 (sm_75).
# torch 2.6+ dropped sm_60 -> P100 dies. torch 2.4.1/2.5.1 cu121 both pin
# nvidia-cudnn-cu12==9.1.0.70 which was REMOVED from PyPI, so the normal
# install fails. Fix: install torch with --no-deps, then pull the nvidia
# runtime stack from the PyTorch cu121 index, using cudnn 9.1.1.17
# (patch-compatible with the yanked 9.1.0.70).
# transformers is pinned to 5.13.1: 5.14+ needs torch.distributed._tensor,
# which does not exist in torch 2.4.1 (repo history notes exactly this).
CU121 = "https://download.pytorch.org/whl/cu121"
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "torch==2.4.1", "torchvision==0.19.1", "torchaudio==2.4.1",
                "--index-url", CU121, "--no-deps"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "nvidia-cudnn-cu12==9.1.1.17", "--index-url", CU121],
               check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "nvidia-cublas-cu12", "nvidia-cuda-cupti-cu12",
                "nvidia-cuda-nvrtc-cu12", "nvidia-cuda-runtime-cu12",
                "nvidia-cufft-cu12", "nvidia-curand-cu12",
                "nvidia-cusolver-cu12", "nvidia-cusparse-cu12",
                "nvidia-nccl-cu12", "nvidia-nvjitlink-cu12",
                "nvidia-nvtx-cu12", "triton",
                "--index-url", CU121], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "bitsandbytes==0.46.1"], check=True)
# Kaggle's base image ships torchao 0.10.0 which newer peft rejects
# ("only versions above 0.16.0 are supported") and we do not use torchao.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "--quiet", "-y",
                "torchao"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "-r", "requirements.txt"], check=True)
# peft 0.20+ calls nn.Module.set_submodule which torch 2.4.1 does not have
# (added in torch 2.5); pin peft to the last version before that refactor.
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "peft==0.14.0"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "transformers==5.13.1"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-U", "wandb"], check=True)
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
'''.strip()

FETCH_DATA_CELL = r'''
import os, subprocess, sys
from pathlib import Path

# the 600k noexplain slice labels + eval live in the HF dataset repo
# (too large for git). Pull them into the repo's expected paths.
from huggingface_hub import hf_hub_download
for name in ("train.jsonl", "eval.jsonl", "manifest.json"):
    hf_hub_download(
        repo_id="vedangfake/chess-slm-benchmark",
        filename=f"noexplain-slice/{name}",
        repo_type="dataset",
        local_dir="/kaggle/working/chess-slm-benchmark",
        token=os.environ.get("HF_WRITE_TOKEN", ""))
    print("fetched", name, flush=True)
'''.strip()

TRAIN_CELL = r'''
import os, subprocess, sys, time
from pathlib import Path

os.environ["BENCH_RUN_ID"] = "sft-slice-noexplain"
cmd = [sys.executable, "scripts/train_mate_lora.py",
       "--train", "data/positions/noexplain-slice/train.jsonl",
       "--eval", "data/positions/noexplain-slice/eval.jsonl",
       "--out", "results/noexplain-slice-adapter",
       "--wandb-project", "%(wandb_project)s",
       "--train-tag", "noexplain-slice",
       "--hf-repo", "vedangfake/chess-slm-benchmark",
       "--hf-upload-every", "1800",
       "--epochs", "%(epochs)s",
       "--rank", "%(rank)s",
       "--batch", "%(batch)s",
       "--grad-accum", "%(grad_accum)s",
       "--eval-steps", "3000",
       "--save-steps", "2000",
       "%(resume)s",
       "%(smoke)s"]
cmd = [c for c in cmd if c]
print("running:", " ".join(cmd))
t0 = time.time()
res = subprocess.run(cmd, stderr=subprocess.STDOUT)
print(f"train exited rc={res.returncode} after {(time.time()-t0)/3600:.2f}h", flush=True)
if res.returncode != 0:
    raise RuntimeError("training failed -- see output above")
'''.strip()

EVAL_CELL = r'''
import os, subprocess, sys, time
from pathlib import Path

# score the adapter on the exact 1k-position noexplain test set,
# thinking ON (protocol parity with the 92.2% deepseek baseline)
cmd = [sys.executable, "scripts/run_mate_eval.py",
       "--model", "gemma4-e2b",
       "--adapter", "results/noexplain-slice-adapter",
       "--task-file", "mate-selection-test-noexplain.json",
       "--n", "1000",
       "--local-thinking",
       "--max_new_tokens", "2048",
       "--output_dir", "results/noexplain-slice-eval",
       "--verbose"]
print("running:", " ".join(cmd))
t0 = time.time()
res = subprocess.run(cmd, stderr=subprocess.STDOUT)
print(f"eval exited rc={res.returncode} after {(time.time()-t0)/60:.1f}min", flush=True)
if res.returncode != 0:
    raise RuntimeError("eval failed -- see output above")
'''.strip()

UPLOAD_EVAL_CELL = r'''
import os, json
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi(token=os.environ.get("HF_WRITE_TOKEN", ""))
eval_dir = Path("results/noexplain-slice-eval")
if eval_dir.exists():
    for f in eval_dir.iterdir():
        if f.is_file():
            api.upload_file(path_or_fileobj=f.read_bytes(),
                            path_in_repo=f"noexplain-slice-eval/{f.name}",
                            repo_id="vedangfake/chess-slm-benchmark",
                            repo_type="dataset",
                            commit_message=f"noexplain slice eval {f.name}")
    print("eval uploaded to HF", flush=True)
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
    import sys as _sys
    smoke = "--smoke" in _sys.argv
    resume = "--resume" in _sys.argv
    epochs = "0.15" if smoke else "1.4"
    rank = "16" if smoke else "32"
    batch = "2"
    grad_accum = "4" if smoke else "8"

    cells = [
        _md("# Noexplain slice SFT — gemma-4-E2B QLoRA\n\n"
            "600k phase-natural noexplain labels + (later) verified lucid "
            "traces. Wandb-logged, HF checkpoints every 30 min, resume-safe. "
            "Targets: beat MATE 8B (63.5%), stretch deepseek (92.2%)."),
        _md("## 1. Secrets (injected at build time)"),
        _code("print('secrets are injected at build time')"),
        _md("## 2. Get the repo"),
        _code(CLONE_CELL),
        _md("## 3. Dependencies"),
        _code(DEPS_CELL),
        _md("## 4. Fetch the noexplain slice data (HF)"),
        _code(FETCH_DATA_CELL),
        _md("## 5. Train (wandb-logged, HF checkpointed)"),
        _code(TRAIN_CELL % {"wandb_project": WANDB_PROJECT,
                            "epochs": epochs, "rank": rank, "batch": batch,
                            "grad_accum": grad_accum,
                            "resume": "--resume-from-hf latest" if resume else "",
                            "smoke": "--smoke" if smoke else ""}),
        _md("## 6. Eval on the 1k noexplain test set (thinking ON)"),
        _code(EVAL_CELL),
        _md("## 7. Upload eval to HF"),
        _code(UPLOAD_EVAL_CELL),
    ]
    nb = _notebook(cells)
    env = load_env()
    inject_secrets(nb, env, ["GITHUB_TOKEN", "HF_WRITE_TOKEN", "WANDB_API_KEY"])

    push_dir = NB_DIR / ("push_sft_slice_smoke" if smoke else "push_sft_slice")
    push_dir.mkdir(parents=True, exist_ok=True)
    code_file = "kaggle_sft_slice.ipynb"
    (push_dir / code_file).write_text(json.dumps(nb, indent=1))
    (push_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{OWNER}/{'sft-slice-noexplain-smoke' if smoke else SLUG}",
        "title": "SFT slice noexplain smoke" if smoke else "SFT slice noexplain",
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
    flags = []
    if smoke:
        flags.append("SMOKE")
    if resume:
        flags.append("RESUME")
    print(f"wrote {push_dir}/{code_file} ({' + '.join(flags) or 'FULL'})")
    print("push with: kaggle kernels push -p notebooks/push_sft_slice")


if __name__ == "__main__":
    main()
