"""Generate kaggle_sft_eval.ipynb — eval the trained adapter on the 1k
noexplain test set (thinking ON, protocol parity with baselines).

Downloads the LATEST checkpoint from HF (noexplain-slice/checkpoint-*),
loads it on gemma-4-E2B, runs run_mate_eval on the 1000-position noexplain
test set, pushes results to wandb + HF.

    python notebooks/build_sft_eval_kernel_notebook.py
    kaggle kernels push -p notebooks/push_sft_eval
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from build_notebook import _code, _md, _notebook

NB_DIR = Path(__file__).resolve().parent
OWNER = os.environ.get("SFT_EVAL_OWNER", "vedanggggg")
SLUG = "sft-eval-noexplain"

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
# Same dependency matrix as the training kernel (P100/T4-safe):
# torch 2.5.1 cu121 --no-deps + nvidia stack (cudnn 9.1.1.17), transformers
# 5.13.1, peft 0.14, bnb 0.46.1, torchao removed.
CU121 = "https://download.pytorch.org/whl/cu121"
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1",
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
subprocess.run([sys.executable, "-m", "pip", "uninstall", "--quiet", "-y",
                "torchao"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "-r", "requirements.txt"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "peft==0.14.0"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "transformers==5.13.1"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-U", "wandb"], check=True)
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
'''.strip()

FETCH_ADAPTER_CELL = r'''
import os, json, shutil
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

api = HfApi(token=os.environ.get("HF_WRITE_TOKEN", ""))
files = api.list_repo_files("vedangfake/chess-slm-benchmark", repo_type="dataset")
prefix = "noexplain-slice/checkpoint-"
cps = sorted({f.split("/")[2] for f in files
              if f.startswith(prefix) and len(f.split("/")) > 2})
if not cps:
    raise RuntimeError("no checkpoints found in HF repo")
latest = cps[-1]
print("latest checkpoint:", latest, flush=True)
out = Path("results/noexplain-slice-adapter")
out.mkdir(parents=True, exist_ok=True)
n = 0
for f in files:
    if not f.startswith(f"{prefix}{latest}/"):
        continue
    # only adapter files (skip optimizer/scheduler/trainer_state)
    if f.endswith("adapter_model.safetensors") or f.endswith("adapter_config.json") \
       or f.endswith("README.md"):
        hf_hub_download(repo_id="vedangfake/chess-slm-benchmark", filename=f,
                        repo_type="dataset", local_dir="/kaggle/working/chess-slm-benchmark",
                        token=os.environ.get("HF_WRITE_TOKEN", ""))
        n += 1
print(f"downloaded {n} adapter files from {latest}", flush=True)
print("adapter dir:", out, list(out.iterdir()) if out.exists() else "MISSING", flush=True)
'''.strip()

EVAL_CELL = r'''
import os, subprocess, sys, time, json, glob
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

UPLOAD_CELL = r'''
import os, json, glob
from pathlib import Path
from huggingface_hub import HfApi
import wandb

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

# push final metrics into the wandb run
try:
    api_w = wandb.Api()
    runs = api_w.runs("vedanggg-mit-manipal/chess-slm-benchmark")
    target = None
    for r in runs:
        if r.name == "noexplain-slice" and r.state == "running":
            target = r
            break
    if target is None:
        for r in runs:
            if r.name == "noexplain-slice":
                target = r
    summary_paths = sorted(glob.glob("results/noexplain-slice-eval/*summary*.json"))
    if target is not None and summary_paths:
        m = json.loads(open(summary_paths[-1]).read())
        acc = m.get("accuracy", m)
        toks = m.get("token_usage", {})
        update = {
            "final/accuracy_strict": acc.get("accuracy_strict"),
            "final/accuracy_of_parsed": acc.get("accuracy_of_parsed"),
            "final/parse_rate": acc.get("parse_rate"),
            "final/correct": acc.get("correct"),
            "final/n": acc.get("n"),
            "final/mean_output_tokens": toks.get("output_tokens_mean"),
            "final/mean_reasoning_tokens": toks.get("reasoning_tokens_mean"),
        }
        tpc = (toks.get("output_tokens_total") / acc.get("correct")
               if acc.get("correct") and toks.get("output_tokens_total") else None)
        update["final/tokens_per_correct"] = tpc
        target.summary.update(update)
        target.update()
        print("wandb final metrics pushed:", update, flush=True)
except Exception as e:
    print("wandb push failed (non-fatal):", e, flush=True)
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
        _md("# SFT eval — noexplain 1000 (thinking ON)\n\n"
            "Loads the latest HF checkpoint, evals on the exact 1k noexplain "
            "test set with the protocol-parity config, pushes to wandb + HF."),
        _md("## 1. Secrets (injected at build time)"),
        _code("print('secrets are injected at build time')"),
        _md("## 2. Get the repo"),
        _code(CLONE_CELL),
        _md("## 3. Dependencies"),
        _code(DEPS_CELL),
        _md("## 4. Fetch latest adapter from HF"),
        _code(FETCH_ADAPTER_CELL),
        _md("## 5. Eval on 1k noexplain (thinking ON)"),
        _code(EVAL_CELL),
        _md("## 6. Upload results to HF + wandb"),
        _code(UPLOAD_CELL),
    ]
    nb = _notebook(cells)
    env = load_env()
    inject_secrets(nb, env, ["GITHUB_TOKEN", "HF_WRITE_TOKEN", "WANDB_API_KEY"])

    push_dir = NB_DIR / "push_sft_eval"
    push_dir.mkdir(parents=True, exist_ok=True)
    code_file = "kaggle_sft_eval.ipynb"
    (push_dir / code_file).write_text(json.dumps(nb, indent=1))
    (push_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{OWNER}/{SLUG}",
        "title": "SFT eval noexplain",
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
    print("push with: kaggle kernels push -p notebooks/push_sft_eval")


if __name__ == "__main__":
    main()
