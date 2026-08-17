"""Generate kaggle_gate0_probe.ipynb — Gate 0: eval checkpoint-4000 (the
0.107-epoch labels model) on 200 noexplain positions, thinking ON.

This single number decides the training mix (labels carry competence vs not).
~1-2h GPU. Uploads report to HF + wandb.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from build_notebook import _code, _md, _notebook

NB_DIR = Path(__file__).resolve().parent
OWNER = os.environ.get("GATE0_OWNER", "vedanggggg")
SLUG = "gate0-ckpt4000-probe"
CHECKPOINT = "checkpoint-4000"

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
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
'''.strip()

FETCH_ADAPTER_CELL = r'''
import os, shutil
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

api = HfApi(token=os.environ.get("HF_WRITE_TOKEN", ""))
cp = "%(checkpoint)s"
print("fetching adapter from", cp, flush=True)
# run_mate_eval expects the adapter at results/noexplain-slice-adapter/
out = Path("results/noexplain-slice-adapter")
out.mkdir(parents=True, exist_ok=True)
tmp = Path("/tmp/adapter-fetch")
tmp.mkdir(parents=True, exist_ok=True)
n = 0
for f in api.list_repo_files("vedangfake/chess-slm-benchmark", repo_type="dataset"):
    if not f.startswith(f"noexplain-slice/{cp}/"):
        continue
    if f.endswith(("adapter_model.safetensors", "adapter_config.json", "README.md")):
        dl = hf_hub_download(repo_id="vedangfake/chess-slm-benchmark", filename=f,
                             repo_type="dataset", local_dir=str(tmp),
                             token=os.environ.get("HF_WRITE_TOKEN", ""))
        dest = out / Path(dl).name
        shutil.copyfile(dl, dest)
        n += 1
print(f"downloaded {n} adapter files -> {out}", flush=True)
print("adapter dir:", sorted(p.name for p in out.iterdir()), flush=True)
'''.strip()

PROBE_CELL = r'''
import os, subprocess, sys, time, json, random, traceback
from pathlib import Path

# eval 200 positions (fixed seed slice of the 1000) thinking ON.
# Protocol MUST match the 58.1% baseline exactly (HF archive meta):
# thinking ON, unbounded budget, force_answer_prompt=true, max 32768.
cmd = [sys.executable, "scripts/run_mate_eval.py",
       "--model", "gemma4-e2b",
       "--adapter", "results/noexplain-slice-adapter",
       "--task-file", "mate-selection-test-noexplain.json",
       "--n", "200",
       "--offset", "0",
       "--local-thinking",
       "--force-answer-prompt",
       "--max_new_tokens", "32768",
       "--output_dir", "results/gate0-probe",
       "--verbose"]
print("running:", " ".join(cmd))
t0 = time.time()
try:
    res = subprocess.run(cmd, stderr=subprocess.STDOUT, timeout=8*3600)
    print(f"probe exited rc={res.returncode} after {(time.time()-t0)/60:.1f}min",
          flush=True)
    if res.returncode != 0:
        raise RuntimeError("probe failed -- see output above")
except Exception as e:
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_WRITE_TOKEN", ""))
        body = f"GATE0 FAILED: {type(e).__name__}: {e}\n{traceback.format_exc()[-2500:]}"
        api.upload_file(path_or_fileobj=body.encode(),
                        path_in_repo="gate0-probe/error.txt",
                        repo_id="vedangfake/chess-slm-benchmark",
                        repo_type="dataset",
                        commit_message="gate0 failure")
        print("error written to HF gate0-probe/error.txt", flush=True)
    except Exception as e2:
        print("error upload failed:", e2, flush=True)
    raise
'''.strip()

UPLOAD_CELL = r'''
import os, json, glob
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi(token=os.environ.get("HF_WRITE_TOKEN", ""))
eval_dir = Path("results/gate0-probe")
if eval_dir.exists():
    for f in eval_dir.iterdir():
        if f.is_file():
            api.upload_file(path_or_fileobj=f.read_bytes(),
                            path_in_repo=f"gate0-probe/{f.name}",
                            repo_id="vedangfake/chess-slm-benchmark",
                            repo_type="dataset",
                            commit_message=f"gate0 probe {f.name}")
    print("gate0 probe uploaded to HF", flush=True)

# print the summary for the log
for sp in sorted(glob.glob("results/gate0-probe/*summary*.json")):
    m = json.loads(open(sp).read())
    acc = m.get("accuracy", m)
    print("=== GATE0 SUMMARY ===")
    print(json.dumps({k: acc.get(k) for k in ("n","correct","wrong","parse_rate","accuracy_strict","accuracy_of_parsed")}, indent=1))
    break
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
        _md("# Gate 0 — ckpt-4000 thinking-ON probe\n\n"
            "Eval the 0.107-epoch labels model on 200 noexplain positions, "
            "thinking ON. This number decides the training mix."),
        _md("## 1. Secrets"),
        _code("print('secrets are injected at build time')"),
        _md("## 2. Get the repo"),
        _code(CLONE_CELL),
        _md("## 3. Dependencies"),
        _code(DEPS_CELL),
        _md("## 4. Fetch checkpoint-4000 adapter"),
        _code(FETCH_ADAPTER_CELL % {"checkpoint": CHECKPOINT}),
        _md("## 5. Probe eval (200 pos, thinking ON)"),
        _code(PROBE_CELL),
        _md("## 6. Upload + print summary"),
        _code(UPLOAD_CELL),
    ]
    nb = _notebook(cells)
    env = load_env()
    inject_secrets(nb, env, ["GITHUB_TOKEN", "HF_WRITE_TOKEN"])

    push_dir = NB_DIR / "push_gate0_probe"
    push_dir.mkdir(parents=True, exist_ok=True)
    code_file = "kaggle_gate0_probe.ipynb"
    (push_dir / code_file).write_text(json.dumps(nb, indent=1))
    (push_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{OWNER}/{SLUG}",
        "title": "Gate0 ckpt4000 probe",
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
    print("push with: kaggle kernels push -p notebooks/push_gate0_probe")


if __name__ == "__main__":
    main()
