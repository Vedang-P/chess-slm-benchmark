"""Generate kaggle_rlvr_pretest.ipynb — P100 pre-flight for the RLVR run.

Verifies on GPU, in ~15 min, every path the full RLVR run touches:
1. P100 dep stack + trl 0.17 imports (never run together before)
2. stockfish installed + engine works
3. SFT adapter MERGE into the 4-bit base (dequant to fp16 — memory check)
4. 2 real GRPO steps with the stockfish oracle (outcome/process/style
   rewards on real positions)
5. HF checkpoint upload (save-steps 1)

    python3 notebooks/build_rlvr_kernel_notebook.py
    KAGGLE_API_TOKEN=<token> kaggle kernels push -p notebooks/push_rlvr_pretest
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from build_notebook import _code, _md, _notebook

NB_DIR = Path(__file__).resolve().parent
REPO_ID = "vedangfake/chess-slm-benchmark"

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
tok = find_token()
if tok:
    url = url.replace("https://", f"https://x-access-token:{tok}@")
res = subprocess.run(["git", "clone", "--quiet", "-b", "main", url, str(REPO)],
                     capture_output=True, text=True)
if res.returncode != 0:
    raise RuntimeError("clone failed: " + res.stderr[-300:])
os.chdir(REPO)
# guard against cloning mid-push: always run the latest main
subprocess.run(["git", "fetch", "--quiet", "origin", "main"], check=True)
subprocess.run(["git", "reset", "--hard", "--quiet", "origin/main"], check=True)
print("cwd:", Path.cwd())
'''.strip()

DEPS_CELL = r'''
import subprocess, sys
def pip(*pkgs, **kw):
    cmd = [sys.executable, "-m", "pip", "install", "-q", "--no-input"]
    cmd += pkgs
    subprocess.run(cmd, check=True, **kw)

# P100 stack (proven by the SFT runs) + trl 0.17 for GRPO.
pip("torch==2.4.1", "torchvision==0.19.1", "torchaudio==2.4.1",
    "--index-url", "https://download.pytorch.org/whl/cu118")
pip("transformers==5.13.1", "peft==0.14.0", "bitsandbytes==0.46.1",
    "accelerate", "datasets", "python-chess", "huggingface_hub",
    "trl==0.17.0")

# the RLVR oracle needs the stockfish BINARY (python-chess talks to it)
r = subprocess.run(["apt-get", "install", "-y", "-q", "stockfish"],
                   capture_output=True, text=True)
print("apt stockfish:", "ok" if r.returncode == 0 else r.stderr[-200:])
print("deps installed")
'''.strip()

GPU_CELL = r'''
import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("capability:", torch.cuda.get_device_capability(0))
    x = torch.randn(64, 64, device="cuda")
    print("cuda matmul ok:", float((x @ x).sum()))
else:
    raise SystemExit("CUDA NOT available")
# trl import on this stack (never verified together)
import trl
print("trl", trl.__version__)
# stockfish engine works?
import chess, chess.engine
eng = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")
info = eng.analyse(chess.Board(), chess.engine.Limit(depth=10))
print("stockfish analyse ok, best line:", [m.uci() for m in (info.get("pv") or [])][:2])
eng.quit()
'''.strip()

FETCH_CELL = r'''
import os, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

WORK = Path("/kaggle/working")
os.chdir(WORK / "chess-slm-benchmark")

ad = WORK / "caveman-sft-adapter"
ad.mkdir(parents=True, exist_ok=True)
for f in ("adapter_model.safetensors", "adapter_config.json"):
    p = hf_hub_download(repo_id="%REPO_ID%", filename=f"adapters/caveman-sft-final/{f}",
                        repo_type="dataset")
    shutil.copy(p, ad / f)
print("SFT adapter fetched:", sorted(p.name for p in ad.iterdir()))

pool = WORK / "pool.jsonl"
p = hf_hub_download(repo_id="%REPO_ID%", filename="rlvr-pool/train.jsonl",
                    repo_type="dataset")
shutil.copy(p, pool)
print("pool fetched:", len([l for l in pool.read_text().splitlines() if l.strip()]), "rows")
'''.strip()

PREP_CELL = r'''
import torch
from pathlib import Path
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

# One-time: merge the SFT adapter into the fp16 base and save it locally.
# The run cell then loads THIS dir through the 4-bit QLoRA path (~2.6GB)
# so the fp16 10.3GB base no longer blows the P100 at the first backward.
# KL reference stays correct: the merged base IS the policy with the RL
# adapter disabled (the SFT'd model), exactly the intended anchor.
OUT = Path("/kaggle/working/gemma-4-E2B-sft-merged")
if not OUT.exists():
    model = AutoModelForImageTextToText.from_pretrained(
        "google/gemma-4-E2B-it", device_map={"": 0}, dtype=torch.float16)
    model = PeftModel.from_pretrained(model, "/kaggle/working/caveman-sft-adapter")
    model = model.merge_and_unload()
    model.config.use_cache = False
    model.save_pretrained(OUT)
    AutoProcessor.from_pretrained("google/gemma-4-E2B-it").save_pretrained(OUT)
    print("merged base saved:", OUT)
    # CRITICAL: the fp16 model is 10.3GB on the GPU — free it before the
    # run cell loads the 4-bit version, or the load's allocator warmup
    # OOMs (measured 2026-08-20 pretest v4: 5.06GB free, warmup wants
    # 6.24GB). Notebook cells share one kernel; del + gc + empty_cache.
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print("prep model freed from GPU")
else:
    print("merged base already exists:", OUT)
'''.strip()

RUN_CELL = r'''
import os, signal, subprocess, sys

# Hard wall-clock cap: a hang or runaway must stop the run, not burn GPU.
# SIGINT first so the trainer saves + uploads its checkpoint (transformers
# handles KeyboardInterrupt gracefully), then SIGKILL if it ignores that.
RUN_TIMEOUT_MIN = 45   # demo; the full 200-step run uses 720

cmd = [sys.executable, "scripts/train_mate_grpo.py",
       "--base", "/kaggle/working/gemma-4-E2B-sft-merged",
       "--train", "/kaggle/working/pool.jsonl",
       "--out", "/kaggle/working/rlvr-pretest-adapter",
       "--oracle", "stockfish", "--stockfish", "/usr/games/stockfish",
       "--depth", "12",
       "--max-steps", "2", "--group", "4",
       "--optim", "adamw_bnb_8bit",
       "--max-train-rows", "64",
       "--save-steps", "1",
       "--hf-repo", "%REPO_ID%", "--hf-tag", "rlvr-pretest",
       "--hf-upload-every", "60",
       "--progress-every", "60",
       "--step-timeout-min", "45",
       "--wandb-project", "chess-slm-rlvr"]
print("running:", " ".join(cmd[:6]), "...")
proc = subprocess.Popen(cmd)
try:
    rc = proc.wait(timeout=RUN_TIMEOUT_MIN * 60)
except subprocess.TimeoutExpired:
    print(f"[run] wall-clock cap {RUN_TIMEOUT_MIN} min hit; "
          f"SIGINT for a graceful checkpoint", flush=True)
    proc.send_signal(signal.SIGINT)
    try:
        rc = proc.wait(timeout=90)
    except subprocess.TimeoutExpired:
        print("[run] no graceful exit; SIGKILL", flush=True)
        proc.kill()
        rc = proc.wait()
    raise SystemExit(f"rlvr timed out after {RUN_TIMEOUT_MIN} min "
                     f"(rc={rc}); checkpoint should be on HF")
if rc != 0:
    raise SystemExit(f"rlvr pretest failed: {rc}")
print("rlvr pretest done")
'''.strip()

UPLOAD_CELL = r'''
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_WRITE_TOKEN"])
api.upload_file(path_or_fileobj="/kaggle/working/rlvr-pretest-adapter/adapter_config.json",
                path_in_repo="rlvr-pretest/adapter_config.json",
                repo_id="%REPO_ID%", repo_type="dataset",
                commit_message="rlvr pretest adapter config")
print("pretest artifacts uploaded")
'''.strip()


def load_env() -> dict:
    env_path = NB_DIR.parent / ".env"
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
    slug = "rlvr-pretest"
    cells = [
        _md("# RLVR pretest (P100)\n\n"
            "Verify trl 0.17 + stockfish + SFT-adapter merge + 2 real "
            "GRPO steps before any full RLVR run."),
        _md("## 1. Secrets"),
        _code("print('secrets are injected at build time')"),
        _md("## 2. Get the repo"),
        _code(CLONE_CELL),
        _md("## 3. Dependencies (P100 stack + trl + stockfish)"),
        _code(DEPS_CELL),
        _md("## 3b. GPU + trl + stockfish sanity"),
        _code(GPU_CELL),
        _md("## 4. Fetch SFT adapter + pool"),
        _code(FETCH_CELL.replace("%REPO_ID%", REPO_ID)),
        _md("## 4b. Merge SFT into fp16 base (one-time, 4-bit base for the run)"),
        _code(PREP_CELL),
        _md("## 5. RLVR: 2 steps, stockfish oracle, 4-bit QLoRA on merged base"),
        _code(RUN_CELL.replace("%REPO_ID%", REPO_ID)),
        _md("## 6. Upload artifacts"),
        _code(UPLOAD_CELL.replace("%REPO_ID%", REPO_ID)),
    ]
    nb = _notebook(cells)
    env = load_env()
    inject_secrets(nb, env, ["HF_WRITE_TOKEN", "GITHUB_TOKEN", "WANDB_API_KEY"])

    push_dir = NB_DIR / f"push_{slug}"
    push_dir.mkdir(parents=True, exist_ok=True)
    code_file = f"kaggle_{slug}.ipynb"
    (push_dir / code_file).write_text(json.dumps(nb, indent=1))
    (push_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"softmaxsimp/{slug}",
        "title": "rlvr pretest",
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
    print(f"push with: KAGGLE_API_TOKEN=... "
          f"kaggle kernels push -p notebooks/{push_dir.name}")


if __name__ == "__main__":
    main()
