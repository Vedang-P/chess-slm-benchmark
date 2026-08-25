"""Generate kaggle_rlvr_base_5k.ipynb — GRPO on base gemma-4-E2B (no SFT) with 5k stratified pool.
2-step gate + 200-step probes A1/B, real gemma, CLI push, HF checkpoints.
"""
from pathlib import Path
import json, os
from build_notebook import _code, _md, _notebook

NB_DIR = Path(__file__).resolve().parent
REPO_ID = "vedangfake/chess-slm-benchmark"

CLONE_CELL = r'''
import os, subprocess, pathlib
repo = "https://github.com/Vedang-P/chess-slm-benchmark.git"
if not pathlib.Path("chess-slm-benchmark").exists():
    subprocess.run(["git", "clone", repo], check=True)
%cd chess-slm-benchmark
subprocess.run(["git", "pull"], check=True)
print("repo ready:", pathlib.Path.cwd())
'''.strip()

DEPS_CELL = r'''
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.49.*", "trl==0.14.*", "peft==0.15.*", "bitsandbytes==0.46.*", "huggingface_hub", "python-chess", "datasets"], check=True)
subprocess.run(["apt-get", "update", "-qq"], check=True)
subprocess.run(["apt-get", "install", "-y", "stockfish"], check=True)
import shutil
print("stockfish:", shutil.which("stockfish") or "/usr/games/stockfish")
'''.strip()

FETCH_CELL = r'''
from pathlib import Path
from huggingface_hub import HfApi
import os
# The 5k pool is built locally and pushed to HF dataset under rlvr-pool/train-5k.jsonl
# For now, if not on HF, expect it at data/positions or results/rlvr-pool
pool_src = Path("results/rlvr-pool/train-5k.jsonl")
if not pool_src.exists():
    # fallback: try HF dataset
    from huggingface_hub import hf_hub_download
    try:
        pool_src = Path(hf_hub_download(repo_id="vedangfake/chess-slm-benchmark", repo_type="dataset", filename="rlvr-pool/train-5k.jsonl"))
        print("pool from HF:", pool_src)
    except Exception as e:
        print("pool not found:", e)
        raise
pool_dst = Path("/kaggle/working/pool.jsonl")
pool_dst.write_bytes(pool_src.read_bytes())
print("pool ready:", pool_dst, len(pool_dst.read_text().splitlines()), "rows")
# preview
import json
print(json.loads(pool_dst.read_text().splitlines()[0]))
'''.strip()

GATE_CELL = r'''
import os, signal, subprocess, sys
# 2-step gate on base gemma-4-E2B (no SFT merge) — must show MoveA: and outcome>0
cmd = [sys.executable, "scripts/train_mate_grpo.py",
       "--base", "google/gemma-4-E2B-it",
       "--train", "/kaggle/working/pool.jsonl",
       "--out", "/kaggle/working/rlvr-gate-adapter",
       "--oracle", "stockfish", "--stockfish", "/usr/games/stockfish",
       "--depth", "12",
       "--max-steps", "2", "--group", "8",
       "--max-completion-length", "256",
       "--temperature", "0.7", "--top-p", "0.9",
       "--save-steps", "1",
       "--hf-repo", "vedangfake/chess-slm-benchmark", "--hf-tag", "rlvr-gate-base5k",
       "--hf-upload-every", "60",
       "--progress-every", "60",
       "--step-timeout-min", "45"]
print("gate running:", " ".join(cmd))
proc = subprocess.Popen(cmd)
try:
    rc = proc.wait(timeout=600)
except subprocess.TimeoutExpired:
    proc.send_signal(signal.SIGINT)
    rc = proc.wait(timeout=90)
if rc != 0:
    raise SystemExit(f"gate failed: {rc}")
print("gate done — check logs for MoveA: and outcome_reward mean>0")
'''.strip()

TRAIN_A1_CELL = r'''
import subprocess, sys, signal
# Arm A1: 1.0 outcome +0.3 process +0.0 style, P1, Dr.GRPO, G8 256
cmd = [sys.executable, "scripts/train_mate_grpo.py",
       "--base", "google/gemma-4-E2B-it",
       "--train", "/kaggle/working/pool.jsonl",
       "--out", "/kaggle/working/rlvr-a1-adapter",
       "--oracle", "stockfish", "--stockfish", "/usr/games/stockfish",
       "--depth", "12",
       "--max-steps", "200", "--group", "8",
       "--max-completion-length", "256",
       "--temperature", "0.7", "--top-p", "0.9",
       "--save-steps", "25",
       "--hf-repo", "vedangfake/chess-slm-benchmark", "--hf-tag", "rlvr-a1-base5k",
       "--hf-upload-every", "300",
       "--progress-every", "60",
       "--step-timeout-min", "45",
       "--wandb-project", "chess-slm-rlvr"]
print("A1 running:", " ".join(cmd))
proc = subprocess.Popen(cmd)
try:
    rc = proc.wait(timeout=720*60)
except subprocess.TimeoutExpired:
    proc.send_signal(signal.SIGINT)
    rc = proc.wait(timeout=90)
if rc != 0:
    raise SystemExit(f"A1 failed: {rc}")
print("A1 done")
'''.strip()

TRAIN_B_CELL = r'''
import subprocess, sys, signal
# Arm B: 0.1 outcome +0.9 process +0.0 style
cmd = [sys.executable, "scripts/train_mate_grpo.py",
       "--base", "google/gemma-4-E2B-it",
       "--train", "/kaggle/working/pool.jsonl",
       "--out", "/kaggle/working/rlvr-b-adapter",
       "--oracle", "stockfish", "--stockfish", "/usr/games/stockfish",
       "--depth", "12",
       "--reward-weights", "0.1,0.9,0.0",
       "--max-steps", "200", "--group", "8",
       "--max-completion-length", "256",
       "--temperature", "0.7", "--top-p", "0.9",
       "--save-steps", "25",
       "--hf-repo", "vedangfake/chess-slm-benchmark", "--hf-tag", "rlvr-b-base5k",
       "--hf-upload-every", "300",
       "--progress-every", "60",
       "--step-timeout-min", "45",
       "--wandb-project", "chess-slm-rlvr"]
print("B running:", " ".join(cmd))
proc = subprocess.Popen(cmd)
try:
    rc = proc.wait(timeout=720*60)
except subprocess.TimeoutExpired:
    proc.send_signal(signal.SIGINT)
    rc = proc.wait(timeout=90)
if rc != 0:
    raise SystemExit(f"B failed: {rc}")
print("B done")
'''.strip()

def main():
    nb = _notebook([
        _md("# RLVR Base 5k — GRPO on gemma-4-E2B (no SFT) — 2-step gate + A1/B 200-step"),
        _md("Pool: results/rlvr-pool/train-5k.jsonl (5k stratified 30/40/30) — base gemma, P1 endorsed-only, Dr.GRPO, G8 256, HF checkpoints every 300s. Push via `kaggle kernels push -p notebooks/push_rlvr_base_5k`"),
        _code(CLONE_CELL),
        _code(DEPS_CELL),
        _code(FETCH_CELL),
        _code(GATE_CELL),
        _code(TRAIN_A1_CELL),
        _code(TRAIN_B_CELL),
    ])
    out = NB_DIR / "kaggle_rlvr_base_5k.ipynb"
    out.write_text(json.dumps(nb, indent=2))
    print(f"wrote {out}")
    # push dir
    push_dir = NB_DIR / "push_rlvr_base_5k"
    push_dir.mkdir(exist_ok=True)
    (push_dir / "kaggle_rlvr_base_5k.ipynb").write_text(json.dumps(nb, indent=2))
    (push_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": "vedangpandeyyy/rlvr-base-5k",
        "title": "RLVR Base 5k — GRPO gemma-4-E2B",
        "code_file": "kaggle_rlvr_base_5k.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": []
    }, indent=2))
    print(f"push dir {push_dir} ready — run: kaggle kernels push -p {push_dir}")

if __name__ == "__main__":
    main()
