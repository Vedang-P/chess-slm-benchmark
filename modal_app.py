"""Run the chess sweep on Modal GPUs from the terminal.

    pip install modal
    modal token new            # one-time auth (free $30/mo credits)
    modal run modal_app.py     # runs the full sweep on a T4, streams logs live

The monitor (--monitor) publishes progress to the public live repo exactly as
on Kaggle, so the dashboard works identically. Results are stored in a Modal
volume (`chess-results`) and also backed up to the live repo per cell.

Optional secrets: `hf_token` (Modal secret) for the gated Gemma models.
"""
from __future__ import annotations

import modal

app = modal.App("chess-bench")

VOLUME = modal.Volume.from_name("chess-results", create_if_missing=True)
MOUNT = modal.Mount.from_local_dir(
    ".", condition=lambda p: not any(
        part in p.parts for part in (".git", "results", "results_check", "monitor", "venv", ".rstack", ".opencode")
    )
)

IMAGE = (
    modal.Image.debian_slim()
    .pip_install("transformers", "peft", "bitsandbytes", "accelerate", "numpy",
                 "huggingface_hub", "tqdm", "pyyaml", "python-chess", "zstandard")
    .env({"WANDB_DISABLED": "true"})
)


@app.function(
    image=IMAGE,
    gpu="T4",
    timeout=12 * 3600,
    mounts=[MOUNT],
    volumes={"/results": VOLUME},
    secrets=[] if not None else [modal.Secret.from_name("hf_token", required=False)],
    retries=0,
)
def run_sweep(args: str) -> str:
    import subprocess
    import sys

    cmd = [sys.executable, "scripts/run_suite.py", "--monitor", "--monitor-interval", "120",
           "--output_dir", "/results/chess"] + args.split()
    return subprocess.run(cmd, check=False).returncode


@app.local_entrypoint()
def main(
    check: bool = False,
    smoke: bool = False,
    models: str = "",
    tasks: str = "",
):
    """modal run modal_app.py [--check] [--smoke] [--models 'a b'] [--tasks 'x y']"""
    args = []
    if check:
        args.append("--check")
    if smoke:
        args.append("--smoke")
    if models:
        args += ["--models"] + models.split()
    if tasks:
        args += ["--tasks"] + tasks.split()
    rc = run_sweep.remote(" ".join(args))
    print(f"sweep finished rc={rc}")
