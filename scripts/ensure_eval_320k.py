"""Safety net: ensure the automatic 320k eval kernel is running once the
checkpoint exists. Push it from CI if it is not active. Idempotent: exits
without action when the eval is done, still training, or already running.

Called by .github/workflows/ensure-eval-320k.yml; also safe to run locally
(requires kaggle credentials for the owning account).
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HF_REPO = "vedangfake/chess-slm-benchmark"
CKPT = "ccgavn-5m-seed0/checkpoint-320000"
PREFIX = "eval-results/ccgavn-5m-seed0-320k-frozen-2026-09-19"
REF = "vedangpandeyyy/eval-ccgavn-320k-auto"
KERNEL_DIR = ROOT / "kernels" / "eval-ccgavn"


def hf_files() -> set[str]:
    from huggingface_hub import HfApi
    sys.path.insert(0, str(ROOT / "scripts"))
    from kaggle_checkpoint import hf_token
    api = HfApi(token=hf_token(ROOT))
    return set(api.list_repo_files(HF_REPO, repo_type="dataset"))


def kernel_status() -> str:
    r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "status", REF],
                       capture_output=True, text=True, timeout=90)
    return (r.stdout or r.stderr).strip()


def main() -> None:
    files = hf_files()
    if f"{CKPT}/config.json" not in files or f"{CKPT}/state.pt" not in files:
        print("[ensure] checkpoint not on HF yet; nothing to do")
        return
    status_file = f"{PREFIX}/run-status.txt"
    if status_file in files:
        from huggingface_hub import hf_hub_download
        sys.path.insert(0, str(ROOT / "scripts"))
        from kaggle_checkpoint import hf_token
        p = hf_hub_download(HF_REPO, status_file, repo_type="dataset",
                            token=hf_token(ROOT))
        if Path(p).read_text().startswith("DONE"):
            print("[ensure] eval already DONE; nothing to do")
            return
    st = kernel_status()
    if "RUNNING" in st or "QUEUED" in st or "PENDING" in st:
        print(f"[ensure] eval kernel active: {st}")
        return
    print(f"[ensure] kernel not active ({st}); sleeping 120s before re-check")
    time.sleep(120)
    st = kernel_status()
    if "RUNNING" in st or "QUEUED" in st or "PENDING" in st:
        print(f"[ensure] eval kernel became active: {st}")
        return
    r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push",
                        "-p", str(KERNEL_DIR)], capture_output=True, text=True, timeout=300)
    out = (r.stdout + r.stderr).strip()
    print(f"[ensure] push rc={r.returncode}: {out[-300:]}")
    if r.returncode != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
