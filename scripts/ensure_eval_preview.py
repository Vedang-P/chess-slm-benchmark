"""Push the preview-eval kernel whenever a milestone checkpoint of
ccgavn-5m-seed0 exists on HF without an eval summary. Idempotent: does nothing
when all reached milestones are evaluated or the kernel is already active.

Never holds a GPU session: the kernel is pushed only when work is ready.
Called by .github/workflows/pipeline-tick.yml; safe locally.
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HF_REPO = "vedangfake/chess-slm-benchmark"
RUN = "ccgavn-5m-seed0"
REF = "vedangpandeyyy/eval-ccgavn-preview"
KERNEL_DIR = ROOT / "kernels" / "eval-preview"
TARGETS = [400_000, 500_000, 600_000, 700_000, 800_000, 900_000, 1_000_000, 1_100_000, 1_200_000, 1_300_000, 1_400_000, 1_500_000, 1_600_000, 1_620_000]


def out_prefix(step: int) -> str:
    k = step // 1000
    return (f"eval-results/{RUN}-{k}k-frozen" if step == 1_620_000
            else f"eval-results/{RUN}-{k}k-preview")


def hf_files() -> set[str]:
    from huggingface_hub import HfApi
    sys.path.insert(0, str(ROOT / "scripts"))
    from kaggle_checkpoint import hf_token
    return set(HfApi(token=hf_token(ROOT)).list_repo_files(HF_REPO, repo_type="dataset"))


def kernel_status() -> str:
    r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "status", REF],
                       capture_output=True, text=True, timeout=90)
    return (r.stdout or r.stderr).strip()


def main() -> None:
    files = hf_files()
    steps = [int(m.group(1)) for f in files
             if (m := re.match(rf"{RUN}/checkpoint-(\d+)/state\.pt$", f))]
    if not steps:
        print("[preview] no checkpoints on HF; nothing to do")
        return
    latest = max(steps)
    pending = [t for t in TARGETS
               if t <= latest and f"{out_prefix(t)}/eval-summary.json" not in files]
    if not pending:
        print(f"[preview] latest checkpoint {latest}; all reached milestones evaluated")
        return
    print(f"[preview] latest {latest}; pending: {pending}")
    st = kernel_status()
    if "RUNNING" in st or "QUEUED" in st or "PENDING" in st:
        print(f"[preview] eval kernel active: {st}")
        return
    print(f"[preview] kernel not active ({st}); sleeping 120s before re-check")
    time.sleep(120)
    st = kernel_status()
    if "RUNNING" in st or "QUEUED" in st or "PENDING" in st:
        print(f"[preview] eval kernel became active: {st}")
        return
    r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push",
                        "-p", str(KERNEL_DIR)], capture_output=True, text=True, timeout=300)
    out = (r.stdout + r.stderr).strip()
    print(f"[preview] push rc={r.returncode}: {out[-300:]}")
    if r.returncode != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
