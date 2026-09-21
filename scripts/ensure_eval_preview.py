"""Push the preview-eval kernel whenever a milestone checkpoint of
ccgavn-5m-seed0 exists on HF without an eval summary. Idempotent: does nothing
when all reached milestones are evaluated or the kernel is already active.

The eval kernel is pushed to the account with the most free GPU quota (the
owner account's quota can be exhausted by the training session), so both the
kernel status checks and the push use that account's credentials. Rewriting
the kernel id per account mirrors scripts/watch_1b.py.

Never holds a GPU session: the kernel is pushed only when work is ready.
Called by .github/workflows/pipeline-tick.yml; safe locally
(`python3 scripts/ensure_eval_preview.py --check-only` never pushes).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HF_REPO = "vedangfake/chess-slm-benchmark"
RUN = "ccgavn-5m-seed0"
KERNEL_DIR = ROOT / "kernels" / "eval-preview"
ACCOUNTS = ["vedanggggg", "vedangpandeyyy", "softmaxsimp", "samaltmannnn", "shoumikmitra"]
CRED_DS = {"vedanggggg": "vedanggggg/chess-creds",
           "vedangpandeyyy": "vedangpandeyyy/chess-creds",
           "softmaxsimp": "softmaxsimp/chess-creds",
           "samaltmannnn": "samaltmannnn/hf-creds",
           "shoumikmitra": "shoumikmitra/hf-creds"}
ACTIVE = ("RUNNING", "QUEUED", "PENDING")
TARGETS = [400_000, 500_000, 600_000, 700_000, 800_000, 900_000, 1_000_000,
           1_100_000, 1_200_000, 1_300_000, 1_400_000, 1_500_000, 1_600_000,
           1_620_000]


def out_prefix(step: int) -> str:
    k = step // 1000
    return (f"eval-results/{RUN}-{k}k-frozen" if step == 1_620_000
            else f"eval-results/{RUN}-{k}k-preview")


def hf_files() -> set[str]:
    from huggingface_hub import HfApi
    sys.path.insert(0, str(ROOT / "scripts"))
    from kaggle_checkpoint import hf_token
    return set(HfApi(token=hf_token(ROOT)).list_repo_files(HF_REPO, repo_type="dataset"))


def env_for(account: str) -> dict:
    sys.path.insert(0, str(ROOT / "scripts"))
    from launch_trainers import env_for_account
    return env_for_account(account)


def kernel_status(account: str) -> str:
    try:
        r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "status",
                            f"{account}/eval-ccgavn-preview"], env=env_for(account),
                           capture_output=True, text=True, timeout=90)
        return (r.stdout or r.stderr).strip()
    except Exception as exc:
        return f"status error: {exc}"


def gpu_quota(account: str) -> float:
    try:
        r = subprocess.run([sys.executable, "-m", "kaggle", "quota"], env=env_for(account),
                           capture_output=True, text=True, timeout=90)
        for line in (r.stdout or "").splitlines():
            parts = line.split()
            if len(parts) >= 4 and parts[0] == "GPU":
                return float(parts[2].rstrip("h"))
    except Exception:
        pass
    return -1.0


def push(account: str) -> tuple[int, str]:
    with tempfile.TemporaryDirectory(prefix=f"evalprev_{account}_") as tmp:
        tmpd = Path(tmp)
        (tmpd / "eval_preview.py").write_text(
            (KERNEL_DIR / "eval_preview.py").read_text(encoding="utf-8"), encoding="utf-8")
        (tmpd / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{account}/eval-ccgavn-preview", "title": "eval-ccgavn-preview",
            "code_file": "eval_preview.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [CRED_DS[account]],
            "competition_sources": [], "kernel_sources": []}, indent=1))
        r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push",
                            "-p", str(tmpd)], env=env_for(account),
                           capture_output=True, text=True, timeout=300)
        out = (r.stdout + r.stderr).strip()
        print(f"[preview] push {account}: rc={r.returncode} {out[-200:]}", flush=True)
        return r.returncode, out


def main(check_only: bool = False) -> None:
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
    for acct in ACCOUNTS:
        st = kernel_status(acct)
        if any(a in st for a in ACTIVE):
            print(f"[preview] eval kernel active on {acct}: {st[-60:]}")
            return
    if check_only:
        print("[preview] check-only: no active kernel; a real run would push now")
        return
    quotas = {a: gpu_quota(a) for a in ACCOUNTS}
    usable = {a: q for a, q in quotas.items() if q != 0}
    if not usable:
        print(f"[preview] quotas={quotas}; every account reports 0 GPU hours; "
              "waiting for the weekly refresh")
        return
    best = max(usable, key=usable.get)
    print(f"[preview] no active kernel; quotas={quotas}; pushing on {best}")
    time.sleep(120)
    for acct in ACCOUNTS:
        st = kernel_status(acct)
        if any(a in st for a in ACTIVE):
            print(f"[preview] {acct} kernel became active; skipping push")
            return
    for acct in sorted(usable, key=usable.get, reverse=True):
        rc, out = push(acct)
        if "successfully pushed" in out.lower():
            print(f"[preview] pushed on {acct}")
            return
        print(f"[preview] {acct} push failed; trying next account")


if __name__ == "__main__":
    main(check_only="--check-only" in sys.argv[1:])
