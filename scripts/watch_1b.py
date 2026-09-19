"""Keep the 1B continuation alive across Kaggle session limits and accounts.

- Done when ccgavn-5m-seed0/checkpoint-1,620,000 exists on HF.
- If no <account>/ccgavn-1b kernel is RUNNING/QUEUED, push the continuation
  kernel to the account with the most free GPU quota (metadata generated per
  account; the kernel itself waits for checkpoint-320000 + the 1B corpus).

Called by .github/workflows/watch-1b.yml every 30 minutes; also safe locally.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HF_REPO = "vedangfake/chess-slm-benchmark"
RUN = "ccgavn-5m-seed0"
FINAL_STEP = 1_620_000
ACCOUNTS = ["vedanggggg", "vedangpandeyyy", "softmaxsimp", "samaltmannnn", "shoumikmitra"]
KERNEL_DIR = ROOT / "kernels" / "ccgavn-1b"


def ensure_creds_files() -> None:
    """CI layout: accounts with a profiles/<acct>/access_token or the default
    ~/.kaggle/access_token (vedanggggg) / kaggle.json (vedangpandeyyy)."""
    prof = Path.home() / ".kaggle" / "profiles"
    prof.mkdir(parents=True, exist_ok=True)
    for acct in ACCOUNTS:
        d = prof / acct
        d.mkdir(parents=True, exist_ok=True)
        tok = d / "access_token"
        if not tok.exists():
            default = Path.home() / ".kaggle" / "access_token"
            if acct == "vedanggggg" and default.exists():
                tok.write_text(default.read_text())


def hf_files() -> set[str]:
    from huggingface_hub import HfApi
    sys.path.insert(0, str(ROOT / "scripts"))
    from kaggle_checkpoint import hf_token
    return set(HfApi(token=hf_token(ROOT)).list_repo_files(HF_REPO, repo_type="dataset"))


def gpu_quota(account: str) -> float:
    sys.path.insert(0, str(ROOT / "scripts"))
    from launch_trainers import env_for_account
    try:
        r = subprocess.run([sys.executable, "-m", "kaggle", "quota"],
                           env=env_for_account(account), capture_output=True,
                           text=True, timeout=90)
        for line in (r.stdout or "").splitlines():
            parts = line.split()
            if len(parts) >= 4 and parts[0] == "GPU":
                return float(parts[2].rstrip("h"))
    except Exception:
        pass
    return -1.0


def kernel_status(account: str) -> str:
    sys.path.insert(0, str(ROOT / "scripts"))
    from launch_trainers import env_for_account
    try:
        r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "status",
                            f"{account}/ccgavn-1b"], env=env_for_account(account),
                           capture_output=True, text=True, timeout=90)
        return (r.stdout or r.stderr).strip()
    except Exception as exc:
        return f"status error: {exc}"


def push(account: str) -> tuple[int, str]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from launch_trainers import env_for_account
    with tempfile.TemporaryDirectory(prefix=f"ccgavn1b_{account}_") as tmp:
        tmpd = Path(tmp)
        (tmpd / "train_1b.py").write_text((KERNEL_DIR / "train_1b.py").read_text())
        (tmpd / "shard_rows.json").write_text(
            (ROOT / "kernels" / "build-2b" / "shard_rows.json").read_text())
        cred_ds = {"vedanggggg": "vedanggggg/chess-creds",
                   "vedangpandeyyy": "vedangpandeyyy/chess-creds",
                   "softmaxsimp": "softmaxsimp/chess-creds",
                   "samaltmannnn": "samaltmannnn/hf-creds",
                   "shoumikmitra": "shoumikmitra/hf-creds"}[account]
        (tmpd / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{account}/ccgavn-1b", "title": "ccgavn-1b",
            "code_file": "train_1b.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [cred_ds],
            "competition_sources": [], "kernel_sources": []}, indent=1))
        r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push",
                            "-p", str(tmpd)], env=env_for_account(account),
                           capture_output=True, text=True, timeout=300)
        out = (r.stdout + r.stderr).strip()
        print(f"[1b] push {account}: rc={r.returncode} {out[-200:]}", flush=True)
        return r.returncode, out


def main() -> None:
    ensure_creds_files()
    files = hf_files()
    if f"{RUN}/checkpoint-{FINAL_STEP}/config.json" in files:
        print("[1b] DONE: checkpoint-1,620,000 exists; nothing to do")
        return
    # Readiness gate: never hold a GPU kernel while waiting. A Kaggle GPU
    # kernel burns quota per wall-clock hour regardless of utilisation, so the
    # continuation is only pushed once BOTH prerequisites exist.
    if f"{RUN}/checkpoint-320000/config.json" not in files or \
            f"{RUN}/checkpoint-320000/state.pt" not in files:
        print("[1b] waiting: checkpoint-320000 not on HF yet; not pushing")
        return
    rows_map = json.loads((ROOT / "kernels" / "build-2b" / "shard_rows.json").read_text())
    done_rows = sum(int(v) for k, v in rows_map.items()
                    if f"chessbench-full-build/shard-{k}/teacher_logp.npy" in files)
    if done_rows < 920_000_000:
        print(f"[1b] waiting: corpus {done_rows/1e6:.0f}M / 920M new rows; not pushing")
        return
    print(f"[1b] prerequisites ready (corpus {done_rows/1e6:.0f}M new rows); arming training")
    for acct in ACCOUNTS:
        st = kernel_status(acct)
        if "RUNNING" in st or "QUEUED" in st or "PENDING" in st:
            print(f"[1b] {acct}/ccgavn-1b active: {st[-60:]}")
            return
    quotas = {a: gpu_quota(a) for a in ACCOUNTS}
    usable = {a: q for a, q in quotas.items() if q != 0}
    if not usable:
        print(f"[1b] quotas={quotas}; every account reports 0 GPU hours; "
              "waiting for the weekly refresh")
        return
    best = max(usable, key=usable.get)
    print(f"[1b] no active kernel; quotas={quotas}; pushing on {best}")
    time.sleep(120)  # let a just-pushed kernel appear as RUNNING
    for acct in ACCOUNTS:
        st = kernel_status(acct)
        if "RUNNING" in st or "QUEUED" in st or "PENDING" in st:
            print(f"[1b] {acct}/ccgavn-1b became active; skipping push")
            return
    for acct in sorted(usable, key=usable.get, reverse=True):
        rc, out = push(acct)
        if "successfully pushed" in out.lower():
            print(f"[1b] pushed on {acct}")
            return
        print(f"[1b] {acct} push failed; trying next account")


if __name__ == "__main__":
    main()
