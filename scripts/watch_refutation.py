"""Keep the CC-GAVN-R pilot alive across Kaggle sessions and accounts.

The kernel is self-contained: source prep, oracle labels, planner training, and
the final 4k/10k evaluation all resume from Hugging Face. This supervisor only
starts it when no account has it active, and never starts a second copy.
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
KERNEL = "ccgavn-refutation-r0"
DATA_RUN = "refutation-data-r0"
TRAIN_RUN = "ccgavn-refutation-r0-seed0"
EVAL_RUN = "eval-results/ccgavn-refutation-r0"
PHASE_RUN = "refutation-r0"
FINAL_STEPS = 12500
ACCOUNTS = ["vedanggggg", "vedangpandeyyy", "softmaxsimp", "samaltmannnn", "shoumikmitra"]
OTHER_KERNELS = ["ccgavn-2b", "ccgavn-1b", "build-2b-slice"]
CRED_DATASETS = {
    "vedanggggg": "vedanggggg/chess-creds",
    "vedangpandeyyy": "vedangpandeyyy/chess-creds",
    "softmaxsimp": "softmaxsimp/chess-creds",
    "samaltmannnn": "samaltmannnn/hf-creds",
    "shoumikmitra": "shoumikmitra/hf-creds",
}
FAILURE_BACKOFF_S = 3600


def ensure_creds_files() -> None:
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


def hf_text(path: str) -> str:
    from huggingface_hub import hf_hub_download
    sys.path.insert(0, str(ROOT / "scripts"))
    from kaggle_checkpoint import hf_token
    try:
        p = hf_hub_download(HF_REPO, path, repo_type="dataset", token=hf_token(ROOT))
        return Path(p).read_text()
    except Exception:
        return ""


def hf_json(path: str):
    text = hf_text(path)
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def status(account: str, kernel: str = KERNEL) -> str:
    sys.path.insert(0, str(ROOT / "scripts"))
    from kaggle_accounts import env_for_account
    try:
        r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "status",
                            f"{account}/{kernel}"], env=env_for_account(account),
                           capture_output=True, text=True, timeout=90)
        return (r.stdout or r.stderr).strip()
    except Exception as exc:
        return f"status error: {exc}"


def quota(account: str) -> float:
    sys.path.insert(0, str(ROOT / "scripts"))
    from kaggle_accounts import env_for_account
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


def active(text: str) -> bool:
    return any(x in text for x in ("RUNNING", "QUEUED", "PENDING"))


def complete(files: set[str]) -> bool:
    if f"{TRAIN_RUN}/checkpoint-{FINAL_STEPS}/config.json" not in files:
        return False
    for path in files:
        if path.startswith(f"{EVAL_RUN}/") and path.endswith("/summary.json"):
            data = hf_json(path)
            if data and data.get("complete"):
                return True
    return False


def failure_age(files: set[str]) -> float | None:
    if f"{PHASE_RUN}/run-status.txt" not in files:
        return None
    text = hf_text(f"{PHASE_RUN}/run-status.txt")
    if "FAILED" not in text.upper():
        return None
    from datetime import datetime, timezone
    from huggingface_hub import HfApi
    sys.path.insert(0, str(ROOT / "scripts"))
    from kaggle_checkpoint import hf_token
    commits = HfApi(token=hf_token(ROOT)).list_repo_commits(HF_REPO, repo_type="dataset")
    for commit in commits:
        if f"{PHASE_RUN}/run-status.txt" not in commit.title:
            continue
        created = commit.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - created).total_seconds()
    return None


def push(account: str) -> tuple[int, str]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from kaggle_accounts import env_for_account
    with tempfile.TemporaryDirectory(prefix=f"refutation_{account}_") as tmp:
        d = Path(tmp)
        (d / "run.py").write_text(
            (ROOT / "kernels" / "ccgavn-refutation" / "run.py").read_text(), encoding="utf-8")
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{account}/{KERNEL}", "title": KERNEL,
            "code_file": "run.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [CRED_DATASETS[account]],
            "competition_sources": [], "kernel_sources": [],
        }, indent=1), encoding="utf-8")
        r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(d)],
                           env=env_for_account(account), capture_output=True,
                           text=True, timeout=300)
        out = (r.stdout + r.stderr).strip()
        print(f"[refutation] push {account}: rc={r.returncode} {out[-220:]}", flush=True)
        return r.returncode, out


def main(check_only: bool = False) -> None:
    ensure_creds_files()
    files = hf_files()
    if f"ccgavn-5m-seed0-v2/checkpoint-3896658/config.json" not in files or \
            f"ccgavn-5m-seed0-v2/checkpoint-3896658/state.pt" not in files:
        print("[refutation] waiting: base checkpoint is not complete on HF")
        return
    if complete(files):
        print("[refutation] DONE: final checkpoint and evaluation are on HF")
        return
    for acct in ACCOUNTS:
        st = status(acct)
        if active(st):
            print(f"[refutation] {acct}/{KERNEL} active: {st[-80:]}")
            return
    if check_only:
        print("[refutation] check-only: no active kernel; a real run would push")
        return
    age = failure_age(files)
    if age is not None and age < FAILURE_BACKOFF_S:
        print(f"[refutation] latest failure was {age/60:.0f} min ago; backing off "
              f"for ~{(FAILURE_BACKOFF_S-age)/60:.0f} min")
        return
    usable = {}
    for acct in ACCOUNTS:
        busy = [k for k in OTHER_KERNELS if active(status(acct, k))]
        if busy:
            print(f"[refutation] {acct} busy with {busy}; skipping")
            continue
        q = quota(acct)
        if q > 0:
            usable[acct] = q
        else:
            print(f"[refutation] {acct} quota unavailable ({q})")
    if not usable:
        print("[refutation] no account with positive GPU quota and no conflicting kernel")
        return
    print(f"[refutation] no active kernel; quotas={usable}")
    time.sleep(120)
    for acct in ACCOUNTS:
        if active(status(acct)):
            print(f"[refutation] {acct}/{KERNEL} became active; skipping push")
            return
    for acct in sorted(usable, key=usable.get, reverse=True):
        rc, out = push(acct)
        if rc == 0 and "successfully pushed" in out.lower():
            print(f"[refutation] pushed on {acct}")
            return


if __name__ == "__main__":
    main(check_only="--check-only" in sys.argv[1:])
