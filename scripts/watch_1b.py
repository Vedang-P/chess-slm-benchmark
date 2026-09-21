"""Keep the 1B continuation alive across Kaggle session limits and accounts.

- Done when ccgavn-5m-seed0/checkpoint-1,620,000 exists on HF.
- Gate: the FROZEN 1B-first corpus (configs/ccgavn-1b-shard-tags.json) must be
  fully on HF plus checkpoint-320000. The frozen list is also embedded into the
  pushed kernel code (Kaggle uploads only the code file), so the kernel sees
  exactly the same set.
- If no <account>/ccgavn-1b kernel is RUNNING/QUEUED, push the continuation
  kernel to the account with the most free GPU quota.

Called by .github/workflows/pipeline-tick.yml every ~10-30 minutes; also safe
locally (`python3 scripts/watch_1b.py --check-only` never pushes).
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
FINAL_STEP = 1_620_000
PREFIX = "chessbench-full-build"
ACCOUNTS = ["vedanggggg", "vedangpandeyyy", "softmaxsimp", "samaltmannnn", "shoumikmitra"]
KERNEL_DIR = ROOT / "kernels" / "ccgavn-1b"
TAGS_FILE = ROOT / "configs" / "ccgavn-1b-shard-tags.json"
TAGS_MARKER = "__CCGAVN1B_TRAIN_TAGS__"
CRASH_LOOP_LIMIT = 3          # failed resumes since the last checkpoint
CRASH_BACKOFF_S = 3600        # ... then retry at most once per hour


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


def frozen_tags() -> list[str]:
    payload = json.loads(TAGS_FILE.read_text(encoding="utf-8"))
    tags = payload.get("tags") if isinstance(payload, dict) else payload
    if not tags:
        raise RuntimeError(f"{TAGS_FILE} contains no shard tags")
    return sorted({str(t) for t in tags})


def missing_shards(tags: list[str], files: set[str]) -> list[str]:
    return [t for t in tags
            if f"{PREFIX}/shard-{t}/train_set.npz" not in files
            or f"{PREFIX}/shard-{t}/teacher_logp.npy" not in files]


def render_kernel(tags: list[str]) -> str:
    src = (KERNEL_DIR / "train_1b.py").read_text(encoding="utf-8")
    placeholder = f'"""{TAGS_MARKER}"""'
    if placeholder not in src:
        raise RuntimeError(
            f"{KERNEL_DIR / 'train_1b.py'} lost its {placeholder} placeholder")
    return src.replace(placeholder, '"""' + json.dumps(tags) + '"""')


def recent_crash_cycles() -> tuple[int, float]:
    """Failure-status uploads newer than the newest checkpoint, plus the age
    (seconds) of the newest one, from the HF commit log. A deterministic bug
    makes every re-push crash before any new checkpoint appears; after a few
    of those the watcher backs off instead of burning GPU quota silently."""
    import datetime
    from huggingface_hub import HfApi
    sys.path.insert(0, str(ROOT / "scripts"))
    from kaggle_checkpoint import hf_token
    commits = HfApi(token=hf_token(ROOT)).list_repo_commits(HF_REPO, repo_type="dataset")
    now = datetime.datetime.now(datetime.timezone.utc)
    cycles = 0
    newest_age = float("inf")
    for c in commits:
        created = c.created_at if c.created_at.tzinfo else c.created_at.replace(
            tzinfo=datetime.timezone.utc)
        age = (now - created).total_seconds()
        if age > 6 * 3600:
            break
        if re.search(rf"{RUN}/checkpoint-\d+/metrics\.json", c.title):
            break
        if c.title.startswith(f"Upload {RUN}/run-status.txt"):
            cycles += 1
            newest_age = min(newest_age, age)
    return cycles, newest_age


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


def push(account: str, tags: list[str]) -> tuple[int, str]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from launch_trainers import env_for_account
    with tempfile.TemporaryDirectory(prefix=f"ccgavn1b_{account}_") as tmp:
        tmpd = Path(tmp)
        (tmpd / "train_1b.py").write_text(render_kernel(tags), encoding="utf-8")
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


def main(check_only: bool = False) -> None:
    ensure_creds_files()
    tags = frozen_tags()
    files = hf_files()
    if f"{RUN}/checkpoint-{FINAL_STEP}/config.json" in files:
        print("[1b] DONE: checkpoint-1,620,000 exists; nothing to do")
        return
    # Readiness gate: never hold a GPU kernel while waiting. A Kaggle GPU
    # kernel burns quota per wall-clock hour regardless of utilisation, so the
    # continuation is only pushed once BOTH prerequisites exist.
    missing = missing_shards(tags, files)
    if missing:
        shown = ", ".join(missing[:5]) + ("..." if len(missing) > 5 else "")
        print(f"[1b] waiting: corpus {len(tags) - len(missing)}/{len(tags)} frozen "
              f"shards on HF; missing {shown}; not pushing")
        return
    if f"{RUN}/checkpoint-320000/config.json" not in files or \
            f"{RUN}/checkpoint-320000/state.pt" not in files:
        print("[1b] waiting: checkpoint-320000 not on HF yet; not pushing")
        return
    print(f"[1b] prerequisites ready ({len(tags)} frozen corpus shards); arming training")
    cycles, crash_age = recent_crash_cycles()
    looping = cycles >= CRASH_LOOP_LIMIT and crash_age < CRASH_BACKOFF_S
    if looping:
        print(f"[1b] CRASH LOOP: {cycles} failed resume(s) since the last checkpoint, "
              f"newest {crash_age/60:.0f} min ago; backing off for "
              f"~{(CRASH_BACKOFF_S - crash_age)/60:.0f} min. "
              "Inspect ccgavn-5m-seed0/run-status.txt on HF, fix, then it resumes.")
    if check_only:
        print(f"[1b] check-only: {'crash loop detected' if looping else 'gate passed'}; "
              "a real run would push the kernel when not looping")
        return
    if looping:
        return
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
        rc, out = push(acct, tags)
        if "successfully pushed" in out.lower():
            print(f"[1b] pushed on {acct}")
            return
        print(f"[1b] {acct} push failed; trying next account")


if __name__ == "__main__":
    main(check_only="--check-only" in sys.argv[1:])
