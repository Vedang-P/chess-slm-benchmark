"""Keep the corrected-v2 stage-2B continuation alive across Kaggle sessions.

- The corrected continuation writes to a NEW prefix (`RUN`, e.g.
  ccgavn-5m-seed0-v2); the legacy `ccgavn-5m-seed0` prefix stays frozen as the
  warm-start source and historical record.
- Stage start = the latest complete checkpoint under the legacy prefix on the
  first push; afterwards it is read back from the new prefix's checkpoint
  config, so the value can never drift between re-pushes.
- Done when `RUN/checkpoint-3,896,658` exists on HF.
- Gate: the frozen stage-2B corpus (configs/ccgavn-2b-shard-tags.json, 102
  tags) must be fully on HF plus the warm-start checkpoint. The run/stage/tag
  values are embedded into the pushed kernel code (Kaggle uploads only the
  code file), so the kernel sees exactly the same set.
- If no <account>/ccgavn-2b kernel is RUNNING/QUEUED, push the continuation
  kernel to the account with the most free GPU quota.

Training is DISARMED by default: pushing/committing this code must never start
a run by itself. Arm explicitly with the marker file configs/ccgavn-2b-ARMED
or CCGAVN2B_ARMED=1 in the workflow environment.

Called by .github/workflows/pipeline-tick.yml every ~10-30 minutes; also safe
locally (`python3 scripts/watch_2b.py --check-only` never pushes).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HF_REPO = "vedangfake/chess-slm-benchmark"
RUN = "ccgavn-5m-seed0-v2"        # corrected-v2 continuation prefix
INIT_RUN = "ccgavn-5m-seed0"      # legacy warm-start prefix (frozen)
FINAL_STEP = 3_896_658
PREFIX = "chessbench-full-build"
ACCOUNTS = ["vedanggggg", "vedangpandeyyy", "softmaxsimp", "samaltmannnn", "shoumikmitra"]
KERNEL_DIR = ROOT / "kernels" / "ccgavn-2b"
TAGS_FILE = ROOT / "configs" / "ccgavn-2b-shard-tags.json"
CONFIG_MARKER = "__CCGAVN2B_CONFIG__"
ARMED_MARKER = ROOT / "configs" / "ccgavn-2b-ARMED"
CRASH_LOOP_LIMIT = 3          # failed resumes since the last checkpoint
CRASH_BACKOFF_S = 3600        # ... then retry at most once per hour


def armed() -> bool:
    """Training must never start as a side effect of a code push."""
    return os.environ.get("CCGAVN2B_ARMED") == "1" or ARMED_MARKER.exists()


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


def _hf():
    sys.path.insert(0, str(ROOT / "scripts"))
    from kaggle_checkpoint import api as make_api
    return make_api(ROOT)


def git_head() -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                   text=True).strip()


def resume_target(hf) -> tuple[int, bool, str]:
    """(stage start, resuming the corrected run, source commit). New-prefix
    values if the corrected run exists; otherwise the latest complete legacy
    checkpoint (the warm-start point) and the CI checkout."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from kaggle_checkpoint import latest_remote_checkpoint
    from huggingface_hub import hf_hub_download
    name = latest_remote_checkpoint(hf, HF_REPO, RUN)
    if name is not None:
        path = hf_hub_download(HF_REPO, f"{RUN}/{name}/config.json",
                               repo_type="dataset", token=hf.token)
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
        return (int(cfg["stage_start_step"]), True,
                str(cfg.get("source_commit") or git_head()))
    legacy = latest_remote_checkpoint(hf, HF_REPO, INIT_RUN)
    if legacy is None:
        raise RuntimeError(f"no complete warm-start checkpoint under {INIT_RUN}")
    return int(legacy.split("-")[-1]), False, git_head()


def render_kernel(cfg: dict) -> str:
    src = (KERNEL_DIR / "train_2b.py").read_text(encoding="utf-8")
    placeholder = f'"""{CONFIG_MARKER}"""'
    if placeholder not in src:
        raise RuntimeError(
            f"{KERNEL_DIR / 'train_2b.py'} lost its {placeholder} placeholder")
    return src.replace(placeholder, '"""' + json.dumps(cfg) + '"""')


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
    sys.path.insert(0, str(ROOT / "scripts"))
    from kaggle_checkpoint import hf_token
    from huggingface_hub import HfApi
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
                            f"{account}/ccgavn-2b"], env=env_for_account(account),
                           capture_output=True, text=True, timeout=90)
        return (r.stdout or r.stderr).strip()
    except Exception as exc:
        return f"status error: {exc}"


def push(account: str, cfg: dict) -> tuple[int, str]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from launch_trainers import env_for_account
    with tempfile.TemporaryDirectory(prefix=f"ccgavn2b_{account}_") as tmp:
        tmpd = Path(tmp)
        (tmpd / "train_2b.py").write_text(render_kernel(cfg), encoding="utf-8")
        cred_ds = {"vedanggggg": "vedanggggg/chess-creds",
                   "vedangpandeyyy": "vedangpandeyyy/chess-creds",
                   "softmaxsimp": "softmaxsimp/chess-creds",
                   "samaltmannnn": "samaltmannnn/hf-creds",
                   "shoumikmitra": "shoumikmitra/hf-creds"}[account]
        (tmpd / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{account}/ccgavn-2b", "title": "ccgavn-2b",
            "code_file": "train_2b.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [cred_ds],
            "competition_sources": [], "kernel_sources": []}, indent=1))
        r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push",
                            "-p", str(tmpd)], env=env_for_account(account),
                           capture_output=True, text=True, timeout=300)
        out = (r.stdout + r.stderr).strip()
        print(f"[2b] push {account}: rc={r.returncode} {out[-200:]}", flush=True)
        return r.returncode, out


def main(check_only: bool = False) -> None:
    ensure_creds_files()
    tags = frozen_tags()
    hf = _hf()
    files = hf_files()
    if f"{RUN}/checkpoint-{FINAL_STEP}/config.json" in files:
        print(f"[2b] DONE: {RUN}/checkpoint-{FINAL_STEP} exists; nothing to do")
        return
    # Readiness gate: never hold a GPU kernel while waiting. A Kaggle GPU
    # kernel burns quota per wall-clock hour regardless of utilisation, so the
    # continuation is only pushed once BOTH prerequisites exist.
    missing = missing_shards(tags, files)
    if missing:
        shown = ", ".join(missing[:5]) + ("..." if len(missing) > 5 else "")
        print(f"[2b] waiting: corpus {len(tags) - len(missing)}/{len(tags)} stage-2B "
              f"shards on HF; missing {shown}; not pushing")
        return
    start, resuming, commit = resume_target(hf)
    init_name = f"checkpoint-{start}"
    if not resuming and (f"{INIT_RUN}/{init_name}/config.json" not in files or
                         f"{INIT_RUN}/{init_name}/state.pt" not in files):
        print(f"[2b] waiting: warm-start {INIT_RUN}/{init_name} not on HF yet; not pushing")
        return
    print(f"[2b] prerequisites ready ({len(tags)} stage-2B corpus shards; "
          f"{'resuming' if resuming else 'fresh'} {RUN} from "
          f"stage step {start})")
    cycles, crash_age = recent_crash_cycles()
    looping = cycles >= CRASH_LOOP_LIMIT and crash_age < CRASH_BACKOFF_S
    if looping:
        print(f"[2b] CRASH LOOP: {cycles} failed resume(s) since the last checkpoint, "
              f"newest {crash_age/60:.0f} min ago; backing off for "
              f"~{(CRASH_BACKOFF_S - crash_age)/60:.0f} min. "
              f"Inspect {RUN}/run-status.txt on HF, fix, then it resumes.")
    if check_only:
        print(f"[2b] check-only: {'crash loop detected' if looping else 'gate passed'}; "
              f"armed={armed()}; a real run would push only when armed and not looping")
        return
    if looping:
        return
    if not armed():
        print("[2b] DISARMED: prerequisites are satisfied but training must not "
              "start as a side effect of a code push. Arm with the marker file "
              f"{ARMED_MARKER} (or CCGAVN2B_ARMED=1) and re-run.")
        return
    for acct in ACCOUNTS:
        st = kernel_status(acct)
        if "RUNNING" in st or "QUEUED" in st or "PENDING" in st:
            print(f"[2b] {acct}/ccgavn-2b active: {st[-60:]}")
            return
    quotas = {a: gpu_quota(a) for a in ACCOUNTS}
    usable = {a: q for a, q in quotas.items() if q != 0}
    if not usable:
        print(f"[2b] quotas={quotas}; every account reports 0 GPU hours; "
              "waiting for the weekly refresh")
        return
    best = max(usable, key=usable.get)
    print(f"[2b] no active kernel; quotas={quotas}; pushing on {best}")
    time.sleep(120)  # let a just-pushed kernel appear as RUNNING
    for acct in ACCOUNTS:
        st = kernel_status(acct)
        if "RUNNING" in st or "QUEUED" in st or "PENDING" in st:
            print(f"[2b] {acct}/ccgavn-2b became active; skipping push")
            return
    cfg = {"run": RUN, "init_run": INIT_RUN, "start_step": start,
           "total_steps": FINAL_STEP, "source_commit": commit, "tags": tags}
    for acct in sorted(usable, key=usable.get, reverse=True):
        rc, out = push(acct, cfg)
        if "successfully pushed" in out.lower():
            print(f"[2b] pushed on {acct}")
            return
        print(f"[2b] {acct} push failed; trying next account")


if __name__ == "__main__":
    main(check_only="--check-only" in sys.argv[1:])
