"""Run the cont1 main run as back-to-back ~11.5 h learner sessions across the pool.

Exactly one learner is alive at a time. The trainer saves a complete checkpoint
to Hugging Face at its session deadline; the next tick launches a new session on
the account with the most GPU quota, which resumes from that checkpoint. Failures
are reported through Hugging Face status files and the monitor, never through
red GitHub runs, and a crash loop stops itself instead of burning quota.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.cont_config import (A4_RECIPE, ACCOUNTS, COMMON_ARGS, HF_REPO,  # noqa: E402
                                 LEARNER_SLUG, MIN_SESSION_SECONDS, QUOTA_RESERVE_H,
                                 SEARCHLESS_COMMIT, SESSION_SECONDS, STATUS_PREFIX)
from scripts.hf_upload import upload_file  # noqa: E402
from scripts.kaggle_checkpoint import api as make_hf_api  # noqa: E402
from scripts.kaggle_pool import (account_slugs, gpu_quota, is_active, kernel_status,  # noqa: E402
                                 push_kernel)

TEMPLATE = ROOT / "kernels" / "cont" / "worker.py"
# The sprint's best arms (a3, a4) both cleared the gate at step 30,000; a4 (30% tactical,
# lr 5e-5) forgot less of the broad corpus, so the main run is a4 continued without a step
# limit. It resumes arm a4's own checkpoint, optimizer and EMA, with the identical recipe.
MAIN_RUN = "ccgavn-cont1-a4"
MAIN_ARGS: list[str] = A4_RECIPE + [
    "--val-every", "10000", "--ckpt-every", "10000", "--milestone-every", "100000",
    "--guard-after", "40000", "--guard-drop", "3.0",
]
COOLDOWN_S = 600
MAX_NO_PROGRESS = 2
STALL_SECONDS = 3 * 3600   # an active learner with no new checkpoint this long is reported as stalled


def strip_status(text: str) -> str:
    """HF failure statuses carry a leading '# timestamp' line; drop it."""
    lines = [ln for ln in (text or "").strip().splitlines()]
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join(lines).strip()


def latest_step(files: set[str], run: str) -> int:
    steps = [int(m.group(1)) for name in files
             if (m := re.fullmatch(rf"{re.escape(run)}/checkpoint-(\d+)/state\.pt", name))
             and f"{run}/checkpoint-{m.group(1)}/config.json" in files]
    return max(steps, default=0)


def decide(*, run_status: str, accounts: dict[str, dict], head: str, marker: dict | None,
           step: int, now: float) -> dict:
    """Pure scheduling decision from observed state (no I/O)."""
    text = strip_status(run_status)
    if text.startswith("DONE"):
        return {"action": "complete", "reason": text.splitlines()[0]}
    if text.startswith("HALTED"):
        return {"action": "blocked", "reason": text.splitlines()[0]}
    unknown = [a for a, s in accounts.items() if s.get("status") is None]
    if unknown:
        return {"action": "noop", "reason": f"Kaggle status unavailable for {unknown}"}
    active = [a for a, s in accounts.items() if is_active(s["status"])]
    if active:
        stalled = bool(marker) and marker.get("step") == step and \
            now - float(marker.get("at_epoch", now)) > STALL_SECONDS
        return {"action": "noop", "reason": f"learner active on {active}", "stalled": stalled}
    if marker and now - float(marker.get("at_epoch", 0)) < COOLDOWN_S:
        return {"action": "noop", "reason": "launched moments ago; waiting for Kaggle to list it"}
    same = bool(marker) and marker.get("commit") == head and marker.get("step") == step
    attempt = int(marker.get("attempt", 1)) + 1 if same else 1
    is_error = bool(text) and not text.startswith(("IN PROGRESS", "DONE", "HALTED"))
    if is_error and same:
        return {"action": "blocked", "reason": "learner failed on this commit; push a fix",
                "detail": text[:300]}
    if attempt > MAX_NO_PROGRESS:
        return {"action": "blocked", "reason": f"{MAX_NO_PROGRESS} launches made no checkpoint "
                                               f"progress at step {step}; investigate"}
    usable = {a: s["quota"] for a, s in accounts.items()
              if s.get("quota") is not None
              and (s["quota"] - QUOTA_RESERVE_H) * 3600 >= MIN_SESSION_SECONDS}
    if not usable:
        return {"action": "wait", "reason": "no account has enough GPU quota; it resets Fridays 00:00 UTC"}
    account = max(usable, key=usable.get)
    seconds = min(SESSION_SECONDS, int((usable[account] - QUOTA_RESERVE_H) * 3600))
    return {"action": "launch", "account": account, "session_seconds": seconds,
            "attempt": attempt}


def git_head() -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def read_text(client, files: set[str], path: str) -> str:
    if path not in files:
        return ""
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(HF_REPO, path, repo_type="dataset", token=client.token)
                ).read_text(encoding="utf-8")


def observe(accounts=ACCOUNTS) -> dict[str, dict]:
    """Learner status and remaining GPU quota per account (None when unavailable)."""
    def one(account: str) -> tuple[str, dict]:
        out = {"status": None, "quota": None}
        try:
            out["status"] = kernel_status(account, LEARNER_SLUG, account_slugs(account))
        except Exception as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"[:200]
        try:
            out["quota"] = gpu_quota(account)
        except Exception as exc:
            out.setdefault("error", f"{type(exc).__name__}: {exc}"[:200])
        return account, out
    with ThreadPoolExecutor(max_workers=len(accounts)) as pool:
        return dict(pool.map(one, accounts))


def main(check_only: bool = False) -> dict:
    import time
    client = make_hf_api(ROOT)
    files = set(client.list_repo_files(HF_REPO, repo_type="dataset"))
    head = git_head()
    step = latest_step(files, MAIN_RUN)
    run_status = read_text(client, files, f"{MAIN_RUN}/run-status.txt")
    marker_path = f"{STATUS_PREFIX}/launch-{MAIN_RUN}.json"
    marker = json.loads(read_text(client, files, marker_path) or "null")
    accounts = observe()
    decision = decide(run_status=run_status, accounts=accounts, head=head, marker=marker,
                      step=step, now=time.time())
    state = {"phase": decision["action"], "reason": decision.get("reason"), "step": step,
             "run": MAIN_RUN, "commit": head, "decision": decision, "accounts": accounts,
             "run_status": strip_status(run_status)[:300],
             "updated_at": datetime.now(timezone.utc).isoformat(), "check_only": check_only}
    if decision["action"] != "launch" or check_only:
        return state
    account = decision["account"]
    config = {"phase": "main", "run": MAIN_RUN, "hf_repo": HF_REPO, "status_prefix": STATUS_PREFIX,
              "source_commit": head, "searchless_commit": SEARCHLESS_COMMIT,
              "session_seconds": decision["session_seconds"],
              "trainer_args": COMMON_ARGS + MAIN_ARGS, "account_index": ACCOUNTS.index(account)}
    # Push first: a failed push must not count as a launch attempt. If the marker
    # write fails afterwards, the next tick sees the queued learner and does nothing.
    push_kernel(account, LEARNER_SLUG, TEMPLATE, "__CONT_CONFIG__", config, gpu=True)
    state["phase"] = "launched"
    new_marker = {"account": account, "commit": head, "step": step, "attempt": decision["attempt"],
                  "at": state["updated_at"], "at_epoch": time.time()}
    upload_file(client, HF_REPO, json.dumps(new_marker).encode(), marker_path,
                commit_message=f"Launch cont1 session on {account} at step {step}",
                attempts=4, max_elapsed_s=300.0)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    print(json.dumps(main(parser.parse_args().check_only), sort_keys=True, default=str), flush=True)
