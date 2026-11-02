"""Keep the later-ply labeling fleet alive until every shard is on the Hub.

Twenty-five CPU kernels (five per Kaggle account) each own a fixed slice of the job
list in scripts/ply_data.py and label it with four Stockfish processes. This module is
called from the cont1 supervisor tick (scripts/watch_cont.py). It never touches the
GPU learner: a tick relaunches a worker whose kernel is no longer running while it
still has unfinished shards, and reports everything else. Failures go to Hugging Face
status files and the monitor snapshot, never to a red GitHub run.

Pause the fleet by creating ``puzzle-ply-v1/PAUSE`` on the Hub; delete it to resume.
"""
from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.cont_config import ACCOUNTS, HF_REPO, SEARCHLESS_COMMIT, STATUS_PREFIX  # noqa: E402
from scripts.hf_upload import upload_file  # noqa: E402
from scripts.kaggle_pool import (account_slugs, is_active, kernel_status,  # noqa: E402
                                 push_kernel)
from scripts.ply_data import PLIES, PREFIX, complete_shards, plan_jobs, worker_jobs  # noqa: E402
from scripts.tactical_config import SOURCE_PREFIX, SOURCE_SHA256, STOCKFISH16_COMMIT  # noqa: E402

TEMPLATE = ROOT / "kernels" / "cont" / "label_worker.py"
MARKER = "__LABEL_CONFIG__"
SLOTS_PER_ACCOUNT = 5
WORKERS = len(ACCOUNTS) * SLOTS_PER_ACCOUNT
SLUG_PREFIX = "ccgavn-ply-label-w"
ENGINES = 4
SAVE_EVERY_SECONDS = 3600          # one Hub commit per kernel-hour keeps well under the commit cap
COOLDOWN_S = 900                   # after a push, wait for Kaggle to list the new session
MAX_NO_PROGRESS = 3                # consecutive launches that finished no shard
MAX_LAUNCHES_PER_TICK = 10
PLAN_PATH, PAUSE_PATH = f"{PREFIX}/plan.json", f"{PREFIX}/PAUSE"
FLEET_MARKERS = f"{STATUS_PREFIX}/ply-label-fleet.json"


def worker_slug(worker: int) -> str:
    return f"{SLUG_PREFIX}{worker:02d}"


def worker_status_path(worker: int) -> str:
    return f"{STATUS_PREFIX}/ply-label-w{worker:02d}.txt"


def worker_account(worker: int) -> str:
    return ACCOUNTS[worker // SLOTS_PER_ACCOUNT]


def strip_status(text: str) -> str:
    """HF failure statuses carry a leading '# timestamp' line; drop it."""
    lines = (text or "").strip().splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join(lines).strip()


def unfinished_jobs(order: list[tuple[int, int]], complete: dict[int, set[int]],
                    workers: int = WORKERS) -> dict[int, list[tuple[int, int]]]:
    """Jobs still missing from the Hub, per worker, in that worker's own order."""
    return {w: [(p, s) for p, s in worker_jobs(order, w, workers) if s not in complete.get(p, ())]
            for w in range(workers)}


def decide_worker(*, remaining: int, finished: int, kaggle_status: str | None,
                  worker_status: str, commit: str, marker: dict | None, now: float) -> dict:
    """Pure scheduling decision for one worker from observed state (no I/O).

    `finished` counts the worker's shards on the Hub; a launch that leaves it
    unchanged is a launch without progress.
    """
    if remaining == 0:
        return {"action": "done"}
    if kaggle_status is None:
        return {"action": "wait", "reason": "Kaggle status unavailable"}
    if is_active(kaggle_status):
        return {"action": "active"}
    if marker and now - float(marker.get("at_epoch", 0)) < COOLDOWN_S:
        return {"action": "wait", "reason": "launched moments ago; waiting for Kaggle to list it"}
    text = strip_status(worker_status)
    if text.startswith("FAILED"):
        stamped = re.search(r"source_commit=([0-9a-f]{40})", text)
        # A failure from an older revision must not block the corrected commit;
        # an unstamped failure cannot be attributed, so it blocks.
        if stamped is None or stamped.group(1) == commit:
            return {"action": "blocked", "reason": "worker failed on this commit; push a fix",
                    "detail": text.splitlines()[0][:200]}
    same = bool(marker) and marker.get("commit") == commit and marker.get("finished") == finished
    attempt = int(marker.get("attempt", 1)) + 1 if same else 1
    if attempt > MAX_NO_PROGRESS:
        return {"action": "blocked", "reason": f"{MAX_NO_PROGRESS} launches finished no shard"}
    return {"action": "launch", "attempt": attempt}


def read_text(client, files: set[str], path: str) -> str:
    if path not in files:
        return ""
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(HF_REPO, path, repo_type="dataset", token=client.token)
                ).read_text(encoding="utf-8")


def worker_config(worker: int, commit: str, phase: str = "labels") -> dict:
    return {"phase": phase, "worker_index": worker, "worker_count": WORKERS,
            "account_index": worker // SLOTS_PER_ACCOUNT, "hf_repo": HF_REPO,
            "status_prefix": STATUS_PREFIX, "source_commit": commit,
            "searchless_commit": SEARCHLESS_COMMIT, "stockfish_commit": STOCKFISH16_COMMIT,
            "source_sha256": SOURCE_SHA256, "source_prefix": SOURCE_PREFIX, "plan_prefix": PREFIX,
            "data_prefix": PREFIX, "engines": ENGINES, "save_every_seconds": SAVE_EVERY_SECONDS}


def observe(workers: list[int]) -> dict[int, str | None]:
    """Kaggle status of each worker's kernel (None when the query fails)."""
    def one_account(account: str) -> dict[int, str | None]:
        mine = [w for w in workers if worker_account(w) == account]
        if not mine:
            return {}
        try:
            slugs = account_slugs(account)
        except Exception:
            return {w: None for w in mine}
        out: dict[int, str | None] = {}
        for w in mine:
            try:
                out[w] = kernel_status(account, worker_slug(w), slugs)
            except Exception:
                out[w] = None
        return out

    merged: dict[int, str | None] = {}
    with ThreadPoolExecutor(max_workers=len(ACCOUNTS)) as pool:
        for part in pool.map(one_account, ACCOUNTS):
            merged.update(part)
    return merged


def launch(workers: list[int], commit: str) -> tuple[list[int], dict[int, str]]:
    """Push each worker's kernel (one account at a time, accounts in parallel)."""
    def one_account(account: str) -> tuple[list[int], dict[int, str]]:
        pushed, errors = [], {}
        for w in (w for w in workers if worker_account(w) == account):
            try:
                push_kernel(account, worker_slug(w), TEMPLATE, MARKER,
                            worker_config(w, commit), gpu=False)
                pushed.append(w)
            except Exception as exc:
                errors[w] = f"{type(exc).__name__}: {exc}"[:300]
        return pushed, errors

    pushed: list[int] = []
    errors: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=len(ACCOUNTS)) as pool:
        for ok, bad in pool.map(one_account, ACCOUNTS):
            pushed += ok
            errors.update(bad)
    return sorted(pushed), errors


def tick(client, files: set[str], commit: str, check_only: bool = False) -> dict:
    """One supervision pass; returns a JSON-safe summary for the monitor."""
    now = time.time()
    summary = {"updated_at": datetime.now(timezone.utc).isoformat(), "commit": commit}
    if PLAN_PATH not in files:
        return {**summary, "phase": "disabled", "reason": "no labeling plan on the Hub"}
    plan = json.loads(read_text(client, files, PLAN_PATH))
    order = plan_jobs(plan)
    complete = complete_shards(files, PREFIX, PLIES)
    total = {int(p): int(v["shards"]) for p, v in plan["plies"].items()}
    summary["shards"] = {"total": sum(total.values()),
                         "complete": sum(len(complete[p]) for p in total),
                         "by_ply": {str(p): {"complete": len(complete[p]), "total": total[p]}
                                    for p in total}}
    if PAUSE_PATH in files:
        return {**summary, "phase": "paused", "reason": f"{PAUSE_PATH} exists"}
    pending = unfinished_jobs(order, complete)
    if not any(pending.values()):
        return {**summary, "phase": "complete", "reason": "every ply shard is on the Hub"}
    statuses = observe([w for w in range(WORKERS) if pending[w]])
    markers = json.loads(read_text(client, files, FLEET_MARKERS) or "{}")
    decisions: dict[int, dict] = {}
    for w in range(WORKERS):
        status = statuses.get(w)
        # Only a worker that is down needs its failure note read.
        down = bool(pending[w]) and status is not None and not is_active(status)
        decisions[w] = decide_worker(
            remaining=len(pending[w]), finished=len(worker_jobs(order, w, WORKERS)) - len(pending[w]),
            kaggle_status=status, worker_status=read_text(client, files, worker_status_path(w))
            if down else "", commit=commit, marker=markers.get(f"{w:02d}"), now=now)
    by_action: dict[str, list[int]] = {}
    for w, d in decisions.items():
        by_action.setdefault(d["action"], []).append(w)
    to_launch = by_action.get("launch", [])[:MAX_LAUNCHES_PER_TICK]
    pushed: list[int] = []
    errors: dict[int, str] = {}
    if to_launch and not check_only:
        pushed, errors = launch(to_launch, commit)
        for w in pushed:
            markers[f"{w:02d}"] = {
                "commit": commit, "at_epoch": time.time(), "at": summary["updated_at"],
                "attempt": decisions[w]["attempt"],
                "finished": len(worker_jobs(order, w, WORKERS)) - len(pending[w])}
        if pushed:
            upload_file(client, HF_REPO, json.dumps(markers, sort_keys=True).encode(),
                        FLEET_MARKERS, commit_message=f"Launch {len(pushed)} later-ply label workers",
                        attempts=4, max_elapsed_s=300.0)
    blocked = {f"{w:02d}": decisions[w]["reason"] + (f": {decisions[w]['detail']}"
                                                     if decisions[w].get("detail") else "")
               for w in by_action.get("blocked", [])}
    phase = "blocked" if blocked else ("launched" if pushed else "running")
    return {**summary, "phase": phase,
            "workers": {k: len(v) for k, v in sorted(by_action.items())},
            "launched": pushed, "launch_errors": {f"{w:02d}": e for w, e in errors.items()},
            "blocked": blocked, "remaining": sum(len(v) for v in pending.values()),
            "check_only": check_only}
