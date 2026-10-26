"""Run one tactical supervisor/monitor tick without noisy Actions failures.

An operational error is persisted to Hugging Face and shown on the monitor.
The workflow itself stays green so a transient provider outage does not send
the user repeated GitHub failure notifications.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.kaggle_checkpoint import api as make_hf_api
from scripts.prune_tactical_checkpoints import main as prune_checkpoints
from scripts.push_tactical_monitor import main as publish_monitor
from scripts.tactical_config import HF_REPO
from scripts.watch_tactical import STATUS_PREFIX, main as supervise


STATE = Path(os.environ.get("TACTICAL_STATE_PATH", "/tmp/tactical-state.json"))


def record_failure(message: str) -> None:
    try:
        client = make_hf_api(ROOT)
        client.upload_file(path_or_fileobj=message.encode(),
                           path_in_repo=f"{STATUS_PREFIX}/supervisor.txt",
                           repo_id=HF_REPO, repo_type="dataset",
                           commit_message="Record tactical pipeline supervisor failure")
    except Exception as exc:
        print(f"[tick] could not persist supervisor failure: {exc}", flush=True)


def main() -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    try:
        state = supervise()
        if state.get("phase") == "blocked":
            print(f"[tick] blocked: {state.get('reason')}", flush=True)
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        state = {"phase": "supervisor-error", "reason": detail,
                 "updated_at": datetime.now(timezone.utc).isoformat()}
        print(f"[tick] supervisor error: {detail}\n{traceback.format_exc()[-2500:]}", flush=True)
        record_failure("FAILED " + state["updated_at"] + " " + detail)
    STATE.write_text(json.dumps(state, sort_keys=True) + "\n", encoding="utf-8")
    os.environ["TACTICAL_STATE_PATH"] = str(STATE)
    try:
        publish_monitor()
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        print(f"[tick] monitor error: {detail}\n{traceback.format_exc()[-2500:]}", flush=True)
        record_failure("FAILED monitor " + datetime.now(timezone.utc).isoformat() + " " + detail)
    if state.get("phase") not in ("supervisor-error", "blocked"):
        try:
            prune_checkpoints()
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            print(f"[tick] retention error: {detail}", flush=True)
            record_failure("FAILED retention " + datetime.now(timezone.utc).isoformat() + " " + detail)
    print(json.dumps(state, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
