"""Run one cont1 supervisor/monitor tick without noisy Actions failures.

An operational error is written to Hugging Face and shown on the monitor, and the
workflow itself stays green so a provider outage does not send repeated GitHub
failure notifications.
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
from scripts.cont_config import HF_REPO, STATUS_PREFIX  # noqa: E402
from scripts.hf_upload import upload_file  # noqa: E402
from scripts.kaggle_checkpoint import api as make_hf_api  # noqa: E402
from scripts.push_cont_monitor import publish  # noqa: E402
from scripts.watch_cont import main as supervise  # noqa: E402


def record_failure(message: str) -> None:
    try:
        upload_file(make_hf_api(ROOT), HF_REPO, message.encode(), f"{STATUS_PREFIX}/supervisor.txt",
                    commit_message="Record cont1 supervisor failure", attempts=4, max_elapsed_s=300.0)
    except Exception as exc:
        print(f"[tick] could not persist supervisor failure: {exc}", flush=True)


def main() -> None:
    try:
        state = supervise()
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        state = {"phase": "supervisor-error", "reason": detail,
                 "updated_at": datetime.now(timezone.utc).isoformat()}
        print(f"[tick] supervisor error: {detail}\n{traceback.format_exc()[-2500:]}", flush=True)
        record_failure("FAILED " + state["updated_at"] + " " + detail)
    try:
        publish(state)
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        print(f"[tick] monitor error: {detail}\n{traceback.format_exc()[-2500:]}", flush=True)
        record_failure("FAILED monitor " + datetime.now(timezone.utc).isoformat() + " " + detail)
    print(json.dumps({k: state.get(k) for k in ("phase", "reason", "step")}, default=str), flush=True)


if __name__ == "__main__":
    main()
