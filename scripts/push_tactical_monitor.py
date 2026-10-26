"""Publish verified puzzle-continuation state to the live Worker monitor."""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.kaggle_checkpoint import api as make_hf_api  # noqa: E402
from scripts.tactical_config import (  # noqa: E402
    ACCOUNTS, BASE_STEP, CPU_BATCH_SESSIONS_PER_ACCOUNT, DATA_PREFIX,
    FINAL_STEP, HF_REPO, INCLUDED_ROOTS, ROOTS_PER_SHARD, RUN, SHARD_COUNT,
    SOURCE_PREFIX, SOURCE_SHA256, milestones,
)
from scripts.watch_tactical import (  # noqa: E402
    DEMO_PREFIX, DEMO_RUN, STATUS_PREFIX, complete_shards, eval_prefix,
    gpu_quota,
)


def remote_json(client, files: set[str], path: str):
    if path not in files:
        return None
    local = hf_hub_download(HF_REPO, path, repo_type="dataset", token=client.token)
    return json.loads(Path(local).read_text(encoding="utf-8"))


def remote_text(client, files: set[str], path: str) -> str:
    if path not in files:
        return ""
    local = hf_hub_download(HF_REPO, path, repo_type="dataset", token=client.token)
    return Path(local).read_text(encoding="utf-8")


def _watch_state() -> dict:
    path = Path(os.environ.get("TACTICAL_STATE_PATH", ""))
    if not path.is_file():
        return {"phase": "supervisor-error", "reason": "supervisor status file missing"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"phase": "supervisor-error", "reason": "supervisor status is invalid"}


def _roots_from_shards(done: set[int]) -> int:
    return sum(min(ROOTS_PER_SHARD, INCLUDED_ROOTS - tag * ROOTS_PER_SHARD)
               for tag in done)


def snapshot() -> dict:
    client = make_hf_api(ROOT)
    files = set(client.list_repo_files(HF_REPO, repo_type="dataset"))
    watch = _watch_state()
    raw = complete_shards(files)
    taught = complete_shards(files, teacher=True)
    partial = []
    for name in files:
        match = re.fullmatch(rf"{re.escape(DATA_PREFIX)}/shard-(\d{{5}})/progress\.json", name)
        if match and int(match.group(1)) not in raw:
            data = remote_json(client, files, name)
            if (data and (data.get("identity") or {}).get("source_sha256") == SOURCE_SHA256):
                partial.append({"shard": int(match.group(1)),
                                "roots": int(data.get("roots_done", 0)),
                                "rows": int(data.get("rows_done", 0)),
                                "updated_at": data.get("updated_at")})
    steps = sorted({int(match.group(1)) for name in files
                    if (match := re.fullmatch(rf"{re.escape(RUN)}/checkpoint-(\d+)/state\.pt", name))
                    and f"{RUN}/checkpoint-{match.group(1)}/config.json" in files})
    latest = steps[-1] if steps else BASE_STEP
    recent = steps[-40:]
    metrics = []
    for step in recent:
        data = remote_json(client, files, f"{RUN}/checkpoint-{step}/metrics.json")
        if data:
            metrics.append({"step": step, "metrics": data})
    config = remote_json(client, files, f"{RUN}/checkpoint-{latest}/config.json") if steps else None
    evaluations = []
    for step in milestones():
        summary = remote_json(client, files, f"{eval_prefix(step)}/eval-summary.json")
        if (summary and summary.get("complete") is True
                and summary.get("returncode") == 0
                and summary.get("checkpoint") == f"{RUN}/checkpoint-{step}"
                and summary.get("mate", {}).get("total") == 4000
                and summary.get("puzzles", {}).get("total") == 10000):
            evaluations.append({"step": step, "mate": summary["mate"],
                                "puzzles": summary["puzzles"],
                                "prefix": eval_prefix(step)})
    accounts = []
    live_statuses = watch.get("statuses", {})
    for index, account in enumerate(ACCOUNTS, 1):
        state = live_statuses.get(account, {})
        label_workers = state.get("labels", {})
        active_label_workers = sum(
            1 for status in label_workers.values()
            if any(word in str(status) for word in ("RUNNING", "QUEUED", "PENDING")))
        accounts.append({"account": f"acct-{index}", "gpu_hours": gpu_quota(account),
                         "worker": state.get("worker", "unknown"),
                         "eval": state.get("eval", "unknown"),
                         "label_workers_active": active_label_workers,
                         "label_workers_limit": CPU_BATCH_SESSIONS_PER_ACCOUNT})
    demo = remote_json(client, files, f"{DEMO_PREFIX}/eval-summary.json")
    return {
        "run": RUN, "updated_at": datetime.now(timezone.utc).isoformat(),
        "phase": watch.get("phase", "unknown"), "supervisor": watch,
        "demo": {"complete": bool(demo and demo.get("mate", {}).get("total") == 10
                                   and demo.get("puzzles", {}).get("total") == 20
                                   and f"{DEMO_RUN}/checkpoint-{BASE_STEP + 2}/state.pt" in files),
                 "summary": demo},
        "corpus": {"source_rows": 6_157_341, "excluded": 9_995,
                   "target_roots": INCLUDED_ROOTS, "root_shards": len(raw),
                   "teacher_shards": len(taught), "target_shards": SHARD_COUNT,
                   "root_count": _roots_from_shards(raw) + sum(x["roots"] for x in partial),
                   "teacher_roots": _roots_from_shards(taught),
                   "partial": partial, "source_sha256": SOURCE_SHA256,
                   "stockfish": "16 · 50 ms per legal action",
                   "teacher": "9M Searchless Chess"},
        "training": {"base_step": BASE_STEP, "step": latest, "target": FINAL_STEP,
                     "additional_complete": max(0, latest - BASE_STEP),
                     "additional_target": FINAL_STEP - BASE_STEP,
                     "checkpoint": f"{RUN}/checkpoint-{latest}" if steps else None,
                     "status": remote_text(client, files, f"{RUN}/run-status.txt"),
                     "metrics": metrics[-1]["metrics"] if metrics else None,
                     "checkpoints": metrics,
                     "parameter_count": (config or {}).get("parameter_count"),
                     "lr": (metrics[-1]["metrics"] or {}).get("lr") if metrics else None},
        "evaluations": evaluations,
        "accounts": accounts,
        "links": {"source": f"https://huggingface.co/datasets/{HF_REPO}/tree/main/{SOURCE_PREFIX}",
                  "corpus": f"https://huggingface.co/datasets/{HF_REPO}/tree/main/{DATA_PREFIX}",
                  "checkpoints": f"https://huggingface.co/datasets/{HF_REPO}/tree/main/{RUN}",
                  "demo": f"https://huggingface.co/datasets/{HF_REPO}/tree/main/{DEMO_PREFIX}",
                  "actions": "https://github.com/Vedang-P/chess-slm-benchmark/actions/workflows/pipeline-tick.yml"},
    }


def main() -> None:
    url = os.environ.get("MONITOR_URL", "").rstrip("/")
    key = os.environ.get("MONITOR_INGEST_KEY", "")
    if not url or not key:
        raise RuntimeError("MONITOR_URL and MONITOR_INGEST_KEY are required")
    payload = snapshot()
    request = urllib.request.Request(url + "/api/ingest",
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"content-type": "application/json", "x-ingest-key": key,
                 "user-agent": "chess-slm-benchmark-tactical-monitor/1.0"},
        method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"monitor ingest returned HTTP {response.status}")
    print(f"[monitor] phase={payload['phase']} roots={payload['corpus']['root_count']:,}/"
          f"{INCLUDED_ROOTS:,} step={payload['training']['step']:,}/{FINAL_STEP:,} "
          f"evals={len(payload['evaluations'])}", flush=True)


if __name__ == "__main__":
    main()
