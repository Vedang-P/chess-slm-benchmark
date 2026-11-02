"""Publish verified cont1 state (main run and sprint arms) to the live Worker monitor.

Only checkpoint metrics that are new since the last tick are fetched: the Worker
keeps the curve history in KV and merges whatever it is sent.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.cont_config import (ACCOUNTS, ARM_RUN_PREFIX, ARMS, BASE_STEP, HF_REPO,  # noqa: E402
                                 LEARNER_SLUG, STATUS_PREFIX)
from scripts.kaggle_checkpoint import api as make_hf_api  # noqa: E402
from scripts.kaggle_pool import gpu_quota  # noqa: E402
from scripts.watch_cont import MAIN_RUN, strip_status  # noqa: E402

RUNS = [MAIN_RUN] + [f"{ARM_RUN_PREFIX}{name}" for name in sorted(ARMS)]
REFERENCE = {"base_puzzles": 7620, "base_mate": 3710, "reference_puzzle_pct": 86.13,
             "gate_pct": 86.13, "reference_mate": 3949}


def fetch_json(client, files: set[str], path: str):
    if path not in files:
        return None
    local = hf_hub_download(HF_REPO, path, repo_type="dataset", token=client.token)
    return json.loads(Path(local).read_text(encoding="utf-8"))


def fetch_text(client, files: set[str], path: str) -> str:
    if path not in files:
        return ""
    local = hf_hub_download(HF_REPO, path, repo_type="dataset", token=client.token)
    return Path(local).read_text(encoding="utf-8")


def checkpoint_steps(files: set[str], run: str) -> list[int]:
    return sorted(int(m.group(1)) for name in files
                  if (m := re.fullmatch(rf"{re.escape(run)}/checkpoint-(\d+)/state\.pt", name))
                  and f"{run}/checkpoint-{m.group(1)}/config.json" in files)


def official_evals(client, files: set[str], run: str) -> list[dict]:
    out = []
    for name in sorted(files):
        m = re.fullmatch(rf"eval-results/{re.escape(run)}-(\d+)k-cont/eval-summary\.json", name)
        if not m:
            continue
        s = fetch_json(client, files, name)
        if s and s.get("complete") is True and s.get("returncode") == 0:
            out.append({"step": int(s["checkpoint"].rsplit("-", 1)[-1]), "mate": s["mate"],
                        "puzzles": {k: s["puzzles"].get(k) for k in ("solved", "total", "pct")},
                        "weights": s.get("weights")})
    return sorted(out, key=lambda x: x["step"])


def run_snapshot(client, files: set[str], run: str, recent: int = 6) -> dict | None:
    steps = checkpoint_steps(files, run)
    status = strip_status(fetch_text(client, files, f"{run}/run-status.txt"))
    if not steps and not status:
        return None
    metrics = []
    for step in steps[-recent:]:
        data = fetch_json(client, files, f"{run}/checkpoint-{step}/metrics.json")
        if data:
            metrics.append({"step": step, "metrics": data})
    config = fetch_json(client, files, f"{run}/checkpoint-{steps[-1]}/config.json") if steps else None
    return {"run": run, "step": steps[-1] if steps else 0, "status": status[:300],
            "baseline": (config or {}).get("baseline"), "recent": metrics,
            "recipe": {k: (config or {}).get(k) for k in
                       ("lr", "ema_decay", "broad_rows", "tactical_groups", "lambda_list", "tau")},
            "evaluations": official_evals(client, files, run)}


def quota_snapshot() -> tuple[dict[str, float | None], list[str]]:
    quotas: dict[str, float | None] = {}
    warnings: list[str] = []
    with ThreadPoolExecutor(max_workers=len(ACCOUNTS)) as pool:
        futures = {a: pool.submit(gpu_quota, a) for a in ACCOUNTS}
        for account, future in futures.items():
            try:
                quotas[account] = future.result()
            except Exception as exc:
                quotas[account] = None
                warnings.append(f"{account}: GPU quota unavailable: {type(exc).__name__}: {exc}")
    return quotas, warnings


def snapshot(state: dict) -> dict:
    client = make_hf_api(ROOT)
    files = set(client.list_repo_files(HF_REPO, repo_type="dataset"))
    runs = {}
    for run in RUNS:
        snap = run_snapshot(client, files, run)
        if snap:
            runs[run] = snap
    quotas, warnings = quota_snapshot()
    live = state.get("accounts") or {}
    accounts = [{"account": f"acct-{i}", "gpu_hours": quotas[a],
                 "learner": str((live.get(a) or {}).get("status", "unknown"))}
                for i, a in enumerate(ACCOUNTS, 1)]
    return {"run": "cont1", "updated_at": datetime.now(timezone.utc).isoformat(),
            "phase": state.get("phase", "unknown"), "supervisor": {
                k: state.get(k) for k in ("phase", "reason", "step", "run", "decision",
                                          "run_status", "updated_at")},
            "label_fleet": state.get("label_fleet"),
            "provider_warnings": warnings, "reference": REFERENCE, "base_step": BASE_STEP,
            "main_run": MAIN_RUN, "runs": runs, "accounts": accounts,
            "links": {"checkpoints": f"https://huggingface.co/datasets/{HF_REPO}/tree/main",
                      "actions": "https://github.com/Vedang-P/chess-slm-benchmark/actions"}}


def publish(state: dict) -> None:
    url = os.environ.get("MONITOR_URL", "").rstrip("/")
    key = os.environ.get("MONITOR_INGEST_KEY", "")
    if not url or not key:
        raise RuntimeError("MONITOR_URL and MONITOR_INGEST_KEY are required")
    payload = snapshot(state)
    request = urllib.request.Request(
        url + "/api/ingest", data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"content-type": "application/json", "x-ingest-key": key,
                 "user-agent": "chess-slm-benchmark-cont1-monitor/1.0"}, method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"monitor ingest returned HTTP {response.status}")
    print(f"[monitor] phase={payload['phase']} runs={sorted(payload['runs'])}", flush=True)
