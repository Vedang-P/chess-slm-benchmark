"""Publish a small, evidence-backed RL monitor snapshot from GitHub Actions."""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.kaggle_checkpoint import api as make_api  # noqa: E402
from scripts.rl_labels import ACCOUNTS  # noqa: E402
from scripts.watch_rl import (DEPTH, HF_REPO, LABEL_PREFIX, ROOTS_PER_ACCOUNT,
                              RUN, TOTAL_STEPS, TRAIN_SLUG, LABEL_SLUG,
                              available_slugs, kernel_status, quota)  # noqa: E402

EVAL_PREFIX = f"eval-results/{RUN}-{TOTAL_STEPS // 1000}k-final"


def remote_json(api, files: set[str], path: str):
    if path not in files:
        return None
    local = hf_hub_download(HF_REPO, path, repo_type="dataset", token=api.token)
    return json.loads(Path(local).read_text(encoding="utf-8"))


def remote_text(api, files: set[str], path: str):
    if path not in files:
        return ""
    local = hf_hub_download(HF_REPO, path, repo_type="dataset", token=api.token)
    return Path(local).read_text(encoding="utf-8").strip()


def checkpoint_steps(files: set[str]) -> list[int]:
    prefix = f"{RUN}/checkpoint-"
    found = set()
    for name in files:
        if name.startswith(prefix) and name.endswith("/state.pt"):
            try:
                step = int(name[len(prefix):].split("/", 1)[0])
            except ValueError:
                continue
            if f"{RUN}/checkpoint-{step}/config.json" in files:
                found.add(step)
    return sorted(found)


def snapshot() -> dict:
    api = make_api(ROOT)
    files = set(api.list_repo_files(HF_REPO, repo_type="dataset"))
    labels = []
    accounts = []
    for index, account in enumerate(ACCOUNTS, 1):
        state = remote_json(api, files, f"{LABEL_PREFIX}/{account}/state.json")
        status = remote_text(api, files, f"{LABEL_PREFIX}/{account}/run-status.txt")
        known = available_slugs(account)
        label_kernel = kernel_status(account, LABEL_SLUG, known)
        train_kernel = kernel_status(account, TRAIN_SLUG, known)
        labels.append({"account": f"acct-{index}",
                       "count": min(int(state["count"]), ROOTS_PER_ACCOUNT) if state else 0,
                       "target": ROOTS_PER_ACCOUNT,
                       "cursor": int(state.get("cursor", 0)) if state else 0,
                       "shard": (state.get("identity") or {}).get("shard_tag") if state else None,
                       "status": status.splitlines()[-1] if status else "waiting",
                       "kernel": label_kernel})
        accounts.append({"account": f"acct-{index}", "gpu_hours": quota(account),
                         "learner": train_kernel})
    steps = checkpoint_steps(files)
    latest = steps[-1] if steps else 0
    metrics = remote_json(api, files, f"{RUN}/checkpoint-{latest}/metrics.json") if steps else None
    config = remote_json(api, files, f"{RUN}/checkpoint-{latest}/config.json") if steps else None
    checkpoints = []
    for step in steps:
        point = remote_json(api, files, f"{RUN}/checkpoint-{step}/metrics.json")
        if point:
            checkpoints.append({"step": step, "metrics": point})
    evaluation = remote_json(api, files, f"{EVAL_PREFIX}/eval-summary.json")
    if evaluation and not (evaluation.get("complete") and evaluation.get("returncode") == 0
                           and (evaluation.get("mate") or {}).get("total") == 4000
                           and (evaluation.get("puzzles") or {}).get("total") == 10000):
        evaluation = None
    return {"updated_at": datetime.now(timezone.utc).isoformat(),
            "run": RUN, "phase": ("evaluated" if evaluation else
                                   "evaluation" if latest >= TOTAL_STEPS else
                                   "training" if steps else "labels"),
            "labels": labels, "accounts": accounts,
            "label_depth": DEPTH, "label_count": sum(item["count"] for item in labels),
            "label_target": ROOTS_PER_ACCOUNT * len(ACCOUNTS),
            "training": {"step": latest, "target": TOTAL_STEPS,
                         "metrics": metrics, "status": remote_text(api, files, f"{RUN}/run-status.txt"),
                         "checkpoint": f"{RUN}/checkpoint-{latest}" if latest else None,
                         "parameter_count": (config or {}).get("parameter_count"),
                         "config": {key: (config or {}).get(key) for key in
                                    ("algorithm", "batch_positions", "group_size", "ppo_epochs",
                                     "lr", "clip", "kl_coef", "value_coef", "entropy_coef",
                                     "seed", "source_commit", "source_checkpoint",
                                     "manifest_sha256", "exclusion_digest")},
                         "checkpoints": checkpoints},
            "evaluation": evaluation,
            "evaluation_status": remote_text(api, files, f"{EVAL_PREFIX}/run-status.txt"),
            "links": {"github": "https://github.com/Vedang-P/chess-slm-benchmark/actions/workflows/pipeline-tick.yml",
                      "labels": f"https://huggingface.co/datasets/{HF_REPO}/tree/main/{LABEL_PREFIX}",
                      "checkpoints": f"https://huggingface.co/datasets/{HF_REPO}/tree/main/{RUN}",
                      "evaluation": f"https://huggingface.co/datasets/{HF_REPO}/tree/main/{EVAL_PREFIX}"}}


def main() -> None:
    url = os.environ.get("MONITOR_URL", "").rstrip("/")
    key = os.environ.get("MONITOR_INGEST_KEY", "")
    if not url or not key:
        raise RuntimeError("MONITOR_URL and MONITOR_INGEST_KEY are required")
    payload = snapshot()
    request = urllib.request.Request(url + "/api/ingest",
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"content-type": "application/json", "x-ingest-key": key,
                 "user-agent": "chess-slm-benchmark-monitor/1.0"}, method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"monitor ingest returned HTTP {response.status}")
    print(f"[monitor] phase={payload['phase']} labels={payload['label_count']}/"
          f"{payload['label_target']} step={payload['training']['step']}/"
          f"{payload['training']['target']} evaluated={payload['evaluation'] is not None}",
          flush=True)


if __name__ == "__main__":
    main()
