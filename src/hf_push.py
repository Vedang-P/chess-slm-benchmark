"""Persist completed benchmark results to a free Hugging Face dataset repo.

Every completed cell uploads:
  - <cell>.report.json      the full clean record per sample: prompt, the
                            model's entire thinking chain, final answer,
                            correct/reference answer, verdict, compliance,
                            cache stats, fallback flag
  - <cell>.samples.jsonl    raw per-sample records (same data, jsonl)
  - <cell>.summary.json     per-cell metrics
  - index.json              list of uploaded cells (for the web/recovery)

Repo: vedangfake/chess-bench-results (public dataset, unlimited storage).
Token: HF_WRITE_TOKEN (env / gitignored .env), never committed.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

HF_REPO = "vedangfake/chess-bench-results"
RUN_DIR = "runs"  # top-level folder inside the dataset repo


def resolve_hf_token() -> str | None:
    for name in ("HF_WRITE_TOKEN", "HF_TOKEN"):
        if os.environ.get(name):
            return os.environ[name]
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("HF_WRITE_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _api():
    from huggingface_hub import HfApi

    token = resolve_hf_token()
    if not token:
        raise RuntimeError("no HF_WRITE_TOKEN — set it in .env or as a Kaggle secret")
    api = HfApi(token=token)
    try:
        api.create_repo(repo_id=HF_REPO, repo_type="dataset", exist_ok=True)
    except Exception:
        pass
    return api


def _report_from_samples(samples_path: Path, summary: dict) -> dict:
    """Collapse a cell's raw samples into the clean public format: one
    record per sample with thinking, answers, prompt, and verdict."""
    samples = []
    for line in samples_path.read_text().splitlines():
        if not line.strip():
            continue
        s = json.loads(line)
        samples.append({
            "position_id": s.get("position_id"),
            "prompt": s.get("prompt"),
            "thinking": s.get("reasoning", ""),
            "model_answer": s.get("output", ""),
            "model_move": s.get("move"),
            "correct_answer": s.get("correct"),
            "verdict": s.get("status"),
            "compliance": s.get("compliance"),
            "format": s.get("format"),
            "fallback": s.get("fallback"),
            "cache_hit_tokens": s.get("cache_hit"),
            "cache_miss_tokens": s.get("cache_miss"),
            "latency_ms": s.get("latency_ms"),
            "output_tokens": s.get("output_tokens"),
        })
    return {
        "cell": summary.get("run"),
        "run_id": summary.get("meta", {}).get("run_id"),
        "model": summary.get("meta", {}).get("model"),
        "task": summary.get("meta", {}).get("task"),
        "variant": summary.get("meta", {}).get("prompt_variant"),
        "generated_at": summary.get("finished_at"),
        "metrics": summary.get("metrics", {}),
        "samples": samples,
    }


def upload_cell(output_dir: Path, run_id: str, summary_path: Path,
                only_missing: bool = True) -> None:
    """Upload one completed cell's report + samples + summary to HF."""
    samples_path = summary_path.with_name(
        summary_path.name.replace(".summary.json", ".samples.jsonl"))
    if not samples_path.exists():
        samples_path = None
    summary = json.loads(summary_path.read_text())
    cell = summary.get("run") or summary_path.stem
    base = f"{RUN_DIR}/{run_id or 'local'}/{cell}"

    api = _api()
    uploaded = []
    if samples_path:
        report = _report_from_samples(samples_path, summary)
        report_bytes = json.dumps(report, indent=1, ensure_ascii=False).encode()
        uploaded.append(("report.json", report_bytes))
        uploaded.append(("samples.jsonl", samples_path.read_bytes()))
    uploaded.append(("summary.json", summary_path.read_bytes()))

    for name, data in uploaded:
        path_in_repo = f"{base}.{name}"
        if only_missing and _exists(api, path_in_repo):
            continue
        api.upload_file(path_or_fileobj=data, path_in_repo=path_in_repo,
                        repo_id=HF_REPO, repo_type="dataset",
                        commit_message=f"cell {cell} · {name}")
        print(f"hf: uploaded {path_in_repo}", flush=True)

    _update_index(api, run_id)


def _exists(api, path_in_repo: str) -> bool:
    try:
        api.file_exists(repo_id=HF_REPO, path_in_repo=path_in_repo,
                        repo_type="dataset")
        return True
    except Exception:
        return False


def _update_index(api, run_id: str) -> None:
    prefix = f"{RUN_DIR}/{run_id or 'local'}/"
    try:
        files = api.list_repo_files(repo_id=HF_REPO, repo_type="dataset")
    except Exception:
        return
    cells = sorted({f[len(prefix):].split(".")[0] for f in files
                    if f.startswith(prefix) and f.endswith(".report.json")})
    index = {"run_id": run_id, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
             "cells": cells}
    api.upload_file(path_or_fileobj=json.dumps(index, indent=1).encode(),
                    path_in_repo=f"{RUN_DIR}/{run_id or 'local'}/index.json",
                    repo_id=HF_REPO, repo_type="dataset",
                    commit_message="index update")
    print(f"hf: index has {len(cells)} cells", flush=True)
