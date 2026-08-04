"""One-off: publish the completed first 100 MATE positions as the fixed
"w0" baseline that scripts/aggregate_live_state.py combines with the 6
live parallel workers covering the remaining 900.

Pulls from HF (runs/2026-08-04T16:09:37Z/), NOT the local
results/mate-selection/*.samples.jsonl file -- that local file turned out
to be stale leftover data from 2026-08-03 (reasoning_chars=0, ~2s
latencies: a different, since-superseded methodology), not the verified
100/100 thinking-mode run this session actually produced and archived.

Usage:
    python scripts/build_worker0_baseline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
from src.live_push import resolve_token, upload_file  # noqa: E402
from src.mate_metrics import compute_mate_metrics  # noqa: E402

BASELINE_RUN_ID = "2026-08-04T16:09:37Z"
CELL = "deepseek-v4-flash_mate-selection-test_strategy"


def main() -> None:
    from huggingface_hub import hf_hub_download

    from src.hf_push import HF_REPO, resolve_hf_token

    hf_token = resolve_hf_token()
    samples_path = hf_hub_download(
        repo_id=HF_REPO, repo_type="dataset", token=hf_token,
        filename=f"runs/{BASELINE_RUN_ID}/{CELL}.samples.jsonl")
    summary_path = hf_hub_download(
        repo_id=HF_REPO, repo_type="dataset", token=hf_token,
        filename=f"runs/{BASELINE_RUN_ID}/{CELL}.summary.json")
    rows = [json.loads(line) for line in Path(samples_path).read_text().splitlines()
            if line.strip()]
    summary = json.loads(Path(summary_path).read_text())
    meta = summary.get("meta", {})

    assert len(rows) == 100, f"expected 100 baseline rows, got {len(rows)}"
    mate = compute_mate_metrics(rows, done_here=0, elapsed_h=None)
    assert mate["n"] == 100 and mate["api_error"] == 0, (
        f"baseline is not the clean 100/100 result: n={mate['n']} "
        f"api_error={mate['api_error']}")
    print(f"baseline: {mate['correct']}/{mate['n']} correct, "
          f"{mate['no_answer']} no_answer, {mate['api_error']} api_error")

    state = {
        "repo": "Vedang-P/chess-slm-benchmark",
        "run_kind": "mate-selection",
        "mode": "mate",
        "run_id": BASELINE_RUN_ID,
        "stage": "complete",
        "started_at": BASELINE_RUN_ID,
        "updated_at": BASELINE_RUN_ID,
        "task": "mate-selection-test",
        "progress": {"done": 100, "total": 100, "failed": 0, "fraction": 1.0,
                     "cells_done": 100, "cells_total": 100, "cells_failed": 0,
                     "cells_attempted": 100},
        "eta_min": None,
        "models": [meta.get("model", "deepseek-v4-flash")],
        "config": {
            "thinking_enabled": meta.get("thinking_enabled"),
            "thinking_budget": meta.get("thinking_budget"),
            "max_new_tokens": meta.get("max_new_tokens"),
            "force_answer_prompt": meta.get("force_answer_prompt"),
        },
        "mate": mate,
        "current": None,
        "last_error": None,
        "cells": [],
    }

    out_local = ROOT / "monitor" / "workers" / "w0.state.json"
    out_local.parent.mkdir(parents=True, exist_ok=True)
    out_local.write_text(json.dumps(state, indent=1))
    print(f"wrote {out_local}")

    token = resolve_token()
    if not token:
        raise SystemExit("no GITHUB_TOKEN -- can't push w0 baseline to the monitor repo")
    err = upload_file(token, "monitor/workers/w0.state.json", out_local.read_bytes(),
                      message="w0 baseline: completed first 100 positions")
    if err:
        raise SystemExit(f"push failed: {err}")
    print("pushed monitor/workers/w0.state.json")


if __name__ == "__main__":
    main()
