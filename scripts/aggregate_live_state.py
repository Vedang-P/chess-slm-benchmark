"""Combine N parallel MATE workers' live-push state into the canonical
dashboard files.

Each Kaggle worker runs scripts/run_mate_eval.py --worker-id wN --live-push,
which publishes ONLY to monitor/workers/wN.state.json + wN.live.json (see
that script's --worker-id help for why: 6 workers writing straight to the
shared monitor/state.json would corrupt it, last write wins, discarding
the other 5 workers' progress). This script is the sole writer of the
canonical monitor/state.json + history.jsonl + live.json once workers are
involved: it polls every worker's small state file (w0 = the fixed,
already-complete first-100 baseline; see build_worker0_baseline.py), and
scripts/aggregate_live_state.py recombines them with combine_mate_metrics
-- sums of raw counts, never an average of averages -- into the same
shape run_mate_eval.py already publishes for a single-worker run, so the
existing frontend needs no changes.

Usage:
    python scripts/aggregate_live_state.py                  # poll forever, 45s
    python scripts/aggregate_live_state.py --interval 20
    python scripts/aggregate_live_state.py --once            # one tick, exit
    python scripts/aggregate_live_state.py --workers w0,w1   # smoke-test subset
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
from src.live_push import fetch_file, resolve_token, upload_file  # noqa: E402
from src.mate_metrics import combine_mate_metrics  # noqa: E402

DEFAULT_WORKERS = ["w0", "w1", "w2", "w3", "w4", "w5"]  # w0 = completed baseline; w1-w5 = 5 concurrent Kaggle workers (Kaggle's own 5-concurrent-CPU-session cap, not a design choice)
COMBINED_RUN_ID = "mate-1000-campaign"


def _utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _fetch_json(token: str, remote: str) -> dict | None:
    data = fetch_file(token, remote)
    return json.loads(data) if data is not None else None


def combine_once(token: str, worker_ids: list[str], verbose: bool = True) -> dict:
    parts: dict[str, dict] = {}
    for wid in worker_ids:
        part = _fetch_json(token, f"monitor/workers/{wid}.state.json")
        if part is not None:
            parts[wid] = part
    if verbose:
        seen = ", ".join(f"{wid}={p['progress']['done']}/{p['progress']['total']}"
                         for wid, p in parts.items()) or "(none published yet)"
        print(f"[{_utc_ts()}] {len(parts)}/{len(worker_ids)} workers reporting: {seen}",
              flush=True)
    if not parts:
        return {}

    total = sum(p["progress"]["total"] for p in parts.values())
    done = sum(p["progress"]["done"] for p in parts.values())
    failed = sum(p["progress"]["failed"] for p in parts.values())
    mate = combine_mate_metrics([p["mate"] for p in parts.values()])
    remaining = max(0, total - done)
    rate = mate.get("positions_per_hour")

    started_candidates = [p.get("started_at") for p in parts.values() if p.get("started_at")]
    config = next((p["config"] for wid, p in parts.items()
                   if wid != "w0" and p.get("config")), None) or next(iter(parts.values()))["config"]
    models = next(iter(parts.values())).get("models", ["deepseek-v4-flash"])

    # most-recently-updated worker with a live error wins the headline slot,
    # prefixed so it's traceable back to a specific worker/kernel.
    errored = [(wid, p) for wid, p in parts.items() if p.get("last_error")]
    last_error = None
    if errored:
        wid, p = max(errored, key=lambda kv: kv[1].get("updated_at") or "")
        last_error = f"[{wid}] {p['last_error']}"

    stage = "complete" if (len(parts) == len(worker_ids)
                           and all(p.get("stage") == "complete" for p in parts.values())) else "sweep"

    state = {
        "repo": "Vedang-P/chess-slm-benchmark",
        "run_kind": "mate-selection",
        "mode": "mate",
        "run_id": COMBINED_RUN_ID,
        "stage": stage,
        "started_at": min(started_candidates) if started_candidates else _utc_ts(),
        "updated_at": _utc_ts(),
        "task": "mate-selection-test",
        "progress": {"done": done, "total": total, "failed": failed,
                     "fraction": round(done / total, 4) if total else 0.0,
                     "cells_done": done, "cells_total": total,
                     "cells_failed": failed, "cells_attempted": done},
        "eta_min": int(remaining / rate * 60) if rate else None,
        "models": models,
        "config": config,
        "mate": mate,
        "current": None,
        "last_error": last_error,
        "cells": [],
        # debug/traceability only -- not read by the frontend
        "workers": {wid: {"done": p["progress"]["done"], "total": p["progress"]["total"],
                          "stage": p.get("stage"), "updated_at": p.get("updated_at")}
                    for wid, p in parts.items()},
    }

    # live sample: whichever live worker most recently produced one
    live_candidates = []
    for wid in parts:
        if wid == "w0":
            continue
        live = _fetch_json(token, f"monitor/workers/{wid}.live.json")
        if live is not None:
            live_candidates.append(live)
    live = max(live_candidates, key=lambda live_json: live_json.get("updated_at") or "") \
        if live_candidates else None

    return {"state": state, "live": live}


def publish_combined(token: str, combined: dict) -> None:
    state = combined["state"]
    monitor_dir = ROOT / "monitor"
    monitor_dir.mkdir(exist_ok=True)
    state_path = monitor_dir / "state.json"
    state_path.write_text(json.dumps(state, indent=1))

    hist = monitor_dir / "history.jsonl"
    lines = hist.read_text().splitlines() if hist.exists() else []
    m = state["mate"]
    lines.append(json.dumps({
        "run_id": state["run_id"], "ts": state["updated_at"],
        "run_kind": "mate-selection",
        "done": state["progress"]["done"], "fraction": state["progress"]["fraction"],
        "eta_min": state["eta_min"],
        "accuracy": m["accuracy"],
        "answer_rate": m["answer_rate"],
        "picked_b_rate": (round(m["picked_b"] / m["n"], 4) if m["n"] else None),
        "cells_done": state["progress"]["done"],
        "legal_avg": m["accuracy"],
        "last_error": state["last_error"],
    }))
    lines = lines[-500:]
    hist.write_text("\n".join(lines) + "\n")

    err = upload_file(token, "monitor/state.json", state_path.read_bytes(),
                      message=f"combined state {state['updated_at']}")
    if err:
        print(f"push state.json failed: {err}", flush=True)
    err = upload_file(token, "monitor/history.jsonl", hist.read_bytes(),
                      message=f"combined history {state['updated_at']}")
    if err:
        print(f"push history.jsonl failed: {err}", flush=True)

    if combined.get("live") is not None:
        live_path = monitor_dir / "live.json"
        live_path.write_text(json.dumps(combined["live"], indent=1))
        err = upload_file(token, "monitor/live.json", live_path.read_bytes(),
                          message=f"combined live {state['updated_at']}")
        if err:
            print(f"push live.json failed: {err}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", default=",".join(DEFAULT_WORKERS))
    ap.add_argument("--interval", type=int, default=45)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    token = resolve_token()
    if not token:
        raise SystemExit("no GITHUB_TOKEN -- can't read/publish the monitor repo")
    worker_ids = [w.strip() for w in args.workers.split(",") if w.strip()]

    while True:
        try:
            combined = combine_once(token, worker_ids)
            if combined:
                publish_combined(token, combined)
                m = combined["state"]["mate"]
                p = combined["state"]["progress"]
                print(f"  -> published: {p['done']}/{p['total']} done, "
                      f"accuracy={m['accuracy']}, eta_min={combined['state']['eta_min']}",
                      flush=True)
        except Exception as e:
            print(f"aggregate tick failed: {type(e).__name__}: {e}", flush=True)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
