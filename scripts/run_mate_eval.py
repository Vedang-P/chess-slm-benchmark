"""Benchmark a model on the MATE move-selection test set.

Reads `data/positions/mate-selection-test.json` (1000 expert-annotated
strategy positions), prompts the model with the exact MATE instruction +
input, parses the MoveA/MoveB choice, and scores against MATE ground truth.

Per-sample records follow the study's token/accounting schema (see
`docs/paper-figures.md`), so results feed the paper figures and the HF
report pipeline. Live.json is written during generation; `--live-push`
also publishes it to the public repo.

Usage:
    python scripts/run_mate_eval.py --model deepseek-v4-flash --n 1000
    python scripts/run_mate_eval.py --model deepseek-v4-flash --n 5 --verbose
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
from src.models import configure_quiet_logging, make_model  # noqa: E402
from src.report import ResultWriter, aggregate_token_usage  # noqa: E402

RECORDS_PATH = ROOT / "data/positions/mate-selection-test.json"
ANSWER_SPEC = (
    "Answer with exactly one of: MoveA:<move> or MoveB:<move>. "
    "Only output the line, nothing else."
)
CHOICE_RE = re.compile(r"Move\s*([AB])")
ANSWER_RE = re.compile(r"Move\s*([AB])\s*[:.-]?\s*([a-h][1-8][a-h][1-8][qrbnQRBN]?)?")
UCI_RE = re.compile(r"[a-h][1-8][a-h][1-8][qrbnQRBN]?")


def _utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_choice(text: str, candidate_a: str, candidate_b: str):
    """Return (label, move) parsed from a model answer, or (None, None)."""
    if not text:
        return None, None
    m = ANSWER_RE.search(text)
    if m:
        label = m.group(1)
        move = m.group(2) or (candidate_a if label == "A" else candidate_b)
        return label, move
    m = CHOICE_RE.search(text)
    if m:
        return m.group(1), (candidate_a if m.group(1) == "A" else candidate_b)
    # bare uci fallback
    for token in UCI_RE.findall(text):
        if token == candidate_a:
            return "A", candidate_a
        if token == candidate_b:
            return "B", candidate_b
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--output_dir", default="results/mate-selection")
    ap.add_argument("--max_new_tokens", type=int, default=32768)
    ap.add_argument("--thinking-budget", type=int, default=None,
                    help="bound reasoning tokens while keeping thinking ON "
                         "(e.g. 2048 ~ 45s/position; unbounded ~3min/position)")
    ap.add_argument("--resume", action="store_true",
                    help="skip position_ids already scored in the samples file")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--stream", action="store_true",
                    help="stream tokens through the SSE path")
    ap.add_argument("--live-push", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    configure_quiet_logging()
    records = json.loads(RECORDS_PATH.read_text())[: args.n]
    run_id = os.environ.get("BENCH_RUN_ID") or _utc_ts()
    run_name = f"{args.model}_mate-selection-test_strategy"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / f"{run_name}.samples.jsonl"
    existing_samples = []
    if args.resume and samples_path.exists():
        for line in samples_path.read_text().splitlines():
            if line.strip():
                existing_samples.append(json.loads(line))
        done_ids = {s["position_id"] for s in existing_samples}
        records = [r for r in records if r["id"] not in done_ids]
        print(f"resume: {len(done_ids)} already scored, {len(records)} remaining",
              flush=True)
    writer = ResultWriter(
        out_dir, run_name,
        {"model": args.model, "task": "mate-selection-test",
         "prompt_variant": "strategy", "task_category": "Short Tactics",
         "run_id": run_id, "smoke": args.smoke,
         "thinking_enabled": args.model == "deepseek-v4-flash",
         "thinking_budget": args.thinking_budget,
         "max_new_tokens": args.max_new_tokens},
    )
    for s in existing_samples:
        writer.add(s)

    model = make_model(args.model, smoke_test=args.smoke)
    model.load()

    # live.json local writer + optional background pusher
    import threading

    from src.live_push import resolve_token, upload_file

    live_token = resolve_token() if args.live_push else None
    _live_pending = [None]
    _live_cond = threading.Condition()

    def _live_pusher() -> None:
        while True:
            with _live_cond:
                while _live_pending[0] is None:
                    _live_cond.wait()
                data = _live_pending[0]
                _live_pending[0] = None
            try:
                upload_file(live_token, "monitor/live.json", data,
                            message=f"live {_utc_ts()}")
            except Exception:
                pass
            time.sleep(1.0)

    if live_token:
        threading.Thread(target=_live_pusher, daemon=True).start()

    def write_live(rec, out, scored, phase, sample_idx):
        extra = rec["task_extra"]
        live = {
            "updated_at": _utc_ts(),
            "cell": {"model": args.model, "task": "mate-selection-test",
                     "variant": "strategy"},
            "task_category": "Short Tactics",
            "run_id": run_id,
            "sample_idx": sample_idx,
            "sample_total": len(records),
            "position_id": rec["id"],
            "prompt": extra["instruction"] + "\n" + extra["input"],
            "model_input": extra["instruction"] + "\n" + extra["input"]
                          + "\n" + ANSWER_SPEC,
            "output": out.get("content", ""),
            "reasoning": out.get("reasoning", ""),
            "finished": out.get("finished") if phase == "scored" else False,
            "token_usage": out.get("token_usage"),
            "phase": phase,
            "status": scored.get("status") if phase == "scored" else None,
            "move": scored.get("move"),
            "compliance": scored.get("compliance"),
            "correct": {"move": extra.get("output"), "note": "MATE expert choice"},
            "oracle": {"kind": "mate_selection",
                       "candidate_a": extra.get("candidate_a"),
                       "candidate_b": extra.get("candidate_b"),
                       "truth_label": extra.get("truth_label")},
            "fen": rec.get("fen"),
            "pieces": rec.get("pieces"),
            "n": 8,
            "record_id": rec["id"],
        }
        (ROOT / "monitor").mkdir(exist_ok=True)
        (ROOT / "monitor" / "live.json").write_text(json.dumps(live, indent=1))
        if live_token:
            with _live_cond:
                _live_pending[0] = (ROOT / "monitor" / "live.json").read_bytes()
                _live_cond.notify()

    samples = []
    t0 = time.time()
    for i, rec in enumerate(records):
        extra = rec["task_extra"]
        prompt = extra["instruction"] + "\n" + extra["input"]
        model_input = prompt + "\n" + ANSWER_SPEC

        def _live_partial(partial):
            write_live(rec, {"content": partial.get("content", ""),
                             "reasoning": partial.get("reasoning", ""),
                             "finished": partial.get("finished", False)},
                       {}, "generating", len(samples) + 1)

        out = model.generate(model_input, max_new_tokens=args.max_new_tokens,
                             stream=args.verbose or args.stream,
                             on_chunk=_live_partial if not args.smoke else None,
                             thinking_budget=args.thinking_budget)
        if args.smoke:
            out = {"content": "MoveB:d5d4", "reasoning": "",
                   "token_usage": {"input_tokens": 10, "output_tokens": 8,
                                   "total_tokens": 18, "usage_complete": True},
                   "latency_ms": 1, "finished": True}
        label, move = parse_choice(out.get("content", ""),
                                   extra["candidate_a"], extra["candidate_b"])
        correct = label == extra["truth_label"]
        if label is None:
            status = "parse_error"
        elif correct:
            status = "correct"
        else:
            status = "wrong"
        scored = {"status": status, "label": label, "move": move,
                  "compliance": correct}
        sample = {
            "position_id": rec["id"],
            "model": args.model,
            "task": "mate-selection-test",
            "task_category": "Short Tactics",
            "representation": "strategy",
            "run_id": run_id,
            "condition": "win",
            "value": rec["value"],
            "latency_ms": out.get("latency_ms"),
            "finished": out.get("finished"),
            "max_new_tokens": args.max_new_tokens,
            "thinking_enabled": args.model == "deepseek-v4-flash",
            "prompt": prompt,
            "model_input": model_input,
            "output": out.get("content", ""),
            "reasoning": out.get("reasoning", ""),
            "reasoning_chars": len(out.get("reasoning") or ""),
            "answer_chars": len(out.get("content") or ""),
            "token_usage": out.get("token_usage"),
            "cache_hit": (out.get("token_usage") or {}).get("cache_hit_tokens"),
            "cache_miss": (out.get("token_usage") or {}).get("cache_miss_tokens"),
            "fallback": None,
            "position_metadata": {"fen": rec.get("fen"),
                                  "task_extra": extra},
            "correct": {"move": extra.get("output"), "note": "MATE expert choice"},
            **scored,
        }
        samples.append(sample)
        writer.add(sample)
        write_live(rec, out, scored, "scored", len(samples))
        if args.verbose or (i + 1) % 25 == 0 or i + 1 == len(records):
            el = time.time() - t0
            print(f"  [mate-selection {args.model}] {i + 1}/{len(records)} "
                  f"({el / (i + 1):.1f}s/pos) acc={sum(s['compliance'] for s in samples)}/{len(samples)}",
                  flush=True)

    parsed = [s for s in samples if s["status"] != "parse_error"]
    accuracy = {
        "n": len(samples),
        "parse_rate": round(len(parsed) / len(samples), 4),
        "accuracy_strict": round(sum(s["compliance"] for s in samples) / len(samples), 4),
        "accuracy_of_parsed": round(sum(s["compliance"] for s in parsed) / len(parsed), 4) if parsed else None,
        "correct": sum(s["compliance"] for s in samples),
        "wrong": sum(not s["compliance"] for s in samples),
        "parse_error": sum(s["status"] == "parse_error" for s in samples),
    }
    metrics = {"accuracy": accuracy, "token_usage": aggregate_token_usage(samples)}
    summary = writer.finish(metrics)
    print(json.dumps(summary["metrics"], indent=1), flush=True)
    try:
        from src.hf_push import upload_cell

        upload_cell(Path(args.output_dir), run_id, writer.summary_path)
    except Exception as e:
        print(f"hf upload skipped: {e}", flush=True)


if __name__ == "__main__":
    main()
