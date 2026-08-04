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
# Prompt-level forcing (user decision 2026-08-04): demand an answer inside
# the budget; if the model cannot be confident, it must still output its best
# guess from the two candidates. The answer text is still the model's own
# output — this is NOT a runner-side fallback.
ANSWER_SPEC_FORCED = (
    "Answer with exactly one of: MoveA:<move> or MoveB:<move>. "
    "Only output the line, nothing else.\n"
    "You MUST output an answer. If you cannot determine which move is better "
    "with confidence, output your best guess from the two candidates anyway. "
    "An answer is required; refusing to answer is not acceptable."
)
# \b after the label stops "move a2a3" (prose naming a uci move) from being
# read as a vote for MoveA; the label must stand alone.
ANSWER_RE = re.compile(
    r"\bMove\s*([AB])\b\s*[:.\-]?\s*([a-h][1-8][a-h][1-8][qrbnQRBN]?)?", re.I)
UCI_RE = re.compile(r"[a-h][1-8][a-h][1-8][qrbnQRBN]?")


def _utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_choice(text: str, candidate_a: str, candidate_b: str):
    """Return (label, move) parsed from a model answer, or (None, None).

    The LAST MoveA/MoveB mention wins, matching the from-to rule in
    src/benchmarks/games/tasks.py. Taking the FIRST match scored whichever
    candidate the model happened to name first, so "MoveA loses material, so
    MoveB" was scored as A. This is invisible for a model whose reasoning
    arrives in a separate reasoning_content channel (deepseek answers with a
    bare "MoveB:d5d4"), and systematically wrong for any model that reasons
    inside `content` — which is every local gemma run.
    """
    if not text:
        return None, None
    matches = list(ANSWER_RE.finditer(text))
    if matches:
        m = matches[-1]
        label = m.group(1).upper()
        move = m.group(2) or (candidate_a if label == "A" else candidate_b)
        return label, move
    # bare uci: the model named a candidate move without the MoveX label
    for token in reversed(UCI_RE.findall(text)):
        if token == candidate_a:
            return "A", candidate_a
        if token == candidate_b:
            return "B", candidate_b
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--ids", type=str, default=None,
                    help="comma-separated position ids to run (probe arms); "
                         "overrides --n")
    ap.add_argument("--output_dir", default="results/mate-selection")
    ap.add_argument("--max_new_tokens", type=int, default=2048,
                    help="total generation budget, identical across models "
                         "(gemma runs the same 2048) for a fair comparison. "
                         "Thinking budget is separate (--thinking-budget); "
                         "the forced-answer fallback covers truncation.")
    ap.add_argument("--thinking-budget", type=int, default=None,
                    help="bound reasoning tokens while keeping thinking ON")
    ap.add_argument("--thinking-disabled", action="store_true",
                    help="disable thinking entirely (native direct answers)")
    ap.add_argument("--local-thinking", action="store_true",
                    help="local gemma (E2B/E4B): render with the <|channel>"
                         "thought channel ENABLED and extract thinking + answer "
                         "separately (thinking tokens are recorded like the "
                         "gateway arm). DeepSeek ignores this flag; the gateway "
                         "thinks whenever thinking is not disabled.")
    ap.add_argument("--force-answer-prompt", action="store_true",
                    help="append the must-answer/best-guess instruction to the "
                         "prompt (prompt-level forcing; answer still the model's own)")
    ap.add_argument("--resume", action="store_true",
                    help="skip position_ids already scored in the samples file")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--stream", action="store_true",
                    help="stream tokens through the SSE path")
    ap.add_argument("--live-push", action="store_true")
    ap.add_argument("--live-namespace", type=str, default=None,
                    help="publish live.json/state.json/history.jsonl under "
                         "monitor/<namespace>/ instead of the canonical "
                         "monitor/ paths, locally and on the dashboard repo. "
                         "Lets independent runs (e.g. a gemma arm) stream to "
                         "their OWN dashboard page without overwriting the "
                         "deepseek page's state.")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    configure_quiet_logging()
    # --live-namespace scopes every monitor path (local + remote) so an
    # independent run owns its dashboard page; the canonical paths stay the
    # default so existing runs are byte-for-byte unchanged.
    live_ns = f"{args.live_namespace.strip('/')}/" if args.live_namespace else ""
    live_dir = ROOT / "monitor" / (args.live_namespace.strip("/") if args.live_namespace else "")
    # what the run ACTUALLY did. The gateway arm is a thinking arm only when
    # --thinking-disabled is absent; local gemma renders with the thought
    # channel enabled only when --local-thinking is passed (thinking is a
    # deliberate choice per arm, never an accidental default). Recording
    # `model == deepseek` alone labelled every direct-mode run as
    # thinking-enabled.
    thinking_enabled = (not args.thinking_disabled
                        and (args.model == "deepseek-v4-flash"
                             or args.local_thinking))
    all_records = json.loads(RECORDS_PATH.read_text())
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",") if i.strip()}
        records = [r for r in all_records if r["id"] in wanted]
        missing = wanted - {r["id"] for r in records}
        if missing:
            raise SystemExit(f"unknown position ids: {sorted(missing)}")
    else:
        records = all_records[: args.n]
    ALL_TOTAL = len(records)  # the run's own n (progress total)
    run_id = os.environ.get("BENCH_RUN_ID") or _utc_ts()
    run_name = f"{args.model}_mate-selection-test_strategy"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # exclusive lock: two processes writing the same output dir would append
    # duplicate samples (this happened once — a stale worker survived a kill)
    lock_path = out_dir / ".run.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(lock_fd, str(os.getpid()).encode())
    except FileExistsError:
        raise SystemExit(f"another runner already holds {lock_path} — refusing to run twice")
    import atexit

    atexit.register(lambda: os.unlink(lock_path))
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
         "thinking_enabled": thinking_enabled,
         "thinking_budget": args.thinking_budget,
         "force_answer_prompt": args.force_answer_prompt,
         "max_new_tokens": args.max_new_tokens},
        resume=args.resume, prior_samples=len(existing_samples),
    )
    # NOTE: existing samples stay in the file as-is — they must NOT be
    # re-added (ResultWriter.add appends, which duplicated everything on
    # resume). They only count toward the final metrics, and
    # prior_samples makes summary.n_samples report the true total (it used to
    # report only this process's rows: a completed 1000-position run whose
    # last leg scored 192 wrote n_samples=192, which then told
    # run_suite --resume the cell was unfinished).

    model = make_model(args.model, smoke_test=args.smoke)
    model.load()

    # live.json local writer + optional background pusher
    import threading

    from src.live_push import resolve_token, upload_file

    live_token = resolve_token() if args.live_push else None
    _live_pending = [None, None]  # [live.json bytes, state+history bundle]
    _live_cond = threading.Condition()

    def _mate_metrics() -> dict:
        """Everything the dashboard needs about a MATE selection run, under
        its OWN names.

        This used to be published as a fake sweep "cell" with tactical field
        names: accuracy was written into `legal_rate`, so the dashboard's
        headline card read "LEGAL MOVE RATE 79.0%" for a task that has no
        notion of legality, and the position count was labelled "cells". The
        monitor now publishes MATE metrics as MATE metrics.

        Covers every scored sample including resumed ones — dividing this
        process's correct count by the resumed total understated live
        accuracy by the resume fraction.
        """
        rows = existing_samples + samples
        scored = [s for s in rows if s.get("status") != "api_error"]
        answered = [s for s in scored if s.get("status") in ("correct", "wrong")]
        n = len(scored)
        truth = lambda s: ((s.get("position_metadata") or {}).get("task_extra") or {}).get("truth_label")
        by_truth = {}
        for label in ("A", "B"):
            group = [s for s in scored if truth(s) == label]
            by_truth[label] = {
                "n": len(group),
                "accuracy": round(sum(bool(s["compliance"]) for s in group) / len(group), 4)
                if group else None,
            }
        reasons = {}
        for s in scored:
            reason = s.get("no_answer_reason")
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1

        def _mean(values):
            values = [v for v in values if isinstance(v, (int, float))]
            return round(sum(values) / len(values), 2) if values else None

        elapsed_h = max(1e-9, (time.time() - run_started_epoch) / 3600)
        done_here = len(samples)
        return {
            "n": n,
            "n_attempted": len(rows),
            "answered": len(answered),
            "correct": sum(bool(s["compliance"]) for s in scored),
            "wrong": sum(s.get("status") == "wrong" for s in scored),
            "no_answer": sum(s.get("status") == "no_answer" for s in scored),
            "parse_error": sum(s.get("status") == "parse_error" for s in scored),
            "api_error": sum(s.get("status") == "api_error" for s in rows),
            "accuracy": round(sum(bool(s["compliance"]) for s in scored) / n, 4) if n else None,
            "accuracy_of_answered": (
                round(sum(bool(s["compliance"]) for s in answered) / len(answered), 4)
                if answered else None),
            "answer_rate": round(len(answered) / n, 4) if n else None,
            # the run's actual signal: does the model just always say B?
            "picked_a": sum(s.get("label") == "A" for s in scored),
            "picked_b": sum(s.get("label") == "B" for s in scored),
            "truth_a": by_truth["A"]["n"],
            "truth_b": by_truth["B"]["n"],
            "accuracy_truth_a": by_truth["A"]["accuracy"],
            "accuracy_truth_b": by_truth["B"]["accuracy"],
            "no_answer_reasons": reasons,
            "mean_latency_s": _mean([(s.get("latency_ms") or 0) / 1000 for s in scored
                                     if s.get("latency_ms") is not None]),
            "mean_output_tokens": _mean([(s.get("token_usage") or {}).get("output_tokens")
                                         for s in scored]),
            "mean_reasoning_tokens": _mean([(s.get("token_usage") or {}).get("reasoning_tokens")
                                            for s in scored]),
            "positions_per_hour": round(done_here / elapsed_h, 1) if done_here else None,
        }

    def _state_payload(done: int, total: int, stage: str, last_error: str = None) -> dict:
        mate = _mate_metrics()
        remaining = max(0, total - done)
        rate = mate.get("positions_per_hour")
        return {
            "repo": "Vedang-P/chess-slm-benchmark",
            # run_kind tells the dashboard which scoreboard to draw. Without
            # it the MATE run was rendered through the sweep layout, which is
            # why four of seven cards were permanently blank.
            "run_kind": "mate-selection",
            "mode": "mate",
            "run_id": run_id,
            "stage": stage,
            "started_at": started_at,
            "updated_at": _utc_ts(),
            "task": "mate-selection-test",
            "progress": {"done": done, "total": total, "failed": mate["api_error"],
                         "fraction": round(done / total, 4) if total else 0.0,
                         # legacy aliases so an older cached dashboard build
                         # does not show blanks during a deploy
                         "cells_done": done, "cells_total": total,
                         "cells_failed": mate["api_error"],
                         "cells_attempted": done},
            "eta_min": int(remaining / rate * 60) if rate else None,
            "models": [args.model],
            "config": {
                "thinking_enabled": thinking_enabled,
                "thinking_budget": args.thinking_budget,
                "max_new_tokens": args.max_new_tokens,
                "force_answer_prompt": args.force_answer_prompt,
            },
            "mate": mate,
            "current": {"model": args.model, "task": "mate-selection-test",
                        "variant": "strategy"} if stage == "sweep" else None,
            "last_error": last_error,
            "cells": [],
        }

    def _publish_state(done: int, total: int, stage: str, last_error: str = None) -> None:
        """Write + enqueue monitor[/ns]/state.json and history.jsonl so the
        dashboard's scoreboard/charts reflect the MATE run."""
        live_dir.mkdir(parents=True, exist_ok=True)
        state = _state_payload(done, total, stage, last_error)
        (live_dir / "state.json").write_text(json.dumps(state, indent=1))
        hist = live_dir / "history.jsonl"
        lines = hist.read_text().splitlines() if hist.exists() else []
        m = state["mate"]
        lines.append(json.dumps({
            "run_id": run_id, "ts": state["updated_at"],
            "run_kind": "mate-selection",
            "done": done, "fraction": state["progress"]["fraction"],
            "eta_min": state["eta_min"],
            "accuracy": m["accuracy"],
            "answer_rate": m["answer_rate"],
            "picked_b_rate": (round(m["picked_b"] / m["n"], 4) if m["n"] else None),
            # legacy keys the old chart code reads
            "cells_done": done,
            "legal_avg": m["accuracy"],
            "last_error": last_error,
        }))
        lines = lines[-500:]
        hist.write_text("\n".join(lines) + "\n")
        with _live_cond:
            # slot 1 carries STATE bytes only; history is re-read from disk
            # at upload time (uploading history bytes to state.json corrupts it)
            _live_pending[1] = (live_dir / "state.json").read_bytes()
            _live_cond.notify()

    def _live_pusher() -> None:
        while True:
            with _live_cond:
                while _live_pending[0] is None and _live_pending[1] is None:
                    _live_cond.wait()
                live_data = _live_pending[0]
                state_data = _live_pending[1]
                _live_pending[0] = None
                _live_pending[1] = None
            if live_data is not None:
                try:
                    upload_file(live_token, f"monitor/{live_ns}live.json",
                                live_data, message=f"live {_utc_ts()}")
                except Exception:
                    pass
            if state_data is not None:
                try:
                    upload_file(live_token, f"monitor/{live_ns}state.json",
                                state_data, message=f"state {_utc_ts()}")
                    hist_path = live_dir / "history.jsonl"
                    upload_file(live_token, f"monitor/{live_ns}history.jsonl",
                                hist_path.read_bytes(),
                                message=f"history {_utc_ts()}")
                except Exception:
                    pass
            time.sleep(1.0)

    if live_token:
        threading.Thread(target=_live_pusher, daemon=True).start()

    started_at = _utc_ts()
    run_started_epoch = time.time()  # throughput/ETA are measured from here

    def write_live(rec, out, scored, phase, sample_idx):
        import chess

        extra = rec["task_extra"]
        board = chess.Board(rec["fen"])
        piece_list = [
            {"sq": chess.square_name(sq),
             "color": "w" if board.piece_at(sq).color == chess.WHITE else "b",
             "kind": board.piece_at(sq).symbol().upper()}
            for sq in chess.SQUARES if board.piece_at(sq) is not None
        ]
        live = {
            "updated_at": _utc_ts(),
            "cell": {"model": args.model, "task": "mate-selection-test",
                     "variant": "strategy"},
            "task_category": "Short Tactics",
            "task_kind": "mate_selection",  # the dashboard branches on this
            "run_kind": "mate-selection",
            "run_id": run_id,
            "sample_idx": sample_idx,
            "sample_total": ALL_TOTAL,
            "position_id": rec["id"],
            "prompt": extra["instruction"] + "\n" + extra["input"],
            "model_input": extra["instruction"] + "\n" + extra["input"]
                          + "\n" + (ANSWER_SPEC_FORCED if args.force_answer_prompt
                                     else ANSWER_SPEC),
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
            "pieces": piece_list,
            "n": 8,
            "record_id": rec["id"],
            "position": {
                "id": rec["id"],
                "n": 8,
                "turn": "w" if board.turn == chess.WHITE else "b",
                "fen": rec["fen"],
                "pieces": piece_list,
                "source": "MATE testset",
            },
        }
        live_dir.mkdir(parents=True, exist_ok=True)
        (live_dir / "live.json").write_text(json.dumps(live, indent=1))
        if live_token:
            with _live_cond:
                _live_pending[0] = (live_dir / "live.json").read_bytes()
                _live_cond.notify()

    samples = []
    t0 = time.time()
    for i, rec in enumerate(records):
        extra = rec["task_extra"]
        prompt = extra["instruction"] + "\n" + extra["input"]
        answer_spec = ANSWER_SPEC_FORCED if args.force_answer_prompt else ANSWER_SPEC
        model_input = prompt + "\n" + answer_spec

        def _live_partial(partial):
            write_live(rec, {"content": partial.get("content", ""),
                             "reasoning": partial.get("reasoning", ""),
                             "finished": partial.get("finished", False)},
                       {}, "generating", len(samples) + 1)

        out = model.generate(model_input, max_new_tokens=args.max_new_tokens,
                             stream=True,  # SSE: live reasoning on the website
                             on_chunk=_live_partial if not args.smoke else None,
                             thinking_budget=args.thinking_budget,
                             thinking_disabled=args.thinking_disabled,
                             local_thinking=args.local_thinking)
        # NO retries and NO fallbacks: one call, whatever comes back is scored.
        if args.smoke:
            out = {"content": "MoveB:d5d4", "reasoning": "",
                   "token_usage": {"input_tokens": 10, "output_tokens": 8,
                                   "total_tokens": 18, "usage_complete": True},
                   "latency_ms": 1, "finished": True}
        api_error = out.get("error")
        label, move = parse_choice(out.get("content", ""),
                                   extra["candidate_a"], extra["candidate_b"])
        # careful no-answer classification: WHY there is no answer matters
        # (gave up vs truncated by the budget vs unparseable)
        content = out.get("content", "") or ""
        reasoning = out.get("reasoning", "") or ""
        no_answer_reason = None
        if label is None:
            if not content.strip() and not out.get("finished", True):
                no_answer_reason = "truncated"      # budget cut off mid-generation
            elif not content.strip():
                no_answer_reason = "gave_up"        # stopped without content
            else:
                no_answer_reason = "unparseable"    # content but no candidate
        # NO fallback: if the model's text has no parseable choice, the sample
        # is a parse_error/no_answer. An answer recorded on a sample is always
        # the model's own output.
        correct = label == extra["truth_label"]
        if api_error:
            # transport failure (gateway 5xx / timeout / rate limit). Not a
            # model answer and not a model failure: excluded from accuracy.
            status, correct, no_answer_reason = "api_error", None, "api_error"
            print(f"  !! api_error on {rec['id']}: {api_error[:160]}", flush=True)
        elif label is None:
            status = "no_answer" if not content.strip() else "parse_error"
        elif correct:
            status = "correct"
        else:
            status = "wrong"
        scored = {"status": status, "label": label, "move": move,
                  "compliance": correct, "no_answer_reason": no_answer_reason}
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
            "attempts": out.get("attempts", 1),
            "max_new_tokens": args.max_new_tokens,
            "thinking_enabled": thinking_enabled,
            "force_answer_prompt": args.force_answer_prompt,
            "prompt": prompt,
            "model_input": model_input,
            "output": out.get("content", ""),
            "reasoning": out.get("reasoning", ""),
            "reasoning_chars": len(out.get("reasoning") or ""),
            "answer_chars": len(out.get("content") or ""),
            "token_usage": out.get("token_usage"),
            "cache_hit": (out.get("token_usage") or {}).get("cache_hit_tokens"),
            "cache_miss": (out.get("token_usage") or {}).get("cache_miss_tokens"),
            "fallback": None,  # never fabricated: answers are the model's own text
            "api_error": api_error,
            "no_answer_reason": no_answer_reason,
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
            try:
                _publish_state(len(existing_samples) + len(samples), ALL_TOTAL, "sweep")
            except Exception as e:
                print(f"state publish failed: {type(e).__name__}: {e}", flush=True)

    all_rows = existing_samples + samples
    # api_error rows are transport failures, not measurements: out of every
    # denominator. `parsed` = the model produced a usable A/B choice; a
    # no_answer is NOT parsed (it used to count as parsed, inflating
    # parse_rate to 1.0 whenever the model returned nothing at all).
    all_scored = [s for s in all_rows if s["status"] != "api_error"]
    parsed = [s for s in all_scored if s["status"] in ("correct", "wrong")]
    n = len(all_scored)
    accuracy = {
        "n": n,
        "n_attempted": len(all_rows),
        "api_error": sum(s["status"] == "api_error" for s in all_rows),
        "parse_rate": round(len(parsed) / n, 4) if n else 0.0,
        "accuracy_strict": round(sum(bool(s["compliance"]) for s in all_scored) / n, 4) if n else 0.0,
        "accuracy_of_parsed": round(sum(bool(s["compliance"]) for s in parsed) / len(parsed), 4) if parsed else None,
        "correct": sum(bool(s["compliance"]) for s in all_scored),
        "wrong": sum(s["status"] == "wrong" for s in all_scored),
        "no_answer": sum(s["status"] == "no_answer" for s in all_scored),
        "parse_error": sum(s["status"] == "parse_error" for s in all_scored),
    }
    metrics = {"accuracy": accuracy, "token_usage": aggregate_token_usage(all_scored)}
    if accuracy["api_error"]:
        print(f"NOTE: {accuracy['api_error']} api_error samples excluded from "
              f"all rates (transport failures, not model answers)", flush=True)
    summary = writer.finish(metrics)
    try:
        _publish_state(len(all_rows), ALL_TOTAL, "complete")
    except Exception as e:
        print(f"state publish failed: {type(e).__name__}: {e}", flush=True)
    print(json.dumps(summary["metrics"], indent=1), flush=True)
    if args.smoke:
        # a stub run must never publish to the public HF dataset repo —
        # the archive is the paper's evidence trail, not a scratch space
        print("smoke run: skipping HF upload", flush=True)
    else:
        try:
            from src.hf_push import upload_cell

            upload_cell(Path(args.output_dir), run_id, writer.summary_path)
        except Exception as e:
            print(f"hf upload skipped: {e}", flush=True)


if __name__ == "__main__":
    main()
