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
from src.mate_metrics import compute_mate_metrics  # noqa: E402
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


def final_metrics(rows: list) -> dict:
    """The paper's accuracy_strict/accuracy_of_parsed metrics block.

    Shared by the periodic HF checkpoint (partial `rows`, mid-run) and the
    true end-of-run summary (all `rows`) -- being the exact same function
    is what guarantees an interim upload's summary.json is a correct
    snapshot of progress so far, never a stale or incomplete stand-in that
    then gets uploaded to the public archive under the same path the
    final, real summary later overwrites.

    api_error rows are transport failures, not measurements: out of every
    denominator. `parsed` = the model produced a usable A/B choice; a
    no_answer is NOT parsed (it used to count as parsed, inflating
    parse_rate to 1.0 whenever the model returned nothing).
    """
    scored = [s for s in rows if s["status"] != "api_error"]
    parsed = [s for s in scored if s["status"] in ("correct", "wrong")]
    n = len(scored)
    accuracy = {
        "n": n,
        "n_attempted": len(rows),
        "api_error": sum(s["status"] == "api_error" for s in rows),
        "parse_rate": round(len(parsed) / n, 4) if n else 0.0,
        "accuracy_strict": round(sum(bool(s["compliance"]) for s in scored) / n, 4) if n else 0.0,
        "accuracy_of_parsed": round(sum(bool(s["compliance"]) for s in parsed) / len(parsed), 4) if parsed else None,
        "correct": sum(bool(s["compliance"]) for s in scored),
        "wrong": sum(s["status"] == "wrong" for s in scored),
        "no_answer": sum(s["status"] == "no_answer" for s in scored),
        "parse_error": sum(s["status"] == "parse_error" for s in scored),
    }
    return {"accuracy": accuracy, "token_usage": aggregate_token_usage(scored)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--offset", type=int, default=0,
                    help="skip this many records from the start of the "
                         "dataset before taking --n. Lets N parallel workers "
                         "each own a disjoint slice, e.g. worker 2 of 5 over "
                         "the remaining 900 positions: --offset 280 --n 180")
    ap.add_argument("--ids", type=str, default=None,
                    help="comma-separated position ids to run (probe arms); "
                         "overrides --n/--offset")
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
    ap.add_argument("--force-answer-prompt", action="store_true",
                    help="append the must-answer/best-guess instruction to the "
                         "prompt (prompt-level forcing; answer still the model's own)")
    ap.add_argument("--resume", action="store_true",
                    help="skip position_ids already scored in the samples file")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--stream", action="store_true",
                    help="stream tokens through the SSE path")
    ap.add_argument("--live-push", action="store_true")
    ap.add_argument("--worker-id", type=str, default=None,
                    help="when set, live-push publishes to "
                         "monitor/workers/{id}.state.json and "
                         "{id}.live.json instead of the shared canonical "
                         "paths, so N parallel workers don't overwrite each "
                         "other's dashboard state. scripts/aggregate_live_state.py "
                         "combines the worker files back into the canonical "
                         "monitor/state.json the frontend reads.")
    ap.add_argument("--hf-upload-every", type=int, default=25,
                    help="upload this worker's samples/summary to HF every "
                         "N scored positions, not just once at the end -- a "
                         "killed/timed-out session must not strand progress "
                         "since the last upload. 0 disables periodic upload "
                         "(still uploads once at the end unless --smoke).")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    configure_quiet_logging()
    # what the run ACTUALLY did. The gateway arm is a thinking arm only when
    # --thinking-disabled is absent; local gemma always renders with
    # enable_thinking=False. Recording `model == deepseek` instead labelled
    # every direct-mode run as thinking-enabled.
    thinking_enabled = (args.model == "deepseek-v4-flash"
                        and not args.thinking_disabled)
    all_records = json.loads(RECORDS_PATH.read_text())
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",") if i.strip()}
        records = [r for r in all_records if r["id"] in wanted]
        missing = wanted - {r["id"] for r in records}
        if missing:
            raise SystemExit(f"unknown position ids: {sorted(missing)}")
    else:
        records = all_records[args.offset: args.offset + args.n]
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
    _live_inflight = [False]  # True while the pusher is mid-upload (dequeued, not yet done)
    _live_cond = threading.Condition()

    # --worker-id namespaces every published path so N parallel workers each
    # get their own slot instead of overwriting one shared file (which would
    # corrupt the dashboard: whichever worker's write lands last "wins",
    # discarding the others' progress from the combined total). Without
    # --worker-id, behavior is unchanged: publish straight to the canonical
    # paths, exactly like every single-worker run so far this project.
    worker_tag = args.worker_id
    monitor_dir = ROOT / "monitor"
    state_local = monitor_dir / (f"workers/{worker_tag}.state.json" if worker_tag else "state.json")
    live_local = monitor_dir / (f"workers/{worker_tag}.live.json" if worker_tag else "live.json")
    state_remote = f"monitor/workers/{worker_tag}.state.json" if worker_tag else "monitor/state.json"
    live_remote = f"monitor/workers/{worker_tag}.live.json" if worker_tag else "monitor/live.json"

    def _mate_metrics() -> dict:
        """This process's own MATE scoreboard (see src/mate_metrics.py).

        This used to be published as a fake sweep "cell" with tactical field
        names: accuracy was written into `legal_rate`, so the dashboard's
        headline card read "LEGAL MOVE RATE 79.0%" for a task that has no
        notion of legality, and the position count was labelled "cells". The
        monitor now publishes MATE metrics as MATE metrics.

        Covers every scored sample including resumed ones — dividing this
        process's correct count by the resumed total understated live
        accuracy by the resume fraction. compute_mate_metrics is the same
        function scripts/aggregate_live_state.py uses to recombine several
        workers, so a single worker's live view and the combined dashboard
        can never silently disagree on the math.
        """
        elapsed_h = max(1e-9, (time.time() - run_started_epoch) / 3600)
        return compute_mate_metrics(existing_samples + samples,
                                    done_here=len(samples), elapsed_h=elapsed_h)

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
        """Write + enqueue this worker's state (+ canonical history.jsonl,
        single-worker runs only) so the dashboard reflects the MATE run.

        Under --worker-id, history.jsonl is deliberately NOT written here:
        scripts/aggregate_live_state.py is the sole writer of the canonical
        history once several workers are involved (see worker_tag comment
        above) — N workers each appending their own view of "history" would
        race on the same file and interleave incomparable partial totals.
        """
        state_local.parent.mkdir(parents=True, exist_ok=True)
        state = _state_payload(done, total, stage, last_error)
        state_local.write_text(json.dumps(state, indent=1))
        if not worker_tag:
            hist = monitor_dir / "history.jsonl"
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
            _live_pending[1] = state_local.read_bytes()
            _live_cond.notify()

    # GitHub core API is 5000 req/hr PER ACCOUNT for classic PATs, and the
    # contents API (2 calls per write: GET sha + PUT) has a secondary write
    # limit. The naive pusher uploaded live.json on EVERY streamed SSE chunk
    # (write_live is called from on_chunk), so a single 14k-token reasoning
    # stream could burn hundreds of API calls -- with 5 workers that is what
    # exhausted the account quota mid-campaign. live.json is transient by
    # design (it is overwritten every chunk), so intermediate versions can
    # simply be dropped; only the newest one matters. A "complete" state is
    # the one exception that must NEVER be throttled away: _wait_for_live_flush
    # only checks the queue, so a dropped final state would leave the
    # dashboard stuck on "sweep" forever.
    LIVE_UPLOAD_MIN_INTERVAL = 60.0  # seconds between live.json uploads
    STATE_UPLOAD_MIN_INTERVAL = 20.0  # seconds between state.json uploads

    def _is_complete_state(data: bytes | None) -> bool:
        if data is None:
            return False
        try:
            return json.loads(data).get("stage") == "complete"
        except Exception:
            return False

    def _live_pusher() -> None:
        last_live_ts = 0.0
        last_state_ts = 0.0
        while True:
            with _live_cond:
                while _live_pending[0] is None and _live_pending[1] is None:
                    _live_cond.wait()
                live_data = _live_pending[0]
                state_data = _live_pending[1]
                _live_pending[0] = None
                _live_pending[1] = None
                _live_inflight[0] = True
            now = time.time()
            try:
                if live_data is not None and now - last_live_ts >= LIVE_UPLOAD_MIN_INTERVAL:
                    try:
                        err = upload_file(live_token, live_remote, live_data,
                                          message=f"live {_utc_ts()}")
                        if err:
                            print(f"live-push live.json failed: {err}", flush=True)
                        else:
                            last_live_ts = time.time()
                    except Exception as e:
                        print(f"live-push live.json raised: {type(e).__name__}: {e}", flush=True)
                if state_data is not None and (
                        _is_complete_state(state_data)
                        or now - last_state_ts >= STATE_UPLOAD_MIN_INTERVAL):
                    try:
                        # upload_file() RETURNS a diagnostic string on failure,
                        # it does not raise -- silently ignoring that return
                        # value (as this used to) meant a worker's state.json
                        # could fail every single upload with zero trace
                        # anywhere: not in this process's own stdout, not on
                        # the dashboard, nothing. Measured directly 2026-08-04:
                        # 3 of 5 parallel workers never got a single
                        # monitor/workers/wN.state.json published, discovered
                        # only by noticing the aggregator never saw them.
                        err = upload_file(live_token, state_remote, state_data,
                                          message=f"state {_utc_ts()}")
                        if err:
                            print(f"live-push state.json failed: {err}", flush=True)
                        else:
                            last_state_ts = time.time()
                        if not worker_tag:
                            hist_path = monitor_dir / "history.jsonl"
                            err = upload_file(live_token, "monitor/history.jsonl",
                                              hist_path.read_bytes(),
                                              message=f"history {_utc_ts()}")
                            if err:
                                print(f"live-push history.jsonl failed: {err}", flush=True)
                    except Exception as e:
                        print(f"live-push state.json raised: {type(e).__name__}: {e}", flush=True)
            finally:
                with _live_cond:
                    _live_inflight[0] = False
                    _live_cond.notify_all()
            time.sleep(1.0)

    def _wait_for_live_flush(timeout_s: float = 30.0) -> None:
        """Block until the background pusher has actually delivered
        everything enqueued so far -- not just dequeued it.

        This is a daemon thread: it is killed outright the instant the
        process exits, mid-HTTP-request if that's where it happens to be.
        Without this, the single most important push of the whole run --
        the final "complete" state -- was a coin flip on Kaggle depending
        on which of (this process exiting) or (that PUT actually landing)
        won the race. Confirmed locally: a 3-position --smoke run finished
        and exited before the pusher thread ever got to make its first
        HTTP call, so the worker's dashboard state was never seen remotely
        despite every _publish_state() call succeeding locally.
        """
        if not live_token:
            return
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with _live_cond:
                idle = (_live_pending[0] is None and _live_pending[1] is None
                        and not _live_inflight[0])
            if idle:
                return
            time.sleep(0.1)
        print("warning: live-push queue did not flush before the timeout -- "
              "the last dashboard update for this worker may be stale",
              flush=True)

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
        live["worker_id"] = worker_tag
        live_local.parent.mkdir(parents=True, exist_ok=True)
        live_local.write_text(json.dumps(live, indent=1))
        if live_token:
            with _live_cond:
                _live_pending[0] = live_local.read_bytes()
                _live_cond.notify()

    # announce this worker immediately, before the first (possibly
    # minutes-long) generation returns -- otherwise the dashboard keeps
    # showing whatever the previous run left behind, and with several
    # workers starting at slightly different times, the aggregator can't
    # see this worker's true `total` (so the combined 1000 undercounts)
    # until its first position happens to finish.
    try:
        _publish_state(len(existing_samples), ALL_TOTAL, "sweep")
    except Exception as e:
        print(f"state publish failed: {type(e).__name__}: {e}", flush=True)

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
                             thinking_disabled=args.thinking_disabled)
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
        # publish every position (not gated behind the print cadence below):
        # each call is a cheap local write + an async enqueue onto the
        # background pusher thread, and a multi-worker run only updates the
        # dashboard between positions that can each take minutes — batching
        # this to "every 25" made the scoreboard look stalled for long
        # stretches even while work was actively happening.
        try:
            _publish_state(len(existing_samples) + len(samples), ALL_TOTAL, "sweep")
        except Exception as e:
            print(f"state publish failed: {type(e).__name__}: {e}", flush=True)
        if args.verbose or (i + 1) % 25 == 0 or i + 1 == len(records):
            el = time.time() - t0
            print(f"  [mate-selection {args.model}] {i + 1}/{len(records)} "
                  f"({el / (i + 1):.1f}s/pos) acc={sum(s['compliance'] for s in samples)}/{len(samples)}",
                  flush=True)
        # periodic HF upload: a killed/timed-out session must not strand
        # everything scored since the run's ONE upload at the very end.
        # upload_cell() re-uploads the whole current samples file each time
        # (see src/hf_push.py docstring: only_missing defaults False because
        # the file grows), so this is a safe, idempotent "sync progress so
        # far" — never a partial or corrupting write.
        if (not args.smoke and args.hf_upload_every
                and len(samples) % args.hf_upload_every == 0):
            try:
                from src.hf_push import upload_cell

                writer.finish(final_metrics(existing_samples + samples))
                upload_cell(Path(args.output_dir), run_id, writer.summary_path)
                print(f"hf: interim upload after {len(samples)} positions this worker",
                      flush=True)
            except Exception as e:
                print(f"hf interim upload skipped: {type(e).__name__}: {e}", flush=True)

    all_rows = existing_samples + samples
    metrics = final_metrics(all_rows)
    if metrics["accuracy"]["api_error"]:
        print(f"NOTE: {metrics['accuracy']['api_error']} api_error samples excluded "
              f"from all rates (transport failures, not model answers)", flush=True)
    summary = writer.finish(metrics)
    try:
        _publish_state(len(all_rows), ALL_TOTAL, "complete")
    except Exception as e:
        print(f"state publish failed: {type(e).__name__}: {e}", flush=True)
    _wait_for_live_flush()
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
