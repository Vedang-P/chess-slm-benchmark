"""Evaluate one model on one chess task (both conditions, or the cap probe).

Runs on Kaggle (CUDA) or locally with --smoke (no model loading).

Usage:
    python scripts/run_chess.py --model smollm2-1.7b --task sm-5x5-win --n 20
    python scripts/run_chess.py --model gemma4-e2b --task mate1-lichess \
        --prompt-variant fen
    python scripts/run_chess.py --model smollm2-1.7b --task cap-legal-8x8 --smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent

from src.benchmarks.games import tasks as T  # noqa: E402
from src.benchmarks.games.envs import ENVS  # noqa: E402
from src.live_push import PUBLIC_LIVE_REPO, resolve_token, upload_file  # noqa: E402
from src.models import MODEL_IDS, configure_quiet_logging, make_model  # noqa: E402
from src.report import ResultWriter, aggregate_samples  # noqa: E402
from src.token_usage import merge_usage  # noqa: E402

TASK_FILES = {
    "cap-legal-8x8": "cap-legal-8x8.json",
    "bestmove-8x8": "bestmove-8x8.json",
    "mate1-lichess": "mate1-lichess.json",
    "mate2-lichess": "mate2-lichess.json",
    "sm-3x3-win": "sm-3x3-win.json",
    "sm-3x3-draw": "sm-3x3-draw.json",
    "sm-5x5-win": "sm-5x5-win.json",
    "sm-5x5-draw": "sm-5x5-draw.json",
    "mate1-8x8": "mate1-8x8.json",
    "mob-8x8": "mob-8x8.json",
}

GAME_TASKS = {"playout-5x5": "playout-5x5", "ttt": "ttt", "c4": "c4"}
GAME_MAX_MOVES = 120
GAME_N = 10

DEFAULT_MAX_NEW_TOKENS = 512


def _utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _cleanup(model) -> None:
    """Explicit release before the process exits. Primarily insurance: each
    cell runs in its own subprocess, so the OS/CUDA runtime already frees all
    GPU memory when the process ends — but clearing explicitly is cheap and
    protects against any lingering allocations inside the same process."""
    try:
        del model
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def play_one_game(model, env_name: str, max_new_tokens: int) -> dict:
    """Play one full game (model vs random opponent). Returns a sample dict."""
    import random

    env = ENVS[env_name]
    rng = random.Random()
    board = env.start()
    model_moves = 0
    illegal = 0
    t0 = time.time()
    for _ in range(GAME_MAX_MOVES):
        terminal, outcome = env.over(board)
        if terminal:
            break
        out = model.generate(env.prompt(board), max_new_tokens=max_new_tokens)
        mv = env.parse(out["content"])
        if mv is None or mv not in env.legal_moves(board):
            illegal += 1
            break
        board = env.apply(board, mv)
        model_moves += 1
        terminal, outcome = env.over(board)
        if terminal:
            break
        board = env.apply(board, env.random_move(board))
    terminal, outcome = env.over(board)
    return {
        "status": "played",
        "condition": "game",
        "position_id": f"game-{env_name}",
        "moves": model_moves,
        "illegal": illegal,
        "terminal": terminal,
        "outcome": outcome,
        "latency_ms": (time.time() - t0) * 1000,
    }


def game_metrics(samples: list) -> dict:
    n = len(samples)
    if not n:
        return {}
    legal = sum(s["moves"] for s in samples)
    illegal = sum(s["illegal"] for s in samples)
    outcomes = [s["outcome"] for s in samples]
    return {
        "games": {
            "n": n,
            "legal_rate": round(legal / (legal + illegal), 4) if legal + illegal else 0.0,
            "illegal_rate": round(illegal / (legal + illegal), 4) if legal + illegal else 0.0,
            "completion_rate": round(sum(1 for s in samples if s["terminal"]) / n, 4),
            "win_rate": round(outcomes.count("model") / n, 4),
            "draw_rate": round(outcomes.count(None) / n, 4),
            "loss_rate": round(outcomes.count("opp") / n, 4),
            "avg_moves": round(sum(s["moves"] for s in samples) / n, 2),
        }
    }


def main() -> None:
    configure_quiet_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smollm2-1.7b")
    ap.add_argument("--task", required=True,
                    choices=sorted(TASK_FILES) + sorted(GAME_TASKS))
    ap.add_argument("--prompt-variant", default="grid",
                    choices=["grid", "fen", "bitboard", "list"])
    ap.add_argument("--n", type=int, default=0, help="limit positions (0 = all)")
    ap.add_argument("--conditions", nargs="+", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--verbose", action="store_true",
                    help="print the exact rendered prompt and stream the model's "
                         "output live (debugging)")
    ap.add_argument("--stream", action="store_true",
                    help="stream tokens to stdout as they are generated")
    ap.add_argument("--cot", action="store_true",
                    help="add 'think step by step' to the prompt so non-reasoning "
                         "models also emit visible reasoning (check/debug only — "
                         "the benchmark itself measures direct answers)")
    ap.add_argument("--smoke", action="store_true", help="stub model, no GPU")
    ap.add_argument("--live-push", action="store_true",
                    help="push live.json to the public repo on every snapshot "
                         "(minimal website lag; ~1 file / 2-5s)")
    ap.add_argument("--data_dir", default="data/positions")
    ap.add_argument("--output_dir", default="results/chess")
    args = ap.parse_args()

    task_name = args.task
    task_category = T.task_category(task_name)
    run_id = os.environ.get("BENCH_RUN_ID") or _utc_ts()
    if task_name in GAME_TASKS:
        n_games = args.n or GAME_N
        model = make_model(args.model, smoke_test=args.smoke)
        model.load()
        run_name = f"{args.model}_{task_name}_{args.prompt_variant}"
        writer = ResultWriter(
            Path(args.output_dir), run_name,
            {"model": args.model, "task": task_name, "prompt_variant": args.prompt_variant,
             "task_category": task_category, "run_id": run_id,
             "smoke": args.smoke, "kind": "game"},
        )
        samples = []
        t0 = time.time()
        for i in range(n_games):
            sample = play_one_game(model, task_name, args.max_new_tokens)
            samples.append(sample)
            writer.add(sample)
            print(f"  [{task_name} {args.model}] game {i + 1}/{n_games} "
                  f"({time.time() - t0:.0f}s total)", flush=True)
        summary = writer.finish({"games": game_metrics(samples)["games"]})
        print(json.dumps(summary["metrics"], indent=1), flush=True)
        _cleanup(model)
        return

    records = json.loads((Path(args.data_dir) / TASK_FILES[task_name]).read_text())
    if args.n:
        records = records[: args.n]
    kind = task_name.split("-")[0]  # sm / mate1 / mate2 / mob / cap / bestmove
    conditions = args.conditions or (T.CAP_CONDITIONS if kind == "cap" else T.CONDITIONS)
    if kind == "cap" and args.prompt_variant == "fen" and args.conditions is None:
        pass  # cap runs the single 'win' condition label; variant still applies

    model = make_model(args.model, smoke_test=args.smoke)
    model.load()
    run_name = f"{args.model}_{task_name}_{args.prompt_variant}"
    writer = ResultWriter(
        Path(args.output_dir),
        run_name,
         {"model": args.model, "task": task_name, "prompt_variant": args.prompt_variant,
          "task_category": task_category, "run_id": run_id,
          "smoke": args.smoke, "thinking_enabled": args.model == "deepseek-v4-flash",
          "max_new_tokens": args.max_new_tokens},
    )
    live_last = [0.0]
    from src.live_push import resolve_token as _resolve_token

    live_token = _resolve_token() if args.live_push else None

    # background live.json pusher: the SSE loop must never wait on GitHub
    import threading

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
            time.sleep(1.0)  # pace GitHub API calls

    if live_token:
        threading.Thread(target=_live_pusher, daemon=True).start()

    def push_live(rec, condition, prompt, model_input, out=None, scored=None,
                  sample_idx=0, total=0, phase="scored", force=False,
                  fallback=None):
        """Write the exact sample context locally (before/after inference).
        No network: the suite-level monitor uploads live.json on its tick."""
        if kind == "game":
            return
        now = time.time()
        if not force and now - live_last[0] < 2.0:
            return
        live_last[0] = now
        out = out or {}
        scored = scored or {}
        correct = T.get_correct(rec, kind)
        extra = rec.get("task_extra") or {}
        live = {
            "updated_at": _utc_ts(),
            "cell": {"model": args.model, "task": task_name, "variant": args.prompt_variant},
            "task_category": task_category,
            "run_id": run_id,
            "sample_idx": sample_idx,
            "sample_total": total,
            "position_id": rec["id"],
            "prompt": prompt,
            "model_input": model_input,
            "output": out.get("content", ""),
            "reasoning": out.get("reasoning", ""),
            "finished": out.get("finished") if phase == "scored" else False,
            "cache_hit": out.get("cache_hit"),
            "cache_miss": out.get("cache_miss"),
            "token_usage": out.get("token_usage"),
            "reasoning_chars": out.get("reasoning_chars"),
            "answer_chars": out.get("answer_chars"),
            "fallback": fallback,
            "phase": phase,
            "status": scored.get("status") if phase == "scored" else None,
            "move": scored.get("move"),
            "compliance": scored.get("compliance"),
            "correct": correct,
            "oracle": {
                "kind": kind,
                "best_move": extra.get("best_move"),
                "mate_moves": extra.get("mate_moves"),
                "first_move": extra.get("first_move"),
                "cp": extra.get("cp"),
            },
            "fen": rec.get("presented_fen") or rec.get("fen"),
            "pieces": rec.get("pieces"),
            "turn": rec.get("turn"),
            "n": rec.get("n"),
            "task_kind": kind,
            "record_id": rec["id"],
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "model_input_sha256": hashlib.sha256(model_input.encode("utf-8")).hexdigest(),
            "cot_requested": bool(args.cot),
            "position": {
                "id": rec["id"],
                "n": rec.get("n"),
                "turn": rec.get("turn"),
                "fen": rec.get("presented_fen") or rec.get("fen"),
                "pieces": rec.get("pieces") or [],
                "source": "position record",
            },
            "position_metadata": {
                key: rec.get(key) for key in (
                    "source", "puzzle_id", "fen", "presented_fen", "rating",
                    "rating_deviation", "popularity", "nb_plays",
                ) if rec.get(key) is not None
            } | {"task_extra": extra},
        }
        (ROOT / "monitor").mkdir(exist_ok=True)
        (ROOT / "monitor" / "live.json").write_text(json.dumps(live, indent=1))
        # local runs: push live.json straight to the public repo from a
        # BACKGROUND thread (never block the generation loop — a synchronous
        # contents-API round trip per snapshot was throttling token
        # consumption to ~2 tok/s of visible text). Only this one file,
        # ~1-2s apart — well under GitHub's 5,000/hr.
        if args.live_push and live_token:
            with _live_cond:
                _live_pending[0] = (ROOT / "monitor" / "live.json").read_bytes()
                _live_cond.notify()

    samples = []
    t0 = time.time()
    for i, rec in enumerate(records):
        for condition in conditions:
            prompt = T.PROMPT_BUILDERS[kind](rec, condition, variant=args.prompt_variant)
            if args.cot:
                prompt += ("\nThink step by step about the position before answering. "
                           "Then give the final move on its own line.")
            model_input = prompt if args.smoke else model.render_chat(prompt)
            push_live(rec, condition, prompt, model_input,
                      sample_idx=len(samples) + 1,
                      total=len(records) * len(conditions),
                      phase="generating", force=True)
            if args.verbose:
                print(f"\n===== {args.model} x {task_name}:{args.prompt_variant} | "
                      f"{rec['id']} | {i + 1}/{len(records)} =====", flush=True)
                print("--- PROMPT (exactly what the model receives; "
                      "special tokens shown) ---", flush=True)
                if args.smoke:
                    print("(smoke mode — stub model, no real prompt rendering)", flush=True)
                else:
                    print(model.render_chat(prompt), flush=True)
                print("\n--- MODEL OUTPUT (streaming) ---", flush=True)
            def _live_partial(partial: dict) -> None:
                """During generation, surface the model's live thinking to
                live.json (the suite monitor uploads it every 60s)."""
                push_live(rec, condition, prompt, model_input,
                          out={"content": partial.get("content", ""),
                               "reasoning": partial.get("reasoning", ""),
                               "finished": partial.get("finished", False)},
                          sample_idx=len(samples) + 1,
                          total=len(records) * len(conditions),
                          phase="generating", force=True)

            out = model.generate(prompt, max_new_tokens=args.max_new_tokens,
                                 stream=args.verbose or args.stream,
                                 on_chunk=_live_partial if not args.smoke else None)
            fallback = None
            def _answer_has_move(text: str) -> bool:
                """A move counts as given only if the text parses against the
                board — prose without a move is NOT an answer."""
                try:
                    from src.benchmarks.games.tasks import _board, parse_move_output

                    uci, _ = parse_move_output(text or "", _board(rec))
                    return uci is not None
                except Exception:
                    return False

            if not _answer_has_move(out.get("content")) and hasattr(model, "force_answer") \
                    and not args.smoke:
                # 'no answer' is not an option for the API model: force a
                # committed second pass, then extract from the reasoning,
                # then a random legal move — always flagged as a fallback.
                push_live(rec, condition, prompt, model_input,
                          out={"content": "", "reasoning": out.get("reasoning", ""),
                               "finished": True},
                          sample_idx=len(samples) + 1,
                          total=len(records) * len(conditions),
                          phase="forcing", force=True)
                out2 = model.force_answer(prompt)
                out = {**out, "content": out2.get("content", ""),
                       "reasoning": (out.get("reasoning", "") or "")
                                    + "\n\n[forced answer pass]\n"
                                    + (out2.get("reasoning", "") or ""),
                       "cache_hit": out2.get("cache_hit"),
                       "cache_miss": out2.get("cache_miss"),
                       "latency_ms": (out.get("latency_ms") or 0)
                                     + (out2.get("latency_ms") or 0)}
                out["token_usage"] = merge_usage(
                    out.get("token_usage"), out2.get("token_usage"),
                    out.get("latency_ms"))
                merged_usage = out["token_usage"]
                out["input_tokens"] = merged_usage.get("input_tokens")
                out["output_tokens"] = merged_usage.get("output_tokens")
                out["reasoning_tokens"] = merged_usage.get("reasoning_tokens")
                out["total_tokens"] = merged_usage.get("total_tokens")
                out["cache_hit"] = merged_usage.get("cache_hit_tokens")
                out["cache_miss"] = merged_usage.get("cache_miss_tokens")
                out["reasoning_chars"] = len(out.get("reasoning") or "")
                out["answer_chars"] = len(out.get("content") or "")
                fallback = "forced"
                if not _answer_has_move(out.get("content")):
                    # last resort: the best legal move we can extract, or a
                    # random legal move so there is always a comparison
                    from src.benchmarks.games.tasks import _board, parse_move_output
                    try:
                        board = _board(rec)
                        uci, fmt = parse_move_output(out.get("reasoning", ""), board)
                        if uci:
                            out["content"] = f"MOVE: {uci}"
                            fallback = "reasoning-extract"
                    except Exception:
                        pass
                if not _answer_has_move(out.get("content")):
                    from src.benchmarks.games.tasks import _board, _legal_ucis
                    try:
                        board = _board(rec)
                        import random
                        moves = list(_legal_ucis(board))
                        if moves:
                            out["content"] = f"MOVE: {random.choice(moves)}"
                            fallback = "random-legal"
                    except Exception:
                        pass
            if args.verbose:
                print(f"\n--- END (tokens={out.get('output_tokens')}, "
                      f"finished={out.get('finished')}) ---", flush=True)
            scored = T.score_record(rec, condition, out["content"], kind=kind)
            if args.verbose:
                print(f"--- PARSED: {scored}", flush=True)
            sample = {
                "position_id": rec["id"],
                "condition": condition,
                "model": args.model,
                "task": task_name,
                "task_category": task_category,
                "representation": args.prompt_variant,
                "run_id": run_id,
                "value": rec["value"],
                "prompt_tokens": out.get("input_tokens"),
                "output_tokens": out.get("output_tokens"),
                "latency_ms": out.get("latency_ms"),
                "finished": out.get("finished"),
                "max_new_tokens": args.max_new_tokens,
                "thinking_enabled": args.model == "deepseek-v4-flash",
                "prompt": prompt,
                "output": out.get("content", ""),
                "reasoning": out.get("reasoning", ""),
                "reasoning_chars": out.get("reasoning_chars"),
                "answer_chars": out.get("answer_chars"),
                "cache_hit": out.get("cache_hit"),
                "cache_miss": out.get("cache_miss"),
                "token_usage": out.get("token_usage"),
                "fallback": fallback,
                "position_metadata": {
                    key: rec.get(key) for key in (
                        "source", "puzzle_id", "fen", "presented_fen", "rating",
                        "rating_deviation", "popularity", "nb_plays",
                    ) if rec.get(key) is not None
                } | {"task_extra": rec.get("task_extra") or {}},
                "correct": T.get_correct(rec, kind),
                **scored,
            }
            samples.append(sample)
            writer.add(sample)
            push_live(rec, condition, prompt, model_input, out, scored,
                      len(samples), len(records) * len(conditions),
                      phase="scored", force=True, fallback=fallback)
        if (i + 1) % 5 == 0 or i + 1 == len(records):
            el = time.time() - t0
            print(f"  [{task_name} {args.model} {args.prompt_variant}] {i + 1}/{len(records)} "
                  f"({el / (i + 1):.1f}s/position)", flush=True)

    agg = aggregate_samples(samples)
    summary = writer.finish(agg)
    print(json.dumps(summary["metrics"], indent=1), flush=True)
    _cleanup(model)


if __name__ == "__main__":
    main()
