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
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent

from src.benchmarks.games import tasks as T  # noqa: E402
from src.benchmarks.games.envs import ENVS  # noqa: E402
from src.live_push import PUBLIC_LIVE_REPO, resolve_token, upload_file  # noqa: E402
from src.models import HFModel, MODEL_IDS, configure_quiet_logging  # noqa: E402
from src.report import ResultWriter, aggregate_samples, divergence_rate  # noqa: E402

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
    ap.add_argument("--data_dir", default="data/positions")
    ap.add_argument("--output_dir", default="results/chess")
    args = ap.parse_args()

    task_name = args.task
    if task_name in GAME_TASKS:
        n_games = args.n or GAME_N
        model = HFModel(args.model, smoke_test=args.smoke)
        model.load()
        run_name = f"{args.model}_{task_name}_{args.prompt_variant}"
        writer = ResultWriter(
            Path(args.output_dir), run_name,
            {"model": args.model, "task": task_name, "prompt_variant": args.prompt_variant,
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

    model = HFModel(args.model, smoke_test=args.smoke)
    model.load()
    run_name = f"{args.model}_{task_name}_{args.prompt_variant}"
    writer = ResultWriter(
        Path(args.output_dir),
        run_name,
        {"model": args.model, "task": task_name, "prompt_variant": args.prompt_variant,
         "smoke": args.smoke},
    )
    live_token = resolve_token()
    live_last = [0.0]

    def push_live(rec, condition, prompt, model_input, out=None, scored=None,
                  sample_idx=0, total=0, phase="scored", force=False):
        """Publish the exact sample context before and after inference."""
        if not live_token or kind == "game":
            return
        now = time.time()
        if not force and now - live_last[0] < 2.0:
            return
        live_last[0] = now
        out = out or {}
        scored = scored or {}
        correct = T.get_correct(rec, kind)
        live = {
            "updated_at": _utc_ts(),
            "cell": {"model": args.model, "task": task_name, "variant": args.prompt_variant},
            "sample_idx": sample_idx,
            "sample_total": total,
            "position_id": rec["id"],
            "prompt": prompt,
            "model_input": model_input,
            "output": out.get("content", ""),
            "finished": out.get("finished") if phase == "scored" else False,
            "phase": phase,
            "status": scored.get("status") if phase == "scored" else None,
            "move": scored.get("move"),
            "compliance": scored.get("compliance"),
            "correct": correct,
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
        }
        (ROOT / "monitor").mkdir(exist_ok=True)
        (ROOT / "monitor" / "live.json").write_text(json.dumps(live, indent=1))
        try:
            upload_file(live_token, "monitor/live.json",
                        (ROOT / "monitor" / "live.json").read_bytes(),
                        message=f"live {_utc_ts()}")
        except Exception as e:
            print(f"live: push failed ({e})", flush=True)

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
            out = model.generate(prompt, max_new_tokens=args.max_new_tokens,
                                 stream=args.verbose or args.stream)
            if args.verbose:
                print(f"\n--- END (tokens={out.get('output_tokens')}, "
                      f"finished={out.get('finished')}) ---", flush=True)
            scored = T.score_record(rec, condition, out["content"], kind=kind)
            if args.verbose:
                print(f"--- PARSED: {scored}", flush=True)
            sample = {
                "position_id": rec["id"],
                "condition": condition,
                "value": rec["value"],
                "prompt_tokens": out.get("input_tokens"),
                "output_tokens": out.get("output_tokens"),
                "latency_ms": out.get("latency_ms"),
                "finished": out.get("finished"),
                "prompt": prompt,
                "output": out.get("content", ""),
                "correct": T.get_correct(rec, kind),
                **scored,
            }
            samples.append(sample)
            writer.add(sample)
            push_live(rec, condition, prompt, model_input, out, scored,
                      len(samples), len(records) * len(conditions),
                      phase="scored", force=True)
        if (i + 1) % 5 == 0 or i + 1 == len(records):
            el = time.time() - t0
            print(f"  [{task_name} {args.model} {args.prompt_variant}] {i + 1}/{len(records)} "
                  f"({el / (i + 1):.1f}s/position)", flush=True)

    agg = aggregate_samples(samples)
    if kind not in ("cap", "bestmove"):
        agg["divergence_rate"] = divergence_rate(samples)
    summary = writer.finish(agg)
    print(json.dumps(summary["metrics"], indent=1), flush=True)
    _cleanup(model)


if __name__ == "__main__":
    main()
