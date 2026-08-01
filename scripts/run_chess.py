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
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmarks.games import tasks as T  # noqa: E402
from src.benchmarks.games.envs import ENVS  # noqa: E402
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
    ap.add_argument("--prompt-variant", default="grid", choices=["grid", "fen"])
    ap.add_argument("--n", type=int, default=0, help="limit positions (0 = all)")
    ap.add_argument("--conditions", nargs="+", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
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

    samples = []
    t0 = time.time()
    for i, rec in enumerate(records):
        for condition in conditions:
            prompt = T.PROMPT_BUILDERS[kind](rec, condition, variant=args.prompt_variant)
            out = model.generate(prompt, max_new_tokens=args.max_new_tokens)
            scored = T.score_record(rec, condition, out["content"], kind=kind)
            sample = {
                "position_id": rec["id"],
                "condition": condition,
                "value": rec["value"],
                "prompt_tokens": out.get("input_tokens"),
                "output_tokens": out.get("output_tokens"),
                "latency_ms": out.get("latency_ms"),
                "finished": out.get("finished"),
                **scored,
            }
            samples.append(sample)
            writer.add(sample)
        if (i + 1) % 5 == 0 or i + 1 == len(records):
            el = time.time() - t0
            print(f"  [{task_name} {args.model} {args.prompt_variant}] {i + 1}/{len(records)} "
                  f"({el / (i + 1):.1f}s/position)", flush=True)

    agg = aggregate_samples(samples)
    if kind not in ("cap", "bestmove"):
        agg["divergence_rate"] = divergence_rate(samples)
    summary = writer.finish(agg)
    print(json.dumps(summary["metrics"], indent=1), flush=True)


if __name__ == "__main__":
    main()
