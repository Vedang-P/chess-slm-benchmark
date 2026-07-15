#!/usr/bin/env python3
"""Single evaluation entry point for MazeBench (token format, real published
data) and GridRoute (NL format and our own token-maze format).

Replaces six ad hoc scripts (eval_alphamaze.py, eval_baseline.py,
replicate_alphamaze.py, save_alphamaze_results.py, reproduce_alphamaze_eval.py,
phase1_demo.py) that had drifted into inconsistent prompts, parsers, sampling
settings and sample sizes -- see idea.md's changelog for what that caused.

Always greedy decoding (temperature=0) by default: an eval number meant to be
a fixed, repeatable point shouldn't depend on a stochastic draw. Every path
(NL and token) is parsed with the ONE canonical parser (hf_models.
parse_path_response / token_maze.moves_to_path) and scored with the ONE fixed
evaluation.py, so numbers across models/checkpoints/formats are comparable.

Usage:
  # Phase 1 replication check: does our harness reproduce AlphaMaze's own
  # published MazeBench number on their own checkpoint?
  python eval.py --model alphamaze --benchmark mazebench --n 100

  # Baseline: untrained model on GridRoute NL
  python eval.py --model deepseek-r1-distill-qwen-1.5b --benchmark gridroute-nl --grid_size 5 --n 50

  # A fine-tuned checkpoint from train_grpo.py (LoRA adapter dir)
  python eval.py --model deepseek-r1-distill-qwen-1.5b --checkpoint ./results/grpo_consistency --benchmark gridroute-token --grid_size 5
"""

import argparse
import importlib.util
import json
import os
import re
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from hf_models import (HFModel, OllamaModel, MODEL_IDS, parse_path_response,
                        extract_reported_answer, configure_quiet_logging)

# Must run before anything below that might import alphamaze_reference's
# benchmark/utils.py (which calls logging.basicConfig(level=logging.INFO)
# itself) -- see configure_quiet_logging()'s docstring for why order matters.
configure_quiet_logging()

from src.evaluation import _is_collision_free, _is_in_bounds, _is_valid_steps
from src.grid_generator import generate_gridroute_maps, gridroute_defaults, GRIDROUTE_NL_ANSWER_SUFFIX
from src import token_maze

ALPHAMAZE_LOCAL_PATH = "./data/models/alphamaze-v0.2-1.5b"


def _load_alphamaze_bench_utils():
    """Load AlphaMaze's own benchmark/utils.py directly from the
    alphamaze_reference submodule (github.com/menloresearch/visual-thinker),
    by file path rather than sys.path/package import -- avoids any risk of
    shadowing an unrelated module also named "utils" elsewhere on the path.

    Their real scoring (benchmark_maze_solution) is NOT exact move-sequence
    match against a stored reference answer -- it re-parses the maze's own
    wall structure straight out of the prompt text and simulates the
    candidate's moves against those real walls, accepting ANY sequence that
    actually reaches the target. An earlier version of eval_mazebench in this
    file did exact-match against the dataset's `Response` field instead,
    which would incorrectly fail a valid-but-different solution. Use their
    real code when the submodule is present (faithful to what actually
    produced the published 93%); fall back to exact-match, with a loud
    warning, only if it isn't.
    """
    utils_path = Path(__file__).parent / "alphamaze_reference" / "benchmark" / "utils.py"
    if not utils_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("alphamaze_bench_utils", utils_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ALPHAMAZE_BENCH_UTILS = _load_alphamaze_bench_utils()

# AlphaMaze's own MazeBench system prompt -- confirmed verbatim (word for
# word) against their real public inference.py via the submodule above, not
# just reconstructed from memory. Kept here rather than imported since
# inference.py isn't structured as an importable module.
# evaluates against their real published data/format for the Phase 1
# replication check, not our own token_maze.py encoding.
MAZEBENCH_SYS = (
    "You are a helpful assistant that solves mazes. You will be given a maze represented by "
    "a series of tokens. The tokens represent: "
    "- Coordinates: <|row-col|> (e.g., <|0-0|>, <|2-4|>) "
    "- Walls: <|no_wall|>, <|up_wall|>, <|down_wall|>, <|left_wall|>, <|right_wall|>, <|up_down_wall|>, etc. "
    "- Origin: <|origin|> "
    "- Target: <|target|> "
    "- Movement: <|up|>, <|down|>, <|left|>, <|right|>, <|blank|> "
    "Your task is to output the sequence of movements "
    "(<|up|>, <|down|>, <|left|>, <|right|>) required to navigate "
    "from the origin to the target, based on the provided maze representation. "
    "Think step by step. At each step, predict only the next movement token. "
    "Output only the move tokens, separated by spaces."
)

def resolve_model_path(args) -> str:
    if args.model == "alphamaze" and not args.checkpoint:
        return ALPHAMAZE_LOCAL_PATH
    return MODEL_IDS.get(args.model, args.model)


def eval_mazebench(model: HFModel, n: int, seed: int, max_new_tokens: int) -> dict:
    """Real Menlo/Maze-Bench-v0.2 data. Correctness uses AlphaMaze's own real
    benchmark_maze_solution (via the alphamaze_reference submodule) when
    available: it re-parses the maze's actual wall structure from the prompt
    and simulates the candidate moves against it, accepting ANY sequence that
    reaches the target -- not exact-match against one stored reference
    solution, which would wrongly fail a valid-but-different path. Falls back
    to exact-match against the `Response` field, with a loud warning, only if
    the submodule isn't present (run `git submodule update --init` to get it).

    Moves are read only from the model's own reported answer (post-thinking,
    via extract_reported_answer), never mined out of the raw text including
    any thinking trace -- a "thinking out loud" model that mentions abandoned
    move sequences before its real answer would otherwise get scored against
    whatever move-tokens happen to appear anywhere, reasoning included."""
    from datasets import load_dataset
    ds = load_dataset("Menlo/Maze-Bench-v0.2", split="test")
    rng = np.random.RandomState(seed)
    idx = sorted(rng.choice(len(ds), size=min(n, len(ds)), replace=False))

    if _ALPHAMAZE_BENCH_UTILS is None:
        print("  ⚠️  alphamaze_reference submodule not found -- falling back to exact-match "
              "against Response (NOT their real scoring; run `git submodule update --init`).")

    records = []
    correct = 0
    pbar = tqdm(list(enumerate(idx)), desc="MazeBench", unit="maze")
    for i, j in pbar:
        row = ds[int(j)]
        prompt = f"{MAZEBENCH_SYS}\n\nMAZE: {row['Prompt']}"
        gen = model.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.0)
        answer = extract_reported_answer(gen["content"], finished=gen["finished"], require_marker=False)

        gt = re.findall(r"<\|(?:up|down|left|right)\|>", row["Response"])
        pred = re.findall(r"<\|(?:up|down|left|right)\|>", answer) if answer else []
        exact_match = bool(answer) and pred == gt

        if _ALPHAMAZE_BENCH_UTILS is not None:
            # Their extract_answer is text.split("</think>")[1] -- simpler
            # than our extract_reported_answer (no FINAL ANSWER/other-marker
            # handling), but this is the exact function that produced the
            # published number, so use it (on the raw generation) rather than
            # our own for this specific faithfulness check.
            am_answer = _ALPHAMAZE_BENCH_UTILS.extract_answer(gen["content"])
            ok = bool(am_answer) and _ALPHAMAZE_BENCH_UTILS.benchmark_maze_solution(row["Prompt"], am_answer)
        else:
            ok = exact_match

        correct += int(ok)
        records.append({
            "idx": int(j), "level": row.get("Level"), "correct": ok, "exact_match": exact_match,
            "has_answer": answer is not None, "finished": gen["finished"],
            "gt": gt, "pred": pred, "raw": gen["content"][:2000],
            "ntokens": gen["output_tokens"],
        })
        status = "ok " if ok else ("no-answer" if answer is None else "x  ")
        tqdm.write(f"  [{i + 1}/{len(idx)}] {status} "
                   f"level={row.get('Level')} pred_len={len(pred)} gt_len={len(gt)}")
        pbar.set_postfix(acc=f"{correct}/{i + 1}")

    n_eval = len(idx)
    return {"benchmark": "mazebench", "n": n_eval, "correct": correct,
            "accuracy": correct / n_eval if n_eval else 0.0,
            "used_official_scoring": _ALPHAMAZE_BENCH_UTILS is not None, "records": records}


def _score_path(path, grid, opt_len):
    valid = bool(path) and _is_in_bounds(path, grid.shape) and _is_collision_free(path, grid) and _is_valid_steps(path)
    optimal = valid and (len(path) - 1 == opt_len)
    return valid, optimal


def eval_gridroute(model: HFModel, n: int, seed: int, grid_size: int, fmt: str, max_new_tokens: int) -> dict:
    """fmt='nl': natural-language coordinate format. fmt='token': our own
    AlphaMaze-vocabulary token-maze format (src/token_maze.py) -- both score
    through the same evaluation.py primitives so they're directly comparable."""
    obs_size, n_obs = gridroute_defaults(grid_size)
    tasks = generate_gridroute_maps(size=grid_size, obstacle_size=obs_size, num_obstacles=n_obs,
                                     num_maps=100, pairs_per_map=5, seed=seed)
    rng = np.random.RandomState(seed)
    idx = sorted(rng.choice(len(tasks), size=min(n, len(tasks)), replace=False))

    valid_count, optimal_count = 0, 0
    records = []
    pbar = tqdm(list(enumerate(idx)), desc=f"GridRoute-{fmt}-{grid_size}x{grid_size}", unit="task")
    for i, ti in pbar:
        t = tasks[ti]
        grid = np.array(t.grid)

        if fmt == "token":
            prompt = token_maze.TASK_INSTRUCTION + "\n\n" + token_maze.grid_to_token_maze(grid, t.start, t.goal)
        else:
            prompt = t.nl_variants["direct"] + GRIDROUTE_NL_ANSWER_SUFFIX

        gen = model.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.0)
        # NL format explicitly asks for a "FINAL ANSWER:" line (require_marker=True);
        # token format doesn't use that convention, just move tokens after
        # thinking closes. Either way, never fall back to scanning the raw
        # text including any thinking trace for stray coordinates.
        answer = extract_reported_answer(gen["content"], finished=gen["finished"],
                                          require_marker=(fmt != "token"))

        if answer is None:
            path = None
        elif fmt == "token":
            moves = token_maze.parse_move_tokens(answer)
            path = token_maze.moves_to_path(moves, t.start, grid, goal=t.goal) if moves else None
        else:
            path = parse_path_response(answer, tuple(t.start), tuple(t.goal))

        valid, optimal = _score_path(path, grid, t.optimal_length) if path else (False, False)
        valid_count += int(valid)
        optimal_count += int(optimal)
        records.append({
            "idx": int(ti), "start": list(t.start), "goal": list(t.goal),
            "optimal_length": t.optimal_length, "valid": valid, "optimal": optimal,
            "has_answer": answer is not None, "finished": gen["finished"],
            "path": path, "raw": gen["content"][:2000], "ntokens": gen["output_tokens"],
        })
        status = "OPT  " if optimal else ("VALID" if valid else "INVAL")
        tqdm.write(f"  [{i + 1}/{len(idx)}] {status} start={t.start} goal={t.goal} opt_len={t.optimal_length}")
        pbar.set_postfix(valid=f"{valid_count}/{i + 1}", optimal=f"{optimal_count}/{i + 1}")

    n_eval = len(idx)
    return {"benchmark": f"gridroute-{fmt}-{grid_size}x{grid_size}", "n": n_eval,
            "valid": valid_count, "optimal": optimal_count,
            "valid_rate": valid_count / n_eval if n_eval else 0.0,
            "optimal_rate": optimal_count / n_eval if n_eval else 0.0, "records": records}


def main():
    p = argparse.ArgumentParser(description="Unified eval: MazeBench (real data) + GridRoute (NL/token).")
    p.add_argument("--model", default="deepseek-r1-distill-qwen-1.5b",
                    help="Model key from hf_models.MODEL_IDS, 'alphamaze' for the local checkpoint, "
                         "or any HuggingFace model id.")
    p.add_argument("--checkpoint", default="",
                    help="Local LoRA adapter dir (from train_sft.py/train_grpo.py) to load on top of --model.")
    p.add_argument("--backend", choices=["hf", "ollama"], default="hf")
    p.add_argument("--benchmark", choices=["mazebench", "gridroute-nl", "gridroute-token"], required=True)
    p.add_argument("--grid_size", type=int, default=5)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_new_tokens", type=int, default=4096,
                    help="Real leverage to think fully matters more than speed here -- raised "
                         "automatically to 8192 for token-format benchmarks (MazeBench/token-maze "
                         "thinking traces routinely need that much; 4096 was already found too "
                         "short once, see idea.md's changelog).")
    p.add_argument("--load_in_4bit", action="store_true", default=True)
    p.add_argument("--no_4bit", dest="load_in_4bit", action="store_false")
    p.add_argument("--output_dir", default="./results/eval")
    args = p.parse_args()

    model_path = resolve_model_path(args)
    print(f"Model: {model_path}" + (f"  + adapter: {args.checkpoint}" if args.checkpoint else ""))
    print(f"4bit={args.load_in_4bit}  backend={args.backend}  benchmark={args.benchmark}")

    if args.backend == "ollama":
        model = OllamaModel(args.model).load()
    else:
        model = HFModel(model_path, load_in_4bit=args.load_in_4bit,
                         adapter_path=args.checkpoint or None).load()

    if args.benchmark == "mazebench":
        report = eval_mazebench(model, args.n, args.seed, max(args.max_new_tokens, 8192))
    elif args.benchmark == "gridroute-nl":
        report = eval_gridroute(model, args.n, args.seed, args.grid_size, "nl", max(args.max_new_tokens, 4096))
    else:
        report = eval_gridroute(model, args.n, args.seed, args.grid_size, "token", max(args.max_new_tokens, 8192))

    os.makedirs(args.output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    model_tag = args.model.replace("/", "_")
    out_path = Path(args.output_dir) / f"{model_tag}_{args.benchmark}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "checkpoint": args.checkpoint, "seed": args.seed, **report},
                   f, indent=2, default=str)

    print(f"\nSaved: {out_path}")
    if args.benchmark == "mazebench":
        print(f"MazeBench accuracy: {report['correct']}/{report['n']} ({100 * report['accuracy']:.1f}%)")
    else:
        print(f"{report['benchmark']}: valid={report['valid']}/{report['n']} ({100 * report['valid_rate']:.1f}%)  "
              f"optimal={report['optimal']}/{report['n']} ({100 * report['optimal_rate']:.1f}%)")


if __name__ == "__main__":
    main()
