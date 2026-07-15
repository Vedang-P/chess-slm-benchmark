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


def _load_alphamaze_prompt_template():
    """Load ALPHA_MAZE_PROMPT directly from the alphamaze_reference submodule's
    real benchmark harness (benchmark/models/instruction_type.py -- what
    BaseModel.format_prompt() actually applies via
    prompt_templates[instruction_type].format(maze_prompt=...) in their
    evaluator, the code path that produced the published 93%), by file path
    for the same shadowing reason as _load_alphamaze_bench_utils() above.

    A hand-copied version of this text lived here until this was added: same
    words, but flattened onto fewer lines with different whitespace around
    the bullet list and "MAZE:" separator than their real template -- close
    enough to read the same, not close enough to call byte-for-byte faithful.
    Importing the real constant removes that gap permanently instead of
    relying on a copy staying in sync by hand.
    """
    template_path = (Path(__file__).parent / "alphamaze_reference" / "benchmark"
                      / "models" / "instruction_type.py")
    if not template_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("alphamaze_instruction_type", template_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ALPHA_MAZE_PROMPT


# Fallback only: used if the submodule isn't present (see the warning at
# call time below). Kept textually close to ALPHA_MAZE_PROMPT but was never
# byte-for-byte identical to it -- prefer `git submodule update --init`
# over relying on this.
_MAZEBENCH_PROMPT_FALLBACK = (
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
    "Output only the move tokens, separated by spaces.\nMAZE:\n{maze_prompt}"
)

MAZEBENCH_PROMPT_TEMPLATE = _load_alphamaze_prompt_template() or _MAZEBENCH_PROMPT_FALLBACK
if MAZEBENCH_PROMPT_TEMPLATE is _MAZEBENCH_PROMPT_FALLBACK:
    print("⚠️  alphamaze_reference submodule not found -- using a hand-copied approximation of "
          "AlphaMaze's MazeBench prompt template, NOT their exact template (run "
          "`git submodule update --init` to get it).")

# Used for every MazeBench call EXCEPT the AlphaMaze-checkpoint replication
# cell (see faithful_prompt in eval_mazebench()). ALPHA_MAZE_PROMPT above
# describes the token vocabulary but never states the actual rule for
# whether a move is legal -- AlphaMaze doesn't need that stated explicitly
# because it was trained on hundreds of thousands of these examples
# (Maze-Reasoning-v0.1/GRPO-v0.1) until the mapping became implicit. A model
# seeing this format cold has no such prior: watched on real hardware,
# Gemma 4 E2B correctly read out every cell's wall tokens, then spent its
# entire generation budget (~530s at the 4096-token floor, one maze) flip-
# flopping between plausible-sounding but different interpretations of what
# a wall token actually blocks, never settling on one long enough to
# actually search for a path.
#
# The rule below isn't guessed -- it's read directly off AlphaMaze's own
# real scorer (alphamaze_reference/benchmark/utils.py, simulate_solution()):
#   if direction in grid[current]["walls"]: <blocked>
# i.e. a wall token attached to a cell blocks LEAVING that cell in the
# listed directions, checked only against the cell you are CURRENTLY on --
# never against the destination cell's own wall token, and never reciprocal
# (a wall blocking "up" from cell A says nothing about entering A from
# below). Stating this explicitly doesn't change what's being measured --
# same task, same scoring against the same simulate_solution() -- it just
# stops burning the whole token budget on the model re-deriving a rule it
# was never told and has no way to look up.
MAZEBENCH_PROMPT_TEMPLATE_CLARIFIED = (
    "You are a helpful assistant that solves mazes. You will be given a maze represented by a series of tokens.\n"
    "The tokens represent:\n"
    "- Coordinates: <|row-col|> (e.g., <|0-0|>, <|2-4|>)\n"
    "- Walls: <|no_wall|>, <|up_wall|>, <|down_wall|>, <|left_wall|>, <|right_wall|>, <|up_down_wall|>, etc. -- "
    "one wall token follows each coordinate token, and lists every direction you may NOT leave that cell through.\n"
    "- Origin: <|origin|>\n"
    "- Target: <|target|>\n"
    "- Movement: <|up|>, <|down|>, <|left|>, <|right|>, <|blank|>\n\n"
    "Exact rule for whether a move is legal: to check if you can move from your CURRENT cell in some direction, "
    "look ONLY at your current cell's own wall token -- never the destination cell's wall token. If that "
    "direction appears in your current cell's wall list, the move is illegal. Otherwise the move is legal, as "
    "long as the destination coordinate is still inside the grid. A cell's wall token never restricts moves "
    "INTO it from a neighbor; it only restricts moves OUT of it.\n\n"
    "Worked example: a cell <|1-2|><|up_left_wall|> means from (1,2) you may NOT move up or left, but you MAY "
    "move down or right if those stay in bounds -- regardless of whatever wall token sits at (0,2) or (1,1).\n\n"
    "Your task is to output the sequence of movements (<|up|>, <|down|>, <|left|>, <|right|>) required to "
    "navigate from the origin to the target, based on the provided maze representation. Apply the rule above "
    "directly and spend your reasoning searching for a path -- don't re-derive or second-guess how the walls "
    "work, the rule above is exact and complete. Think step by step. At each step, predict only the next "
    "movement token. Output only the move tokens, separated by spaces.\n"
    "MAZE:\n{maze_prompt}"
)


def _write_report(path: Path, meta: dict, report: dict, complete: bool):
    """Write the current report state to `path`, overwriting it -- called
    after every sample, not just once at the end, so a killed/timed-out run
    (eval_mazebench/eval_gridroute's n-sample loops can each run for hours;
    see the Kaggle notebooks' TIMEOUT_* ceilings) still leaves real partial
    results on disk instead of losing every sample already scored. Writes to
    a temp file and renames over the real path atomically, so a reader never
    sees a half-flushed/corrupt JSON file if the process is killed mid-write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**meta, **report, "complete": complete}
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    tmp_path.replace(path)


def resolve_model_path(args) -> str:
    if args.model == "alphamaze" and not args.checkpoint:
        return ALPHAMAZE_LOCAL_PATH
    return MODEL_IDS.get(args.model, args.model)


def eval_mazebench(model: HFModel, n: int, seed: int, max_new_tokens: int,
                    out_path: Path, save_meta: dict, faithful_prompt: bool = True) -> dict:
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
    whatever move-tokens happen to appear anywhere, reasoning included.

    faithful_prompt=True (AlphaMaze's own checkpoint only) uses their exact
    ALPHA_MAZE_PROMPT, unmodified -- required for the replication check to
    mean anything. Every other model gets MAZEBENCH_PROMPT_TEMPLATE_CLARIFIED
    instead, which states the wall-legality rule explicitly rather than
    leaving a model untrained on this vocabulary to guess it (see that
    template's comment for why, and where the rule was verified from).
    Scoring is identical either way -- still the same simulate_solution().

    Writes `out_path` after every sample via _write_report(), not just once
    at the end, so a subprocess timeout/kill leaves real partial results."""
    from datasets import load_dataset
    ds = load_dataset("Menlo/Maze-Bench-v0.2", split="test")
    rng = np.random.RandomState(seed)
    idx = sorted(rng.choice(len(ds), size=min(n, len(ds)), replace=False))

    if _ALPHAMAZE_BENCH_UTILS is None:
        print("  ⚠️  alphamaze_reference submodule not found -- falling back to exact-match "
              "against Response (NOT their real scoring; run `git submodule update --init`).")

    template = MAZEBENCH_PROMPT_TEMPLATE if faithful_prompt else MAZEBENCH_PROMPT_TEMPLATE_CLARIFIED

    records = []
    n_eval = len(idx)
    report = {"benchmark": "mazebench", "n": n_eval, "n_completed": 0, "correct": 0,
              "accuracy": 0.0, "used_official_scoring": _ALPHAMAZE_BENCH_UTILS is not None,
              "faithful_prompt": faithful_prompt, "records": records}
    correct = 0
    pbar = tqdm(list(enumerate(idx)), desc="MazeBench", unit="maze")
    for i, j in pbar:
        row = ds[int(j)]
        prompt = template.format(maze_prompt=row['Prompt'])
        tqdm.write(f"  [{i + 1}/{len(idx)}] generating (streamed below)...")
        gen = model.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.0)
        print()  # streamed tokens don't end with a newline of their own
        answer = extract_reported_answer(gen["content"], finished=gen["finished"], require_marker=False)

        gt = re.findall(r"<\|(?:up|down|left|right)\|>", row["Response"])
        pred = re.findall(r"<\|(?:up|down|left|right)\|>", answer) if answer else []
        exact_match = bool(answer) and pred == gt

        if _ALPHAMAZE_BENCH_UTILS is not None:
            # Their extract_answer is text.split("</think>")[1] -- simpler
            # than our extract_reported_answer (no FINAL ANSWER/other-marker
            # handling), but this is the exact function that produced the
            # published number, so use it (on the raw generation) rather than
            # our own for this specific faithfulness check. Guarded: that
            # split()[1] raises IndexError outright if "</think>" isn't a
            # literal substring of the output, which only AlphaMaze's own
            # checkpoint is known to reliably produce -- any other model
            # (baselines, our own GRPO checkpoints) could hit this, and
            # without the guard it crashes the whole n-task loop on one bad
            # generation, discarding every result already collected.
            try:
                am_answer = _ALPHAMAZE_BENCH_UTILS.extract_answer(gen["content"])
                ok = bool(am_answer) and _ALPHAMAZE_BENCH_UTILS.benchmark_maze_solution(row["Prompt"], am_answer)
            except Exception:
                ok = False
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

        report["n_completed"] = len(records)
        report["correct"] = correct
        report["accuracy"] = correct / len(records) if records else 0.0
        _write_report(out_path, save_meta, report, complete=False)

    _write_report(out_path, save_meta, report, complete=True)
    return report


def _score_path(path, grid, opt_len):
    valid = bool(path) and _is_in_bounds(path, grid.shape) and _is_collision_free(path, grid) and _is_valid_steps(path)
    optimal = valid and (len(path) - 1 == opt_len)
    return valid, optimal


def eval_gridroute(model: HFModel, n: int, seed: int, grid_size: int, fmt: str, max_new_tokens: int,
                    out_path: Path, save_meta: dict) -> dict:
    """fmt='nl': natural-language coordinate format. fmt='token': our own
    AlphaMaze-vocabulary token-maze format (src/token_maze.py) -- both score
    through the same evaluation.py primitives so they're directly comparable.

    Writes `out_path` after every sample via _write_report(), not just once
    at the end, so a subprocess timeout/kill leaves real partial results."""
    obs_size, n_obs = gridroute_defaults(grid_size)
    tasks = generate_gridroute_maps(size=grid_size, obstacle_size=obs_size, num_obstacles=n_obs,
                                     num_maps=100, pairs_per_map=5, seed=seed)
    rng = np.random.RandomState(seed)
    idx = sorted(rng.choice(len(tasks), size=min(n, len(tasks)), replace=False))

    valid_count, optimal_count = 0, 0
    records = []
    n_eval = len(idx)
    report = {"benchmark": f"gridroute-{fmt}-{grid_size}x{grid_size}", "n": n_eval, "n_completed": 0,
              "valid": 0, "optimal": 0, "valid_rate": 0.0, "optimal_rate": 0.0, "records": records}
    pbar = tqdm(list(enumerate(idx)), desc=f"GridRoute-{fmt}-{grid_size}x{grid_size}", unit="task")
    for i, ti in pbar:
        t = tasks[ti]
        grid = np.array(t.grid)

        if fmt == "token":
            prompt = token_maze.TASK_INSTRUCTION + "\n\n" + token_maze.grid_to_token_maze(grid, t.start, t.goal)
        else:
            prompt = t.nl_variants["direct"] + GRIDROUTE_NL_ANSWER_SUFFIX

        tqdm.write(f"  [{i + 1}/{len(idx)}] generating (streamed below)...")
        gen = model.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.0)
        print()  # streamed tokens don't end with a newline of their own
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

        report["n_completed"] = len(records)
        report["valid"] = valid_count
        report["optimal"] = optimal_count
        report["valid_rate"] = valid_count / len(records) if records else 0.0
        report["optimal_rate"] = optimal_count / len(records) if records else 0.0
        _write_report(out_path, save_meta, report, complete=False)

    _write_report(out_path, save_meta, report, complete=True)
    return report


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
                    help="4096 confirmed sufficient in practice (a real AlphaMaze-on-MazeBench run "
                         "at this budget had all 100 mazes properly close their think tags, see "
                         "idea.md's changelog) -- also the floor for every benchmark below, raise "
                         "explicitly if you have compute to spare. Measured at ~180s/maze for "
                         "MazeBench at 8192 tokens on a T4 -- that alone is ~5h for n=100, more "
                         "than this project's whole Kaggle session budget, which is why the floor "
                         "was brought back down rather than left raised.")
    p.add_argument("--load_in_4bit", action="store_true", default=True)
    p.add_argument("--no_4bit", dest="load_in_4bit", action="store_false")
    p.add_argument("--output_dir", default="./results/eval")
    args = p.parse_args()

    # AlphaMaze never quantizes, regardless of --load_in_4bit/--no_4bit --
    # enforced here rather than left to every call site remembering to pass
    # --no_4bit. It's the one fixed reference point every other number in
    # this project gets compared against (the Phase 1 replication check
    # exists specifically to confirm this harness reproduces its published
    # ~93%), and it's never trained further in Phase 2 -- there's no
    # training-time VRAM constraint forcing a quantization tradeoff here the
    # way there is for Gemma 4, so there's no reason to ever accept the
    # noise 4-bit adds to this specific model's numbers.
    if args.model == "alphamaze" and args.load_in_4bit:
        print("Note: AlphaMaze always runs full precision (ignoring --load_in_4bit default).")
        args.load_in_4bit = False

    model_path = resolve_model_path(args)
    print(f"Model: {model_path}" + (f"  + adapter: {args.checkpoint}" if args.checkpoint else ""))
    print(f"4bit={args.load_in_4bit}  backend={args.backend}  benchmark={args.benchmark}")

    if args.backend == "ollama":
        model = OllamaModel(args.model).load()
    else:
        model = HFModel(model_path, load_in_4bit=args.load_in_4bit,
                         adapter_path=args.checkpoint or None).load()

    # Built before the eval loop starts (not after it returns) so
    # eval_mazebench/eval_gridroute have a stable path to write to
    # incrementally throughout -- see _write_report().
    os.makedirs(args.output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    model_tag = args.model.replace("/", "_")
    out_path = Path(args.output_dir) / f"{model_tag}_{args.benchmark}_{ts}.json"
    # load_in_4bit recorded explicitly -- without it, a 4-bit run and a
    # --no_4bit ablation run of the same model/benchmark/n are
    # indistinguishable in the saved JSON, which defeats the point of
    # running the ablation at all.
    save_meta = {"model": args.model, "checkpoint": args.checkpoint, "seed": args.seed,
                 "load_in_4bit": args.load_in_4bit}
    print(f"Results saved incrementally to: {out_path}")

    if args.benchmark == "mazebench":
        report = eval_mazebench(model, args.n, args.seed, max(args.max_new_tokens, 4096),
                                 out_path, save_meta, faithful_prompt=(args.model == "alphamaze"))
    elif args.benchmark == "gridroute-nl":
        report = eval_gridroute(model, args.n, args.seed, args.grid_size, "nl", max(args.max_new_tokens, 4096),
                                 out_path, save_meta)
    else:
        report = eval_gridroute(model, args.n, args.seed, args.grid_size, "token", max(args.max_new_tokens, 4096),
                                 out_path, save_meta)

    print(f"\nSaved: {out_path}")
    if args.benchmark == "mazebench":
        print(f"MazeBench accuracy: {report['correct']}/{report['n_completed']} completed "
              f"(of {report['n']} requested) ({100 * report['accuracy']:.1f}%)")
    else:
        print(f"{report['benchmark']}: valid={report['valid']}/{report['n_completed']} completed "
              f"(of {report['n']} requested) ({100 * report['valid_rate']:.1f}%)  "
              f"optimal={report['optimal']}/{report['n_completed']} ({100 * report['optimal_rate']:.1f}%)")


if __name__ == "__main__":
    main()
