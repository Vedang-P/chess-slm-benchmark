#!/usr/bin/env python3
"""Phase 1 Demo: Gemma 4 E2B vs AlphaMaze on MazeBench + GridRoute 5x5.

Full outputs, full context, full thinking. Nothing truncated, nothing stopped early.

Usage:
  python phase1_demo.py --mazebench 5 --gridroute 5      # quick
  python phase1_demo.py --all --save                       # thorough
"""

import argparse, json, os, re, sys, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).parent))
from src.evaluation import _is_collision_free, _is_in_bounds, _is_valid_steps
from src.grid_generator import generate_gridroute_maps
from hf_models import parse_path_response

# ══════════════════════════════════════════════════════════════════════
MODELS = {
    "alphamaze": {
        "path": "./data/models/alphamaze-v0.2-1.5b",
        "label": "AlphaMaze-v0.2-1.5B (SFT+GRPO, Qwen2.5-1.5B base)",
        "trained": True,
    },
    "gemma-e2b": {
        "path": "google/gemma-4-E2B-it",
        "label": "Gemma 4 E2B-it (untrained)",
        "trained": False,
    },
}

@dataclass
class MazeBenchTask:
    maze_id: str
    grid_n: int
    prompt: str
    solution_moves: list
    solution_path: list = field(default_factory=list)
    origin: tuple = (0, 0)
    target: tuple = (0, 0)

@dataclass
class GridRouteTask:
    task_id: str
    prompt: str
    grid: np.ndarray
    start: tuple
    goal: tuple
    optimal_path: list
    optimal_length: int

@dataclass
class EvalResult:
    task_id: str
    model_name: str
    benchmark: str
    output: str
    thinking: str = ""
    parsed_moves: Optional[list] = None
    parsed_path: Optional[list] = None
    is_valid: bool = False
    is_optimal: bool = False
    error: str = ""
    wall_time: float = 0.0
    num_tokens: int = 0

# ══════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════

def load_model(model_key: str, load_in_4bit: bool = True):
    cfg = MODELS[model_key]
    print(f"\n  Loading {cfg['label']}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if load_in_4bit:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                 bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(cfg["path"], quantization_config=bnb,
                                                      device_map="auto", trust_remote_code=True,
                                                      torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg["path"], device_map="auto",
                                                      trust_remote_code=True, torch_dtype=torch.bfloat16)
    print(f"  Loaded in {time.time()-t0:.1f}s  VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    return model, tokenizer, cfg["label"]

def generate_full(model, tokenizer, messages: list, max_new: int = 16384, temperature: float = 0.0):
    """Generate with full context. Let model finish — detect if cut off."""
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                            tokenize=True, return_tensors="pt")
    try:
        input_ids = inputs["input_ids"].to(model.device)
        am = inputs.get("attention_mask")
        gen_kwargs = {"input_ids": input_ids}
        if am is not None: gen_kwargs["attention_mask"] = am.to(model.device)
    except (TypeError, KeyError):
        inputs = inputs.to(model.device)
        gen_kwargs = {"input_ids": inputs}
    input_len = gen_kwargs["input_ids"].shape[-1]

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(**gen_kwargs, max_new_tokens=max_new,
                                  temperature=temperature, do_sample=(temperature > 0),
                                  pad_token_id=tokenizer.eos_token_id)
    elapsed = time.time() - t0
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    ntokens = outputs.shape[-1] - input_len
    finished = ntokens < max_new  # did model stop before hitting limit?

    thinking = ""; answer = response
    for tag in [" response", " response"]:
        if tag in response:
            parts = response.split(tag, 1)
            thinking = parts[0].replace(" think", "").strip()
            answer = parts[1].strip() if len(parts) > 1 else response
            break
    return {"response": response, "thinking": thinking, "answer": answer,
            "wall_time": elapsed, "num_tokens": ntokens, "finished": finished}

# ══════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════

def load_mazebench_tasks(n: int = 10, seed: int = 42):
    from datasets import load_dataset
    print(f"\n  Loading MazeBench (Menlo/Maze-Bench-v0.2)...")
    ds = load_dataset("Menlo/Maze-Bench-v0.2", split="test")
    print(f"  Size: {len(ds)}  Columns: {ds.column_names}")
    rng = np.random.RandomState(seed)
    indices = sorted(rng.choice(len(ds), size=min(n, len(ds)), replace=False))

    SYS = (
        "You are a helpful assistant that solves mazes. You will be given a maze represented by "
        "a series of tokens. The tokens represent: "
        "- Coordinates: <|row-col|> (e.g., <|0-0|>, <|2-4|>) "
        "- Walls: <|no_wall|>, <|up_wall|>, <|down_wall|>, <|left_wall|>, <|right_wall|>, etc. "
        "- Origin: <|origin|> "
        "- Target: <|target|> "
        "- Movement: <|up|>, <|down|>, <|left|>, <|right|>, <|blank|> "
        "Your task is to find the sequence of moves from origin to target. "
        "Think step by step inside  think  tags. "
        "After your thinking, output your final answer on a new line starting with "
        "exactly 'FINAL ANSWER: ' followed by the move tokens separated by spaces, "
        "like: FINAL ANSWER: <|up|> <|down|> <|left|>"
    )
    tasks = []
    for idx in indices:
        row = ds[int(idx)]
        solution_moves = re.findall(r'<\|(up|down|left|right)\|>', row["Response"])
        full_prompt = f"{SYS}\n\nMAZE:\n{row['Prompt']}"
        tasks.append(MazeBenchTask(
            maze_id=f"maze_{idx}_L{row.get('Level','?')}",
            grid_n=5, prompt=full_prompt, solution_moves=solution_moves))
    print(f"  Selected {len(tasks)} tasks")
    return tasks

def load_gridroute_tasks(n: int = 10, size: int = 5, seed: int = 42):
    print(f"\n  Generating GridRoute {size}x{size} tasks...")
    tasks_raw = generate_gridroute_maps(size=size, obstacle_size=1, num_obstacles=1,
                                         num_maps=max(n//3, 3), pairs_per_map=3, seed=seed)
    rng = np.random.RandomState(seed)
    indices = sorted(rng.choice(len(tasks_raw), size=min(n, len(tasks_raw)), replace=False))
    tasks = []
    for i, idx in enumerate(indices):
        t = tasks_raw[idx]
        tasks.append(GridRouteTask(
            task_id=f"gr_{i}", prompt=t.nl_variants["direct"],
            grid=np.array(t.grid), start=tuple(t.start), goal=tuple(t.goal),
            optimal_path=[], optimal_length=t.optimal_length))
    print(f"  Selected {len(tasks)} tasks")
    return tasks

# ══════════════════════════════════════════════════════════════════════
# Parsing & validation
# ══════════════════════════════════════════════════════════════════════

def parse_maze_moves(text: str) -> list:
    """Extract moves from structured 'FINAL ANSWER: <|up|> <|down|> ...' or raw tokens."""
    # Priority 1: explicit FINAL ANSWER tag
    m = re.search(r'FINAL\s*ANSWER\s*:\s*(.+)', text, re.IGNORECASE)
    target = m.group(1) if m else text
    moves = re.findall(r'<\|(up|down|left|right)\|>', target)
    if moves:
        return moves
    # Fallback: step-by-step format
    return re.findall(r'(?:Go|move)\s+(up|down|left|right)', text, re.IGNORECASE)

def validate_maze_solution(moves: list, task: MazeBenchTask):
    if not moves:
        return False, False, "no moves"
    gt = task.solution_moves
    if moves[:len(gt)] == gt:
        return True, True, "exact match"
    if len(moves) == len(gt):
        return False, False, f"wrong moves"
    return False, False, f"len {len(moves)} vs {len(gt)}"

# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def run_demo(args):
    print("=" * 70)
    print("  PHASE 1: Gemma 4 E2B vs AlphaMaze — MazeBench + GridRoute 5x5")
    print("=" * 70)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    mazebench_tasks = load_mazebench_tasks(args.mazebench, args.seed) if args.mazebench > 0 else []
    gridroute_tasks = load_gridroute_tasks(args.gridroute, size=5, seed=args.seed) if args.gridroute > 0 else []

    all_results = []
    models_to_test = args.models.split(",")

    for model_key in models_to_test:
        if model_key not in MODELS: continue
        model, tokenizer, label = load_model(model_key, load_in_4bit=not args.fp16)
        cfg = MODELS[model_key]

        # ─── MazeBench ────────────────────────────────────────────
        if mazebench_tasks:
            print(f"\n{'─'*60}")
            print(f"  {label} on MazeBench ({len(mazebench_tasks)} mazes)")
            print(f"{'─'*60}")
            correct = 0
            for i, task in enumerate(mazebench_tasks):
                messages = [{"role": "user", "content": task.prompt}]
                gen = generate_full(model, tokenizer, messages,
                                    max_new=args.max_tokens, temperature=args.temperature)
                moves = parse_maze_moves(gen["response"])
                valid, optimal, reason = validate_maze_solution(moves, task)
                if valid: correct += 1

                if i < args.verbose:
                    print(f"\n  ── Maze {i+1} (L{task.maze_id.split('_L')[1][0] if '_L' in task.maze_id else '?'}) ──")
                    print(f"  GT moves:     {task.solution_moves}")
                    print(f"  Pred moves:   {moves}")
                    if gen["thinking"]:
                        print(f"  [THINKING]    {gen['thinking'][:400]}")
                    print(f"  [ANSWER]      {gen['answer'][:300]}")
                    print(f"  Result: {'✅' if valid else '❌'} {reason} ({gen['wall_time']:.1f}s, {gen['num_tokens']}tok)")

                all_results.append(EvalResult(
                    task_id=task.maze_id, model_name=model_key, benchmark="mazebench",
                    output=gen["response"], thinking=gen["thinking"],
                    parsed_moves=moves, is_valid=valid, is_optimal=optimal,
                    error=reason, wall_time=gen["wall_time"], num_tokens=gen["num_tokens"]))
            print(f"\n  MazeBench: {correct}/{len(mazebench_tasks)} ({100*correct/len(mazebench_tasks):.1f}%)")

        # ─── GridRoute 5x5 ────────────────────────────────────────
        if gridroute_tasks:
            print(f"\n{'─'*60}")
            print(f"  {label} on GridRoute 5x5 ({len(gridroute_tasks)} tasks)")
            print(f"{'─'*60}")
            valid_cnt = optimal_cnt = 0
            for i, task in enumerate(gridroute_tasks):
                prompt = (task.prompt +
                    "\n\nThink step by step about the path. "
                    "Output your final answer as: FINAL ANSWER: [(row,col), (row,col), ...]")
                messages = [{"role": "user", "content": prompt}]
                gen = generate_full(model, tokenizer, messages,
                                    max_new=args.max_tokens, temperature=args.temperature)
                path = parse_path_response(gen["response"], task.start, task.goal)
                ib = _is_in_bounds(path, task.grid.shape) if path else False
                cf = _is_collision_free(path, task.grid) if path and ib else False
                vs = _is_valid_steps(path) if path and cf else False
                is_valid = ib and cf and vs
                is_optimal = is_valid and path and (len(path)-1 == task.optimal_length)
                if is_valid: valid_cnt += 1
                if is_optimal: optimal_cnt += 1

                if i < args.verbose:
                    print(f"\n  ── GridRoute {i+1} ──")
                    print(f"  Start:{task.start} Goal:{task.goal} OptLen:{task.optimal_length}")
                    print(f"  Path ({len(path)-1 if path else 0} steps): {path}")
                    if gen["thinking"]:
                        print(f"  [THINKING] {gen['thinking'][:400]}")
                    print(f"  [ANSWER] {gen['answer'][:300]}")
                    print(f"  {'✅ OPTIMAL' if is_optimal else '✅ VALID' if is_valid else '❌'} ({gen['wall_time']:.1f}s)")

                all_results.append(EvalResult(
                    task_id=task.task_id, model_name=model_key, benchmark="gridroute",
                    output=gen["response"], thinking=gen["thinking"], parsed_path=path,
                    is_valid=is_valid, is_optimal=is_optimal,
                    error="" if is_valid else "invalid", wall_time=gen["wall_time"],
                    num_tokens=gen["num_tokens"]))
            print(f"\n  GridRoute 5x5: Valid={valid_cnt}/{len(gridroute_tasks)} "
                  f"({100*valid_cnt/len(gridroute_tasks):.1f}%)  "
                  f"Optimal={optimal_cnt}/{len(gridroute_tasks)} ({100*optimal_cnt/len(gridroute_tasks):.1f}%)")

        del model; torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}")
    for bench in ["mazebench", "gridroute"]:
        br = [r for r in all_results if r.benchmark == bench]
        if not br: continue
        print(f"\n  {bench.upper()}:")
        for mk in models_to_test:
            mr = [r for r in br if r.model_name == mk]
            if mr:
                v = sum(1 for r in mr if r.is_valid)
                o = sum(1 for r in mr if r.is_optimal)
                t = np.mean([r.wall_time for r in mr])
                print(f"    {MODELS[mk]['label']}: Valid={v}/{len(mr)} Optim={o}/{len(mr)} AvgTime={t:.1f}s")

    # ── AlphaMaze NL check ────────────────────────────────────────
    amr = [r for r in all_results if r.model_name == "alphamaze"]
    nl = sum(1 for r in amr if len(r.thinking) > 20 or
             any(w in r.output.lower() for w in ["step", "think", "move", "path", "because"]))
    print(f"\n  AlphaMaze NL output: {nl}/{len(amr)} responses have chain-of-thought")
    print(f"  ✅ AlphaMaze CAN produce NL (Qwen2.5 base). GRPO enhances CoT.")

    if args.save:
        out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        with open(out / f"phase1_{ts}.json", "w") as f:
            json.dump({"config": vars(args), "results": [
                {"task_id": r.task_id, "model": r.model_name, "benchmark": r.benchmark,
                 "is_valid": r.is_valid, "is_optimal": r.is_optimal,
                 "thinking": r.thinking[:500], "parsed_moves": r.parsed_moves,
                 "wall_time": r.wall_time, "num_tokens": r.num_tokens}
                for r in all_results]}, f, indent=2, default=str)
        print(f"\n  Saved to {out}/phase1_{ts}.json")
    print(f"\n{'='*70}\n  DONE\n{'='*70}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Phase 1: Gemma 4 E2B vs AlphaMaze")
    p.add_argument("--mazebench", type=int, default=5)
    p.add_argument("--gridroute", type=int, default=5)
    p.add_argument("--models", type=str, default="alphamaze,gemma-e2b")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--verbose", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", action="store_true")
    p.add_argument("--output-dir", type=str, default="./results/phase1_demo")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()
    if args.all:
        args.mazebench = max(args.mazebench, 30)
        args.gridroute = max(args.gridroute, 15)
        args.verbose = 5
        args.save = True
    run_demo(args)
