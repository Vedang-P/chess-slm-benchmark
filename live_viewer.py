#!/usr/bin/env python3
"""Live viewer: watch each task as Gemma solves it."""
import sys, time, json, os
from pathlib import Path

SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC.parent))

from src.ollama_env import OllamaEnv
from src.grid_generator import generate_gridroute_maps, GRIDROUTE_CONFIGS
from src.neuro_symbolic_pipeline import neuro_symbolic_plan
from src.baselines import pure_astar_baseline, PURE_SLM_PROMPT_TEMPLATE
from src.astar_solver import astar
from src.evaluation import _is_collision_free

env = OllamaEnv(model='gemma4-e2b:q3_k_s', base_url='http://localhost:11434')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=int, default=1, help="Config index (0=10, 1=20, 2=30)")
parser.add_argument("--num", type=int, default=10, help="Number of tasks to show")
parser.add_argument("--show-raw", action="store_true", help="Show raw Gemma output")
args = parser.parse_args()

cfg = GRIDROUTE_CONFIGS[args.config]
tasks = generate_gridroute_maps(
    size=cfg["size"], obstacle_size=cfg["obstacle_size"],
    num_obstacles=cfg["num_obstacles"],
    num_maps=args.num, pairs_per_map=1, seed=42,
)

for i, task in enumerate(tasks):
    os.system('clear')
    print("=" * 70)
    print(f"TASK {i+1}/{len(tasks)}  |  {cfg['label']}  |  {task.task_id}")
    print("=" * 70)

    # ── Grid visualization ──
    grid = task.grid
    print(f"\nGrid ({grid.shape[1]}x{grid.shape[0]}):")
    print("   " + "".join(f"{x:2}" for x in range(grid.shape[1])))
    for y in range(grid.shape[0]):
        row = f"{y:2} "
        for x in range(grid.shape[1]):
            if (x, y) == task.start:
                row += " S"
            elif (x, y) == task.goal:
                row += " G"
            elif grid[y, x] == 1:
                row += "██"
            else:
                row += " ."
        print(row)

    print(f"\n  Start: {task.start}  →  Goal: {task.goal}")
    print(f"  NL: {task.nl_variants['direct'][:120]}...")

    # ── neuro_symbolic ──
    t0 = time.time()
    res = neuro_symbolic_plan(env, task.grid, task.nl_variants["direct"])
    t = time.time() - t0

    valid = False
    if res.path and len(res.path) >= 2:
        valid = (_is_collision_free(res.path, task.grid) and
                 res.path[0] == task.start and res.path[-1] == task.goal)

    optimal = astar(task.grid, task.start, task.goal) if valid else None
    is_opt = (valid and optimal and len(res.path) - 1 == len(optimal) - 1)

    status = "✅ VALID" if valid else "❌ FAIL"
    if is_opt:
        status += " ★OPTIMAL"
    elif valid:
        status += f" (subopt: {len(res.path)-1} vs opt {len(optimal)-1})"

    print(f"\n  Gemma + A*: {status}  ({t:.1f}s)")
    print(f"  Path: {res.path}")

    if args.show_raw and hasattr(res, 'raw_output') and res.raw_output:
        print(f"\n  Gemma raw: {res.raw_output[:300]}...")

    # ── A* optimal ──
    if optimal:
        print(f"  A* optimal: {optimal}  ({len(optimal)-1} steps)")

    # ── pure_slm comparison ──
    if i < 2:  # Only for first 2 tasks (slow)
        t0 = time.time()
        obstacles = [(int(x), int(y)) for x in range(task.grid.shape[1])
                     for y in range(task.grid.shape[0]) if task.grid[y, x] == 1]
        prompt = PURE_SLM_PROMPT_TEMPLATE.format(
            start=task.start, goal=task.goal, obstacles=obstacles)
        result = env.generate([{"role": "user", "content": prompt}], max_tokens=4096)
        ps_t = time.time() - t0
        ps_content = result.get("content", "")
        from src.baselines import _parse_path_response
        ps_path = _parse_path_response(ps_content, task.grid, task.start, task.goal)
        ps_valid = ps_path is not None and len(ps_path) >= 2
        print(f"\n  ⚠️  Pure SLM (Gemma alone): {'✅' if ps_valid else '❌'}  ({ps_t:.0f}s)")
        if args.show_raw:
            print(f"  Raw: {ps_content[:400]}")

    print(f"\n  {'─' * 70}")
    time.sleep(1.5)  # Auto-advance

print("\n✅ Done!")
