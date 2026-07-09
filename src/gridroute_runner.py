"""GridRoute benchmark runner for local Gemma 4.

Runs all baselines and the neuro-symbolic pipeline on generated
GridRoute data and computes evaluation metrics.
"""

import json
import time
from pathlib import Path
from typing import List
from dataclasses import dataclass

from .grid_generator import generate_gridroute_maps, GridRouteTask
from .gemma4_env import Gemma4Env
from .baselines import pure_slm_planner, aop_planner, pure_astar_baseline
from .neuro_symbolic_pipeline import neuro_symbolic_plan
from .evaluation import PathResult, compute_metrics, print_report


GRIDROUTE_CONFIGS = [
    {"size": 10, "obstacle_size": 3, "num_obstacles": 2, "num_maps": 100, "pairs_per_map": 5},
    {"size": 20, "obstacle_size": 4, "num_obstacles": 3, "num_maps": 100, "pairs_per_map": 5},
    {"size": 30, "obstacle_size": 5, "num_obstacles": 4, "num_maps": 100, "pairs_per_map": 5},
]


def run_gridroute_benchmark(
    gemma: Gemma4Env,
    config: dict,
    output_dir: Path,
    seed: int = 42,
) -> dict:
    """Run full GridRoute benchmark for a given config."""
    tasks = generate_gridroute_maps(**config, seed=seed)

    pure_slm_results = []
    aop_results = []
    ns_results = []

    for i, task in enumerate(tasks):
        grid, start, goal = task.grid.grid, task.start, task.goal
        opt_path = task.optimal_path
        opt_len = task.optimal_length

        nl = grid_to_text_instruction(grid, start, goal)

        # Pure SLM
        t0 = time.time()
        path = pure_slm_planner(gemma, grid, start, goal)
        lat = (time.time() - t0) * 1000
        pure_slm_results.append(PathResult(
            path=path, optimal_path=opt_path, optimal_length=opt_len,
            latency_ms=lat,
        ))

        # AoP A*
        t0 = time.time()
        path = aop_planner(gemma, grid, start, goal, algorithm="astar")
        lat = (time.time() - t0) * 1000
        aop_results.append(PathResult(
            path=path, optimal_path=opt_path, optimal_length=opt_len,
            latency_ms=lat,
        ))

        # Neuro-symbolic
        path, constraints, lat, tokens = neuro_symbolic_plan(gemma, grid, nl)
        ns_results.append(PathResult(
            path=path, optimal_path=opt_path, optimal_length=opt_len,
            latency_ms=lat, tokens_generated=tokens,
        ))

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(tasks)}")

    return {
        "config": config,
        "pure_slm": compute_metrics(pure_slm_results),
        "aop_astar": compute_metrics(aop_results),
        "neuro_symbolic": compute_metrics(ns_results),
    }


def grid_to_text_instruction(grid, start, goal):
    """Generate NL instruction for GridRoute task."""
    return (
        f"Plan a path from ({start[0]}, {start[1]}) to ({goal[0]}, {goal[1]}) "
        f"on a {grid.shape[1]}x{grid.shape[0]} grid. Only move up, down, left, right."
    )


def run_all_benchmarks(gemma: Gemma4Env, output_dir: Path):
    """Run benchmarks for all GridRoute configs."""
    all_results = {}
    for cfg in GRIDROUTE_CONFIGS:
        size = cfg["size"]
        print(f"\nRunning GridRoute benchmark: {size}x{size}...")
        results = run_gridroute_benchmark(gemma, cfg, output_dir)
        all_results[f"size_{size}"] = results

        for name in ["pure_slm", "aop_astar", "neuro_symbolic"]:
            print_report(results[name], f"{size}x{size} - {name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    # Serialize (convert dataclasses to dicts)
    serialized = {}
    for k, v in all_results.items():
        serialized[k] = {
            "config": v["config"],
            "pure_slm": v["pure_slm"].__dict__,
            "aop_astar": v["aop_astar"].__dict__,
            "neuro_symbolic": v["neuro_symbolic"].__dict__,
        }
    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump(serialized, f, indent=2, default=str)

    return all_results


# Test
if __name__ == "__main__":
    env = Gemma4Env(load_in_4bit=True, enable_thinking=True)
    env.load()
    results = run_all_benchmarks(env, Path("data/results/"))
