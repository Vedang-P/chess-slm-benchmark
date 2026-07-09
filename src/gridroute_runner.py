"""GridRoute benchmark runner for local Gemma 4.

Runs all baselines and the neuro-symbolic pipeline on generated
GridRoute data and computes evaluation metrics.
"""

import json
import time
import csv
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

from .grid_generator import generate_gridroute_maps, generate_all_gridroute, GridRouteTask, GRIDROUTE_CONFIGS
from .gemma4_env import Gemma4Env
from .baselines import pure_slm_planner, aop_planner, pure_astar_baseline
from .neuro_symbolic_pipeline import neuro_symbolic_plan
from .evaluation import PathResult, compute_metrics, print_report


def run_method_on_task(
    gemma: Gemma4Env,
    method: str,
    task: GridRouteTask,
    nl_variant: str = "direct",
) -> PathResult:
    """Run a single method on a single task."""
    grid = task.grid
    start = task.start
    goal = task.goal
    nl_instruction = task.nl_variants.get(nl_variant, task.nl_variants["direct"])

    if method == "pure_slm":
        return pure_slm_planner(gemma, grid, start, goal, nl_instruction)
    elif method == "aop_astar":
        return aop_planner(gemma, grid, start, goal, algorithm="astar")
    elif method == "aop_dijkstra":
        return aop_planner(gemma, grid, start, goal, algorithm="dijkstra")
    elif method == "neuro_symbolic":
        return neuro_symbolic_plan(gemma, grid, nl_instruction)
    elif method == "pure_astar":
        return pure_astar_baseline(grid, start, goal)
    else:
        raise ValueError(f"Unknown method: {method}")


def run_benchmark(
    gemma: Gemma4Env,
    tasks: List[GridRouteTask],
    methods: List[str],
    nl_variant: str = "direct",
    checkpoint_dir: Optional[Path] = None,
    resume: bool = True,
) -> Dict[str, List[PathResult]]:
    """Run multiple methods on a list of tasks with checkpointing."""
    results: Dict[str, List[PathResult]] = {m: [] for m in methods}
    total = len(tasks)

    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if resume:
            for m in methods:
                ckpt = checkpoint_dir / f"{m}_checkpoint.json"
                if ckpt.exists():
                    with open(ckpt) as f:
                        loaded = json.load(f)
                    for item in loaded:
                        r = PathResult(
                            path=[tuple(p) for p in item.get("path", [])] if item.get("path") else None,
                            optimal_path=[tuple(p) for p in item["optimal_path"]],
                            optimal_length=item["optimal_length"],
                            tokens_generated=item.get("tokens_generated", 0),
                            latency_ms=item.get("latency_ms", 0),
                            failure_type=item.get("failure_type", ""),
                            extraction_ok=item.get("extraction_ok", True),
                            raw_output=item.get("raw_output", ""),
                            task_id=item.get("task_id", ""),
                        )
                        results[m].append(r)
                    print(f"  Resumed {len(results[m])} {m} results from checkpoint")
                else:
                    results[m] = []

    for i, task in enumerate(tasks):
        task_id = task.task_id
        for method in methods:
            if i < len(results.get(method, [])):
                continue
            try:
                result = run_method_on_task(gemma, method, task, nl_variant)
                result.task_id = task_id
                result.optimal_path = task.optimal_path
                result.optimal_length = task.optimal_length
                results[method].append(result)
            except Exception as e:
                print(f"  ERROR [{task_id}] {method}: {e}")
                results[method].append(PathResult(
                    path=None,
                    optimal_path=task.optimal_path,
                    optimal_length=task.optimal_length,
                    failure_type="runtime_error",
                    task_id=task_id,
                ))

        if (i + 1) % 25 == 0 or i == total - 1:
            print(f"  Progress: {i+1}/{total}")
            if checkpoint_dir:
                _save_checkpoints(checkpoint_dir, results)

    return results


def _save_checkpoints(checkpoint_dir: Path, results: Dict[str, List[PathResult]]):
    """Save intermediate results."""
    for method, res_list in results.items():
        serialized = []
        for r in res_list:
            serialized.append({
                "path": [[int(x), int(y)] for x, y in r.path] if r.path else None,
                "optimal_path": [[int(x), int(y)] for x, y in r.optimal_path],
                "optimal_length": r.optimal_length,
                "tokens_generated": r.tokens_generated,
                "latency_ms": r.latency_ms,
                "vram_peak_gb": r.vram_peak_gb,
                "failure_type": r.failure_type,
                "extraction_ok": r.extraction_ok,
                "raw_output": r.raw_output,
                "task_id": r.task_id,
            })
        with open(checkpoint_dir / f"{method}_checkpoint.json", "w") as f:
            json.dump(serialized, f, indent=1)


def analyze_results(
    results: Dict[str, List[PathResult]],
    tasks: List[GridRouteTask],
    method_names: List[str],
) -> Dict:
    """Compute metrics for all methods."""
    analysis = {}
    for method in method_names:
        if method not in results or not results[method]:
            continue
        grid = tasks[0].grid if tasks else None
        if grid is not None and len(results[method]) > 0:
            report = compute_metrics(results[method], grid)
            analysis[method] = report.__dict__
            print_report(report, method)
    return analysis


def run_full_gridroute_benchmark(
    gemma: Gemma4Env,
    methods: List[str] = None,
    output_dir: Path = None,
    nl_variant: str = "direct",
) -> Dict:
    """Run all 1,500 GridRoute tasks across all methods."""
    if methods is None:
        methods = ["pure_slm", "aop_astar", "neuro_symbolic"]
    if output_dir is None:
        output_dir = Path("data/results")

    all_analysis = {}

    for cfg in GRIDROUTE_CONFIGS:
        label = cfg["label"]
        print(f"\n{'='*60}")
        print(f"GridRoute Benchmark: {label} ({cfg['size']}x{cfg['size']})")
        print(f"{'='*60}")

        tasks = generate_gridroute_maps(
            size=cfg["size"],
            obstacle_size=cfg["obstacle_size"],
            num_obstacles=cfg["num_obstacles"],
            num_maps=100,
            pairs_per_map=5,
            seed=42,
        )
        print(f"  Generated {len(tasks)} tasks")

        ckpt_dir = output_dir / "checkpoints" / label
        results = run_benchmark(gemma, tasks, methods, nl_variant, ckpt_dir)

        analysis = analyze_results(results, tasks, methods)
        all_analysis[label] = analysis

        with open(output_dir / f"results_{label}.json", "w") as f:
            json.dump(analysis, f, indent=2, default=str)

    return all_analysis


if __name__ == "__main__":
    env = Gemma4Env(load_in_4bit=True, enable_thinking=True)
    env.load()
    vram = env.measure_vram()
    print(f"VRAM: {vram}")
    results = run_full_gridroute_benchmark(env)
