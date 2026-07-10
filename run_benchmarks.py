#!/usr/bin/env python3
"""Run all benchmarks with Ollama-backed Gemma 4 + qwen2.5-coder."""

import sys
import json
from pathlib import Path

SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC.parent))

from src.ollama_env import OllamaEnv
from src.grid_generator import generate_gridroute_maps, GRIDROUTE_CONFIGS, GridRouteTask
from src.baselines import pure_slm_planner, pure_astar_baseline
from src.neuro_symbolic_pipeline import neuro_symbolic_plan
from src.evaluation import PathResult, compute_metrics, print_report


def run_method(method, env, task, nl_variant="direct"):
    grid = task.grid
    start = task.start
    goal = task.goal
    nl_instruction = task.nl_variants.get(nl_variant, task.nl_variants["direct"])

    if method == "pure_slm":
        return pure_slm_planner(env, grid, start, goal, nl_instruction)
    elif method == "neuro_symbolic":
        return neuro_symbolic_plan(env, grid, nl_instruction)
    elif method == "pure_astar":
        return pure_astar_baseline(grid, start, goal)
    raise ValueError(f"Unknown method: {method}")


CHECKPOINT_INTERVAL = 25  # Save checkpoint every N tasks


def _save_checkpoint(path: Path, results: dict, completed: int, total: int, method_order: list):
    """Save intermediate checkpoint as JSON-serializable dict."""
    cp = {
        "completed": completed,
        "total": total,
        "results": {},
        "methods": method_order,
    }
    for method, res_list in results.items():
        cp["results"][method] = [
            {
                "path": r.path,
                "optimal_path": r.optimal_path,
                "optimal_length": r.optimal_length,
                "tokens_generated": r.tokens_generated,
                "latency_ms": r.latency_ms,
                "failure_type": r.failure_type,
                "extraction_ok": r.extraction_ok,
                "task_id": r.task_id,
            }
            for r in res_list
        ]
    with open(path, "w") as f:
        json.dump(cp, f, indent=2, default=str)
    print(f"  Checkpoint saved: {completed}/{total}")


def _load_checkpoint(path: Path, methods: list):
    """Load checkpoint and return (results_list_dict, completed_count)."""
    with open(path) as f:
        cp = json.load(f)
    results = {m: [] for m in methods}
    for method, res_list in cp.get("results", {}).items():
        for r_data in res_list:
            r = PathResult(
                path=r_data.get("path"),
                optimal_path=r_data.get("optimal_path", [(0, 0), (0, 1)]),
                optimal_length=r_data.get("optimal_length", 0),
                tokens_generated=r_data.get("tokens_generated", 0),
                latency_ms=r_data.get("latency_ms", 0),
                failure_type=r_data.get("failure_type", ""),
                extraction_ok=r_data.get("extraction_ok", True),
                task_id=r_data.get("task_id", ""),
            )
            results[method].append(r)
    return results, cp.get("completed", 0)


def run_gridroute(env, methods, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for cfg in GRIDROUTE_CONFIGS:
        label = cfg["label"]
        out_file = output_dir / f"gridroute_{label}.json"
        cp_file = output_dir / f"gridroute_{label}_checkpoint.json"

        # Check if final results exist
        if out_file.exists():
            print(f"\n  SKIP {label}: already exists at {out_file}")
            with open(out_file) as f:
                all_results[label] = json.load(f)
            continue

        print(f"\n{'='*60}")
        print(f"GridRoute: {label} ({cfg['size']}x{cfg['size']})")
        print(f"{'='*60}")

        tasks = generate_gridroute_maps(
            size=cfg["size"], obstacle_size=cfg["obstacle_size"],
            num_obstacles=cfg["num_obstacles"],
            num_maps=100, pairs_per_map=5, seed=42,
        )
        n_total = len(tasks)
        print(f"  Tasks: {n_total}")

        # Resume from checkpoint if exists
        if cp_file.exists():
            results, completed = _load_checkpoint(cp_file, methods)
            print(f"  Resuming from checkpoint: {completed}/{n_total} tasks done")
        else:
            results = {m: [] for m in methods}
            completed = 0

        for i in range(completed, n_total):
            task = tasks[i]
            for method in methods:
                if method == "pure_astar":
                    res = run_method(method, env, task)
                    res.task_id = task.task_id
                    res.optimal_path = task.optimal_path
                    res.optimal_length = task.optimal_length
                    results[method].append(res)
                    continue

                try:
                    res = run_method(method, env, task)
                    res.task_id = task.task_id
                    res.optimal_path = task.optimal_path
                    res.optimal_length = task.optimal_length
                    results[method].append(res)
                except Exception as e:
                    print(f"  ERROR [{task.task_id}] {method}: {e}")
                    results[method].append(PathResult(
                        path=None, optimal_path=task.optimal_path,
                        optimal_length=task.optimal_length,
                        failure_type="runtime_error", task_id=task.task_id,
                    ))

            # Save checkpoint periodically
            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                _save_checkpoint(cp_file, results, i + 1, n_total, methods)
                print(f"  Progress: {i+1}/{n_total}")

        # Final: save results, remove checkpoint
        if cp_file.exists():
            cp_file.unlink()

        grids_list = [t.grid for t in tasks] if len(results[methods[0]]) == n_total else None
        analysis = {}
        for method in methods:
            if results[method]:
                report = compute_metrics(results[method], grids=grids_list)
                analysis[method] = report.__dict__
                print_report(report, method)

        all_results[label] = analysis
        with open(out_file, "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"  ✅ {label} complete, saved to {out_file}")

    return all_results


def run_lost_in_agg(env, methods, output_dir):
    from src.grid_generator import grid_to_nl_variants
    import numpy as np

    data_dir = Path("data/lost_in_aggregation")
    output_dir.mkdir(parents=True, exist_ok=True)
    maze_sizes = [3, 5, 7, 10, 15]
    all_results = {}

    for size in maze_sizes:
        path = data_dir / f"size_{size}_150.json"
        if not path.exists():
            print(f"  SKIP size={size}: file not found")
            continue

        with open(path) as f:
            mazes = json.load(f)

        print(f"\n{'='*60}")
        print(f"Lost in Aggregation: size={size}, mazes={len(mazes)}")
        print(f"{'='*60}")

        results = {m: [] for m in methods}
        for i, maze in enumerate(mazes):
            grid = np.array(maze["grid"])
            nl_instruction = maze.get("nl_instruction", "")
            start = tuple(maze.get("start", (0, 0)))
            goal = tuple(maze.get("goal", (grid.shape[0]-1, grid.shape[1]-1)))

            dummy = GridRouteTask(grid=grid, start=start, goal=goal,
                                  task_id=f"lia_{size}_{i}")
            dummy.nl_variants = {"direct": nl_instruction or f"Navigate from {start} to {goal}"}
            dummy.optimal_path = maze.get("solution", [start, goal])
            dummy.optimal_length = len(dummy.optimal_path) - 1 if len(dummy.optimal_path) > 1 else 1

            for method in methods:
                if method == "pure_astar":
                    try:
                        res = pure_astar_baseline(grid, start, goal)
                        res.task_id = dummy.task_id
                        res.optimal_path = dummy.optimal_path
                        res.optimal_length = dummy.optimal_length
                        results[method].append(res)
                    except Exception as e:
                        results[method].append(PathResult(
                            path=None, optimal_path=dummy.optimal_path,
                            optimal_length=dummy.optimal_length,
                            failure_type="runtime_error", task_id=dummy.task_id,
                        ))
                    continue

                try:
                    res = run_method(method, env, dummy)
                    res.task_id = dummy.task_id
                    res.optimal_path = dummy.optimal_path
                    res.optimal_length = dummy.optimal_length
                    results[method].append(res)
                except Exception as e:
                    print(f"  ERROR [{dummy.task_id}] {method}: {e}")
                    results[method].append(PathResult(
                        path=None, optimal_path=dummy.optimal_path,
                        optimal_length=dummy.optimal_length,
                        failure_type="runtime_error", task_id=dummy.task_id,
                    ))

            if (i + 1) % 30 == 0 or i == len(mazes) - 1:
                print(f"  Progress: {i+1}/{len(mazes)}")

        all_grids = [np.array(m["grid"]) for m in mazes]
        analysis = {}
        for method in methods:
            if results[method]:
                report = compute_metrics(results[method], grids=all_grids)
                analysis[method] = report.__dict__
                print_report(report, method)

        all_results[f"size_{size}"] = analysis
        with open(output_dir / f"lost_in_agg_size_{size}.json", "w") as f:
            json.dump(analysis, f, indent=2, default=str)

    return all_results


if __name__ == "__main__":
    import time

    MODEL = "gemma4-e2b:q3_k_s"

    env = OllamaEnv(model=MODEL, base_url="http://localhost:11434")
    print(f"Using model: {MODEL}")

    # Gemma 4 pure_slm is slow (~28s/task) and often fails (overthinks).
    # Run it on a small subset for failure analysis; main results use neuro_symbolic + astar.
    methods = ["neuro_symbolic", "pure_astar"]
    out = Path("data/results") / MODEL.replace(":", "_")

    # GridRoute
    t0 = time.time()
    gridroute_results = run_gridroute(env, methods, out)
    print(f"\nGridRoute done in {(time.time()-t0)/60:.1f} min")

    # Lost in Aggregation
    t0 = time.time()
    lia_results = run_lost_in_agg(env, methods, out)
    print(f"\nLost in Aggregation done in {(time.time()-t0)/60:.1f} min")

    print(f"\nAll results saved to {out}/")

    # Optional: run pure_slm on a small subset for failure analysis
    print("\nRunning pure_slm subset for failure analysis...")
    ps_out = out / "pure_slm_subset"
    ps_out.mkdir(parents=True, exist_ok=True)
    from src.grid_generator import generate_gridroute_maps
    sample_tasks = generate_gridroute_maps(
        size=GRIDROUTE_CONFIGS[0]["size"],
        obstacle_size=GRIDROUTE_CONFIGS[0]["obstacle_size"],
        num_obstacles=GRIDROUTE_CONFIGS[0]["num_obstacles"],
        num_maps=5, pairs_per_map=2, seed=99,
    )
    ps_results = []
    for task in sample_tasks:
        t1 = time.time()
        res = pure_slm_planner(env, task.grid, task.start, task.goal,
                                task.nl_variants["direct"])
        res.task_id = task.task_id
        res.optimal_path = task.optimal_path
        res.optimal_length = task.optimal_length
        ps_results.append(res)
        print(f"  [{task.task_id}] path={len(res.path) if res.path else 0}, "
              f"t={time.time()-t1:.0f}s")

    report = compute_metrics(ps_results, sample_tasks[0].grid)
    print_report(report, "pure_slm")
    with open(ps_out / "results.json", "w") as f:
        json.dump(report.__dict__, f, indent=2, default=str)
