#!/usr/bin/env python3
"""Validate neuro_symbolic pipeline on a small sample before full benchmark."""
import sys, time, json
from pathlib import Path

SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC.parent))

from src.ollama_env import OllamaEnv
from src.grid_generator import generate_gridroute_maps, GRIDROUTE_CONFIGS
from src.neuro_symbolic_pipeline import neuro_symbolic_plan, extract_constraints
from src.baselines import pure_astar_baseline
from src.evaluation import _is_collision_free, _is_in_bounds, _is_valid_steps

env = OllamaEnv(model='gemma4-e2b:q3_k_s', base_url='http://localhost:11434')

for cfg_idx, cfg in enumerate(GRIDROUTE_CONFIGS):
    label = cfg["label"]
    print(f"\n{'='*60}")
    print(f"VALIDATING: {label} ({cfg['size']}x{cfg['size']})")
    print(f"{'='*60}")

    tasks = generate_gridroute_maps(
        size=cfg["size"], obstacle_size=cfg["obstacle_size"],
        num_obstacles=cfg["num_obstacles"],
        num_maps=5, pairs_per_map=2, seed=42,
    )
    print(f"  Tasks: {len(tasks)}")

    ns_success = 0
    ns_optimal = 0
    ns_time = 0.0
    errors = {"extract_fail": 0, "no_path": 0, "collision": 0, "mismatch": 0}

    for i, task in enumerate(tasks):
        t0 = time.time()

        # --- neuro_symbolic ---
        res = neuro_symbolic_plan(env, task.grid, task.nl_variants["direct"])
        elapsed = time.time() - t0
        ns_time += elapsed

        # --- pure_astar ---
        astar_path = pure_astar_baseline(task.grid, task.start, task.goal).path

        path_ok = False
        optimal_ok = False
        if res.path and len(res.path) >= 2:
            starts_ok = res.path[0] == task.start
            ends_ok = res.path[-1] == task.goal
            coll_ok = _is_collision_free(res.path, task.grid)
            bounds_ok = _is_in_bounds(res.path, task.grid.shape)
            steps_ok = _is_valid_steps(res.path)
            path_ok = all([starts_ok, ends_ok, coll_ok, bounds_ok, steps_ok])
            if path_ok:
                ns_success += 1
                if len(res.path) - 1 == task.optimal_length:
                    ns_optimal += 1
                is_opt = "OPT" if len(res.path) - 1 == task.optimal_length else f"len={len(res.path)-1}(opt={task.optimal_length})"
            else:
                reasons = []
                if not starts_ok: reasons.append("start")
                if not ends_ok: reasons.append("end")
                if not coll_ok: reasons.append("collision")
                if not bounds_ok: reasons.append("bounds")
                if not steps_ok: reasons.append("steps")
                is_opt = f"INVALID: {','.join(reasons)}"
                errors["mismatch" if (not starts_ok or not ends_ok) else "collision" if not coll_ok else "no_path"] += 1
        else:
            is_opt = f"NO_PATH (extract_ok={res.extraction_ok})"
            if not res.extraction_ok:
                errors["extract_fail"] += 1
            else:
                errors["no_path"] += 1

        # A* path validity
        astar_valid = False
        if astar_path:
            astar_valid = all([
                _is_collision_free(astar_path, task.grid),
                _is_in_bounds(astar_path, task.grid.shape),
                _is_valid_steps(astar_path),
                astar_path[0] == task.start,
                astar_path[-1] == task.goal,
            ])

        print(f"  [{i+1:2d}/{len(tasks)}] "
              f"start={task.start} goal={task.goal} "
              f"ns={is_opt} "
              f"astar_valid={astar_valid} "
              f"astar_len={len(astar_path)-1 if astar_path else 0} "
              f"t={elapsed:.1f}s")

        if not astar_valid:
            print(f"    ⚠️  A* path INVALID!")

    # Summary
    print(f"\n  Summary for {label}:")
    print(f"    Neuro-symbolic: {ns_success}/{len(tasks)} valid ({ns_success/len(tasks)*100:.0f}%)")
    print(f"    Optimal: {ns_optimal}/{ns_success} ({ns_optimal/ns_success*100:.0f}% of valid)" if ns_success else "    Optimal: N/A")
    print(f"    Avg time: {ns_time/len(tasks):.1f}s")
    print(f"    Errors: {errors}")
