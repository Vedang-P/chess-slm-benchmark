#!/usr/bin/env python3
"""Compare neuro_symbolic vs pure_slm on small validation set."""
import sys, time, json
from pathlib import Path

SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC.parent))

from src.ollama_env import OllamaEnv
from src.grid_generator import generate_gridroute_maps, GRIDROUTE_CONFIGS
from src.neuro_symbolic_pipeline import neuro_symbolic_plan
from src.baselines import pure_slm_planner, pure_astar_baseline
from src.evaluation import _is_collision_free

env = OllamaEnv(model='gemma4-e2b:q3_k_s', base_url='http://localhost:11434')

for cfg in GRIDROUTE_CONFIGS:
    label = cfg["label"]
    print(f"\n{'='*60}")
    print(f"{label} ({cfg['size']}x{cfg['size']})")
    print(f"{'='*60}")

    tasks = generate_gridroute_maps(
        size=cfg["size"], obstacle_size=cfg["obstacle_size"],
        num_obstacles=cfg["num_obstacles"],
        num_maps=3, pairs_per_map=2, seed=42,
    )

    ns_ok, ns_opt, ns_time, ns_tokens = 0, 0, 0.0, 0
    ps_ok, ps_opt, ps_time, ps_tokens = 0, 0, 0.0, 0

    for i, task in enumerate(tasks):
        # neuro_symbolic
        t0 = time.time()
        ns_res = neuro_symbolic_plan(env, task.grid, task.nl_variants["direct"])
        ns_t = time.time() - t0
        ns_time += ns_t

        ns_valid = False
        if ns_res.path and len(ns_res.path) >= 2:
            if _is_collision_free(ns_res.path, task.grid) and ns_res.path[0] == task.start and ns_res.path[-1] == task.goal:
                ns_valid = True
                ns_ok += 1
                if len(ns_res.path) - 1 == task.optimal_length:
                    ns_opt += 1

        # pure_slm (Gemma generates path directly)
        t0 = time.time()
        ps_res = pure_slm_planner(env, task.grid, task.start, task.goal, task.nl_variants["direct"])
        ps_t = time.time() - t0
        ps_time += ps_t

        ps_valid = False
        if ps_res.path and len(ps_res.path) >= 2:
            if _is_collision_free(ps_res.path, task.grid) and ps_res.path[0] == task.start and ps_res.path[-1] == task.goal:
                ps_valid = True
                ps_ok += 1
                if len(ps_res.path) - 1 == task.optimal_length:
                    ps_opt += 1

        print(f"  [{i+1:2d}] ns={'✓' if ns_valid else '✗'} (opt={ns_opt if ns_valid else '-'} t={ns_t:.1f}s) "
              f"ps={'✓' if ps_valid else '✗'} (opt={ps_opt if ps_valid else '-'} t={ps_t:.1f}s) "
              f"astar_opt_len={task.optimal_length}")

        # Show raw output when pure_slm fails
        if not ps_valid:
            print(f"        ps_raw[:300]: {ps_res.raw_output[:300]}")

    n = len(tasks)
    print(f"\n  Summary:")
    print(f"    neuro_symbolic: {ns_ok}/{n} valid ({ns_ok/n*100:.0f}%), {ns_opt}/{n} optimal, "
          f"avg {ns_time/n:.1f}s")
    print(f"    pure_slm:      {ps_ok}/{n} valid ({ps_ok/n*100:.0f}%), {ps_opt}/{n} optimal, "
          f"avg {ps_time/n:.1f}s")
    print(f"    pure_astar:     {n}/{n} valid (100%), {n}/{n} optimal (100%, ground truth)")
