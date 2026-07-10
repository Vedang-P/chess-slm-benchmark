#!/usr/bin/env python3
"""Validate two things:
1. pure_slm with higher token budget (4096)
2. neuro_symbolic contribution across NL variants
"""
import sys, time
from pathlib import Path

SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC.parent))

from src.ollama_env import OllamaEnv
from src.grid_generator import generate_gridroute_maps, GRIDROUTE_CONFIGS
from src.neuro_symbolic_pipeline import neuro_symbolic_plan
from src.baselines import pure_slm_planner, pure_astar_baseline
from src.evaluation import _is_collision_free

env = OllamaEnv(model='gemma4-e2b:q3_k_s', base_url='http://localhost:11434')

# ── Part 1: pure_slm with 4096 tokens on size_10 ──
print("=" * 60)
print("PART 1: pure_slm with max_tokens=4096")
print("=" * 60)

cfg = GRIDROUTE_CONFIGS[0]
tasks = generate_gridroute_maps(
    size=cfg["size"], obstacle_size=cfg["obstacle_size"],
    num_obstacles=cfg["num_obstacles"],
    num_maps=5, pairs_per_map=2, seed=42,
)

ps_ok, ps_opt, ps_time = 0, 0, 0.0
for i, task in enumerate(tasks):
    t0 = time.time()
    # Override max_tokens to 4096
    from src.baselines import PURE_SLM_PROMPT_TEMPLATE
    obstacles = [(int(x), int(y)) for x in range(task.grid.shape[1])
                 for y in range(task.grid.shape[0]) if task.grid[y, x] == 1]
    prompt = PURE_SLM_PROMPT_TEMPLATE.format(
        start=task.start, goal=task.goal, obstacles=obstacles)

    # Call generate directly with 4096 tokens
    result = env.generate([{"role": "user", "content": prompt}], max_tokens=4096)
    content = result.get("content", "") if isinstance(result, dict) else str(result)

    # Parse path
    from src.baselines import _parse_path_response
    path = _parse_path_response(content, task.grid, task.start, task.goal)
    t = time.time() - t0
    ps_time += t

    valid = False
    if path and len(path) >= 2:
        valid = (_is_collision_free(path, task.grid) and
                 path[0] == task.start and path[-1] == task.goal)
        if valid:
            ps_ok += 1
            if len(path) - 1 == task.optimal_length:
                ps_opt += 1

    print(f"  [{i+1:2d}] valid={valid} len={len(path)-1 if path else 0} "
          f"opt={task.optimal_length} t={t:.1f}s toks={result.get('output_tokens', 0)}")

    if not valid:
        print(f"        raw_last_200: ...{content[-200:] if content else 'empty'}")

n = len(tasks)
print(f"\n  pure_slm@4096: {ps_ok}/{n} valid ({ps_ok/n*100:.0f}%), "
      f"{ps_opt}/{n} optimal, avg {ps_time/n:.1f}s")

# ── Part 2: neuro_symbolic with all 3 NL variants ──
print("\n" + "=" * 60)
print("PART 2: neuro_symbolic across NL variants (direct/descriptive/constrained)")
print("=" * 60)

for variant in ["direct", "descriptive", "constrained"]:
    ns_ok = 0
    ns_time = 0.0

    # Use 2 tasks per config
    for cfg in GRIDROUTE_CONFIGS:
        tasks = generate_gridroute_maps(
            size=cfg["size"], obstacle_size=cfg["obstacle_size"],
            num_obstacles=cfg["num_obstacles"],
            num_maps=2, pairs_per_map=1, seed=42,
        )

        for task in tasks:
            t0 = time.time()
            res = neuro_symbolic_plan(env, task.grid, task.nl_variants[variant])
            t = time.time() - t0
            ns_time += t

            path_ok = False
            if res.path and len(res.path) >= 2:
                path_ok = (_is_collision_free(res.path, task.grid) and
                          res.path[0] == task.start and
                          res.path[-1] == task.goal)
                if path_ok:
                    ns_ok += 1

    print(f"  variant='{variant}': {ns_ok}/6 valid ({ns_ok/6*100:.0f}%), avg {ns_time/6:.1f}s")
