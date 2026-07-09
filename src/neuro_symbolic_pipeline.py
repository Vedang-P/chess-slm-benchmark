"""Neuro-Symbolic Pathfinding Pipeline.

Main pipeline:
1. Gemma 4 E2B parses NL instructions -> structured JSON constraints
2. A* solver computes optimal path
3. Gemma 4 handles replanning on constraint updates
"""

import json
import time
from typing import Tuple, List, Optional, Dict
import numpy as np

from .gemma4_env import Gemma4Env
from .astar_solver import astar, astar_with_replan
from .prompts import NEURO_SYMBOLIC_SYSTEM_PROMPT, PATHFINDING_TOOL, REPLANNER_SYSTEM_PROMPT


def extract_constraints(
    gemma: Gemma4Env,
    nl_instruction: str,
) -> Dict:
    """Use Gemma 4 function calling to extract structured constraints from NL.

    Returns: {"start": [x,y], "goal": [x,y], "obstacles": [[x1,y1],...], "constraints": str}
    """
    messages = [
        {"role": "system", "content": NEURO_SYMBOLIC_SYSTEM_PROMPT},
        {"role": "user", "content": nl_instruction},
    ]

    result = gemma.generate(
        messages,
        tools=[PATHFINDING_TOOL],
        max_tokens=256,
    )

    tool_calls = result.get("tool_calls", [])
    if tool_calls:
        func = tool_calls[0].get("function", {})
        return json.loads(func.get("arguments", "{}"))

    content = result.get("content", "")
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"start": None, "goal": None, "obstacles": [], "constraints": "parse_error"}


def neuro_symbolic_plan(
    gemma: Gemma4Env,
    grid: np.ndarray,
    nl_instruction: str,
) -> Tuple[Optional[List[Tuple[int, int]]], Dict, float, int]:
    """Full neuro-symbolic pipeline: NL -> JSON -> A* -> path.

    Returns: (path, constraints_dict, latency_ms, tokens_generated)
    """
    t0 = time.time()

    constraints = extract_constraints(gemma, nl_instruction)

    start = tuple(constraints.get("start", (None, None)))
    goal = tuple(constraints.get("goal", (None, None)))
    parsed_obstacles = constraints.get("obstacles", [])

    if None in start or None in goal:
        return None, constraints, (time.time() - t0) * 1000, 0

    path = astar(grid, start, goal)

    latency = (time.time() - t0) * 1000
    tokens = 0  # TODO: track tokens from gemma.generate

    return (path if path else None), constraints, latency, tokens


def neuro_symbolic_replan(
    gemma: Gemma4Env,
    grid: np.ndarray,
    current_position: Tuple[int, int],
    goal: Tuple[int, int],
    nl_update: str,
) -> Tuple[Optional[List[Tuple[int, int]]], Dict, float]:
    """Handle replanning when user updates constraints.

    Returns: (new_path, updated_constraints_dict, latency_ms)
    """
    t0 = time.time()

    messages = [
        {"role": "system", "content": REPLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Navigation update: {nl_update}"},
    ]

    result = gemma.generate(messages, tools=[PATHFINDING_TOOL], max_tokens=256)
    tool_calls = result.get("tool_calls", [])
    if tool_calls:
        func = tool_calls[0].get("function", {})
        constraints = json.loads(func.get("arguments", "{}"))
    else:
        content = result.get("content", "")
        try:
            constraints = json.loads(content)
        except json.JSONDecodeError:
            constraints = {}

    new_obstacles = constraints.get("obstacles", [])
    if new_obstacles:
        modified_grid = grid.copy()
        for x, y in new_obstacles:
            if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:
                modified_grid[y, x] = 1
    else:
        modified_grid = grid

    path = astar(modified_grid, current_position, goal)
    latency = (time.time() - t0) * 1000

    return (path if path else None), constraints, latency


# Test
if __name__ == "__main__":
    from grid_generator import generate_gridroute_maps
    tasks = generate_gridroute_maps(size=10, obstacle_size=3, num_obstacles=2, num_maps=1, pairs_per_map=1)

    t = tasks[0]
    nl_instruction = (
        f"Navigate from position ({t.start[0]}, {t.start[1]}) "
        f"to the goal at ({t.goal[0]}, {t.goal[1]}). "
        f"The grid is {t.grid.size}x{t.grid.size}. "
        f"Avoid the obstacles at the predefined locations."
    )

    print(f"NL Instruction: {nl_instruction}")
    print(f"Optimal path length: {t.optimal_length}")

    env = Gemma4Env(load_in_4bit=True, enable_thinking=True)
    env.load()

    path, constraints, latency, tokens = neuro_symbolic_plan(env, t.grid.grid, nl_instruction)
    print(f"Extracted constraints: {constraints}")
    print(f"Path length: {len(path)-1 if path else 'None'}")
    print(f"Latency: {latency:.0f}ms")
