"""Baselines for pathfinding comparison.

Implements:
1. Pure SLM Planner - Gemma 4 generates path directly
2. AoP (Algorithm of Planning) - Gemma 4 with GridRoute AoP/A* prompt
3. Pure A* - Classical solver (ground truth upper bound)
"""

from typing import Tuple, List, Optional
import numpy as np
from .gemma4_env import Gemma4Env
from .grid_generator import grid_to_text
from .astar_solver import astar
from .prompts import (
    PURE_SLM_PROMPT_TEMPLATE,
    AOP_ASTAR_PROMPT_TEMPLATE,
    AOP_DIJKSTRA_PROMPT_TEMPLATE,
)


def pure_slm_planner(
    gemma: Gemma4Env,
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    """LLM generates the path directly (no algorithm)."""
    env_desc = grid_to_text(grid, start, goal)
    prompt = PURE_SLM_PROMPT_TEMPLATE.format(env=env_desc)

    messages = [{"role": "user", "content": prompt}]
    result = gemma.generate(messages, max_tokens=1024)

    return _parse_path_response(result, grid, start, goal)


def aop_planner(
    gemma: Gemma4Env,
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    algorithm: str = "astar",
) -> Optional[List[Tuple[int, int]]]:
    """LLM with Algorithm of Planning (AoP) prompt."""
    env_desc = grid_to_text(grid, start, goal)
    template = AOP_ASTAR_PROMPT_TEMPLATE if algorithm == "astar" else AOP_DIJKSTRA_PROMPT_TEMPLATE
    prompt = template.format(env=env_desc)

    messages = [{"role": "user", "content": prompt}]
    result = gemma.generate(messages, max_tokens=1024)

    return _parse_path_response(result, grid, start, goal)


def pure_astar_baseline(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """Classical A* solver (optimal upper bound)."""
    return astar(grid, start, goal)


def _parse_path_response(result: dict, grid: np.ndarray,
                         start: Tuple[int, int], goal: Tuple[int, int]
                         ) -> Optional[List[Tuple[int, int]]]:
    """Parse LLM output to extract path coordinates.

    Returns None if parsing fails (non-compliant output).
    """
    content = result.get("content", "") if isinstance(result, dict) else str(result)
    h, w = grid.shape
    try:
        path = []
        pairs = content.replace("[", "").replace("]", "").replace("(", "").replace(")", "").split(",")
        i = 0
        while i < len(pairs) - 1:
            try:
                x = int(pairs[i].strip())
                y = int(pairs[i+1].strip())
                path.append((x, y))
            except (ValueError, IndexError):
                pass
            i += 2

        if len(path) < 2:
            return None
        if path[0] != start or path[-1] != goal:
            return None
        return path
    except Exception:
        return None
