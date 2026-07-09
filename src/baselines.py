"""Baselines for pathfinding comparison.

Implements:
1. Pure SLM Planner - Gemma 4 generates path directly
2. AoP (Algorithm of Planning) - Gemma 4 with GridRoute AoP/A* prompt
3. Pure A* - Classical solver (ground truth upper bound)
"""

import re
import time
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
from .evaluation import PathResult


def pure_slm_planner(
    gemma: Gemma4Env,
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    nl_instruction: str = None,
    max_retries: int = 3,
) -> PathResult:
    """LLM generates the path directly (no algorithm)."""
    t0 = time.time()
    if nl_instruction is None:
        env_desc = grid_to_text(grid, start, goal)
    else:
        env_desc = nl_instruction
    prompt = PURE_SLM_PROMPT_TEMPLATE.format(env=env_desc)

    best_path = None
    total_tokens = 0
    raw_output = ""

    for attempt in range(max_retries):
        messages = [{"role": "user", "content": prompt}]
        result = gemma.generate(messages, max_tokens=1024)
        content = result.get("content", "") if isinstance(result, dict) else str(result)
        raw_output = content
        total_tokens += result.get("input_tokens", 0) + result.get("output_tokens", 0) \
            if isinstance(result, dict) else 0

        path = _parse_path_response(content, grid, start, goal)
        if path is not None:
            best_path = path
            break

    latency = (time.time() - t0) * 1000
    return PathResult(
        path=best_path,
        optimal_path=[start, goal],
        optimal_length=0,
        tokens_generated=total_tokens,
        latency_ms=latency,
        raw_output=raw_output,
    )


def aop_planner(
    gemma: Gemma4Env,
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    algorithm: str = "astar",
    max_retries: int = 3,
) -> PathResult:
    """LLM with Algorithm of Planning (AoP) prompt."""
    t0 = time.time()
    env_desc = grid_to_text(grid, start, goal)
    template = AOP_ASTAR_PROMPT_TEMPLATE if algorithm == "astar" else AOP_DIJKSTRA_PROMPT_TEMPLATE
    prompt = template.format(env=env_desc)

    best_path = None
    total_tokens = 0
    raw_output = ""

    for attempt in range(max_retries):
        messages = [{"role": "user", "content": prompt}]
        result = gemma.generate(messages, max_tokens=1024)
        content = result.get("content", "") if isinstance(result, dict) else str(result)
        raw_output = content
        total_tokens += result.get("input_tokens", 0) + result.get("output_tokens", 0) \
            if isinstance(result, dict) else 0

        path = _parse_path_response(content, grid, start, goal)
        if path is not None:
            best_path = path
            break

    latency = (time.time() - t0) * 1000
    return PathResult(
        path=best_path,
        optimal_path=[start, goal],
        optimal_length=0,
        tokens_generated=total_tokens,
        latency_ms=latency,
        raw_output=raw_output,
    )


def pure_astar_baseline(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> PathResult:
    """Classical A* solver (optimal upper bound)."""
    t0 = time.time()
    path = astar(grid, start, goal)
    latency = (time.time() - t0) * 1000
    return PathResult(
        path=path,
        optimal_path=[start, goal],
        optimal_length=len(path) - 1 if path else 0,
        latency_ms=latency,
    )


def _parse_path_response(
    content: str,
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    """Parse LLM output to extract path coordinates.

    Tries multiple parsing strategies:
    1. Extract coordinate tuples from text
    2. Extract from formatted lists
    Returns None if parsing fails.
    """
    path = _extract_coords_from_text(content)
    if path and len(path) >= 2:
        if path[0] == start and path[-1] == goal:
            return path
        if _norm(path[0], start) <= 1 and _norm(path[-1], goal) <= 1:
            path[0] = start
            path[-1] = goal
            return path
    return None


def _norm(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _extract_coords_from_text(text: str) -> List[Tuple[int, int]]:
    patterns = [
        r'\((\d+),\s*(\d+)\)',
        r'\[(\d+),\s*(\d+)\]',
        r'\((\d+)\s+(\d+)\)',
    ]
    all_coords = []
    seen = set()
    for pat in patterns:
        matches = re.findall(pat, text)
        for x, y in matches:
            coord = (int(x), int(y))
            if coord not in seen:
                seen.add(coord)
                all_coords.append(coord)
    return all_coords
