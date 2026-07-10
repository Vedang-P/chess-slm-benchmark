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
    obstacles = [(int(x), int(y)) for x in range(grid.shape[1])
                 for y in range(grid.shape[0]) if grid[y, x] == 1]
    prompt = PURE_SLM_PROMPT_TEMPLATE.format(
        start=start, goal=goal, obstacles=obstacles)

    best_path = None
    total_tokens = 0
    raw_output = ""

    for attempt in range(max_retries):
        messages = [{"role": "user", "content": prompt}]
        result = gemma.generate(messages, max_tokens=2048)
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
    """Parse LLM output to extract a valid path.

    Tries multiple strategies:
    1. Extract coordinates from text and find a valid start→goal sequence
    2. Extract from formatted lists
    Returns None if parsing fails.
    """
    path = _extract_coords_from_text(content)
    if not path or len(path) < 2:
        return None

    # Strategy 1: Direct match (first is start, last is goal)
    if path[0] == start and path[-1] == goal:
        return _filter_valid_path(path, grid)

    # Strategy 2: Trim obstacles from start/end
    if _norm(path[0], start) <= 1 and _norm(path[-1], goal) <= 1:
        path[0] = start
        path[-1] = goal
        return _filter_valid_path(path, grid)

    # Strategy 3: Find the longest contiguous start→goal subsequence
    best = None
    for i, p in enumerate(path):
        if p != start:
            continue
        for j in range(len(path) - 1, i, -1):
            if path[j] != goal:
                continue
            candidate = path[i:j + 1]
            valid = _filter_valid_path(candidate, grid)
            if valid and (best is None or len(valid) > len(best)):
                best = valid

    return best


def _filter_valid_path(
    path: List[Tuple[int, int]],
    grid: np.ndarray,
) -> Optional[List[Tuple[int, int]]]:
    """Filter a path to remove invalid steps and obstacles."""
    if not path or len(path) < 2:
        return None
    valid = [path[0]]
    for p in path[1:]:
        last = valid[-1]
        # Check adjacency (4-directional)
        if abs(p[0] - last[0]) + abs(p[1] - last[1]) == 1:
            # Check not on obstacle
            if 0 <= p[0] < grid.shape[1] and 0 <= p[1] < grid.shape[0] and grid[p[1], p[0]] == 0:
                valid.append(p)
        # Allow skipping back to a non-adjacent coord if it's more direct
        elif p == path[-1] and len(valid) > 1:
            pass  # Skip non-adjacent intermediate coords
    if len(valid) >= 2 and valid[0] == path[0] and valid[-1] == path[-1]:
        return valid
    return None


def _norm(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _extract_coords_from_text(text: str) -> List[Tuple[int, int]]:
    markers = ["**Path:**", "Path:", "path:", "Output:", "output:", "->", "\n\n"]
    # Split at the last Path: marker if present
    seg = text
    for m in markers:
        idx = text.rfind(m)
        if idx >= 0:
            seg = text[idx:]
            break

    patterns = [
        r'\((\d+),\s*(\d+)\)',
        r'\[(\d+),\s*(\d+)\]',
        r'\((\d+)\s+(\d+)\)',
    ]
    all_coords = []
    for pat in patterns:
        matches = re.findall(pat, seg)
        for x, y in matches:
            coord = (int(x), int(y))
            all_coords.append(coord)
    return all_coords
