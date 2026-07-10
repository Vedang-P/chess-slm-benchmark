"""Neuro-Symbolic Pathfinding Pipeline.

Main pipeline:
1. Gemma 4 E2B parses NL instructions -> structured JSON constraints
2. A* solver computes optimal path
3. Gemma 4 handles replanning on constraint updates
"""

import json
import re
import time
from typing import Tuple, List, Optional, Dict, Any
import numpy as np

from .gemma4_env import Gemma4Env
from .astar_solver import astar, astar_with_replan
from .prompts import NEURO_SYMBOLIC_SYSTEM_PROMPT, PATHFINDING_TOOL, REPLANNER_SYSTEM_PROMPT
from .evaluation import PathResult


def _extract_json_from_text(text: str) -> Optional[Dict]:
    """Extract JSON object from text, trying multiple strategies."""
    # Strategy 1: Find complete JSON object
    brace_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass

    # Strategy 2: Find coordinate patterns like start: (x,y) or goal: [x,y]
    coords = {"start": None, "goal": None, "obstacles": []}
    for key, patterns in [
        ("start", [r"start[:\s]*\(?(\d+)[,\s]+(\d+)\)?", r"from[:\s]*\(?(\d+)[,\s]+(\d+)\)?"]),
        ("goal", [r"goal[:\s]*\(?(\d+)[,\s]+(\d+)\)?", r"to[:\s]*\(?(\d+)[,\s]+(\d+)\)?"]),
    ]:
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                coords[key] = [int(m.group(1)), int(m.group(2))]
                break

    # Strategy 3: Match coordinates from navigation instruction
    inst_match = re.search(r"Navigate from\s*\(?(\d+),\s*(\d+)\)?\s*to\s*\(?(\d+),\s*(\d+)\)?", text, re.IGNORECASE)
    if inst_match:
        coords["start"] = [int(inst_match.group(1)), int(inst_match.group(2))]
        coords["goal"] = [int(inst_match.group(3)), int(inst_match.group(4))]

    # Extract obstacles
    obs_match = re.findall(r"obstacle[:\s]*\[([^\]]+)\]", text, re.IGNORECASE)
    if obs_match:
        try:
            obs_list = json.loads("[" + obs_match[0] + "]")
            coords["obstacles"] = obs_list
        except (json.JSONDecodeError, TypeError):
            pass

    if coords["start"] and coords["goal"]:
        coords["obstacles"] = coords.get("obstacles", [])
        return coords
    return None


def extract_constraints(
    gemma: Gemma4Env,
    nl_instruction: str,
    max_retries: int = 2,
) -> Tuple[Dict, int, float]:
    """Extract structured constraints from NL instruction.

    Uses plain-text prompt (no function calling) for Gemma 4 compatibility.
    Falls back to regex extraction from thinking text.

    Returns: (constraints_dict, tokens_used, latency_ms)
    """
    t0 = time.time()
    system = NEURO_SYMBOLIC_SYSTEM_PROMPT
    user = f"Instruction: {nl_instruction}\n\nExtract start and goal coordinates as JSON: {{\"start\": [x, y], \"goal\": [x, y]}}"

    for attempt in range(max_retries):
        result = gemma.generate(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=1024,
        )
        tokens = (result.get("input_tokens", 0) + result.get("output_tokens", 0)
                  if isinstance(result, dict) else 0)
        content = result.get("content", "")

        # Try JSON extraction from response
        constraints = _extract_json_from_text(content)
        if constraints and constraints.get("start") and constraints.get("goal"):
            latency = (time.time() - t0) * 1000
            return constraints, tokens, latency

    latency = (time.time() - t0) * 1000
    return {"start": None, "goal": None, "obstacles": [], "constraints": "parse_error"}, tokens, latency


def neuro_symbolic_plan(
    gemma: Gemma4Env,
    grid: np.ndarray,
    nl_instruction: str,
) -> PathResult:
    """Full neuro-symbolic pipeline: NL -> JSON -> A* -> path."""
    t0 = time.time()

    constraints, tokens, parse_latency = extract_constraints(gemma, nl_instruction)

    start = tuple(constraints.get("start", (None, None)))
    goal = tuple(constraints.get("goal", (None, None)))

    extraction_ok = (
        start[0] is not None and start[1] is not None and
        goal[0] is not None and goal[1] is not None
    )

    if not extraction_ok:
        return PathResult(
            path=None,
            optimal_path=[(0, 0), (0, 1)],
            optimal_length=0,
            tokens_generated=tokens,
            latency_ms=parse_latency,
            extraction_ok=False,
            failure_type="parse_error",
            raw_output=str(constraints),
        )

    if not (0 <= start[0] < grid.shape[1] and 0 <= start[1] < grid.shape[0] and
            0 <= goal[0] < grid.shape[1] and 0 <= goal[1] < grid.shape[0]):
        return PathResult(
            path=None,
            optimal_path=[start, goal],
            optimal_length=0,
            tokens_generated=tokens,
            latency_ms=parse_latency,
            extraction_ok=True,
            failure_type="bounds_error",
            raw_output=str(constraints),
        )

    if grid[start[1], start[0]] != 0 or grid[goal[1], goal[0]] != 0:
        return PathResult(
            path=None,
            optimal_path=[start, goal],
            optimal_length=0,
            tokens_generated=tokens,
            latency_ms=parse_latency,
            extraction_ok=True,
            failure_type="obstacle_start_goal",
            raw_output=str(constraints),
        )

    path = astar(grid, start, goal)
    latency = (time.time() - t0) * 1000

    if not path:
        return PathResult(
            path=None,
            optimal_path=[start, goal],
            optimal_length=0,
            tokens_generated=tokens,
            latency_ms=latency,
            extraction_ok=True,
            failure_type="no_path",
            raw_output=str(constraints),
        )

    return PathResult(
        path=path,
        optimal_path=[start, goal],
        optimal_length=len(path) - 1,
        tokens_generated=tokens,
        latency_ms=latency,
        extraction_ok=True,
        raw_output=str(constraints),
    )


def neuro_symbolic_replan(
    gemma: Gemma4Env,
    grid: np.ndarray,
    current_position: Tuple[int, int],
    goal: Tuple[int, int],
    nl_update: str,
) -> PathResult:
    """Handle replanning when user updates constraints."""
    t0 = time.time()

    messages = [
        {"role": "system", "content": REPLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Navigation update: {nl_update}"},
    ]

    result = gemma.generate(messages, tools=[PATHFINDING_TOOL], max_tokens=256)
    tokens = result.get("input_tokens", 0) + result.get("output_tokens", 0) \
        if isinstance(result, dict) else 0

    tool_calls = result.get("tool_calls", [])
    if tool_calls:
        try:
            func = tool_calls[0].get("function", {})
            constraints = json.loads(func.get("arguments", "{}"))
        except (json.JSONDecodeError, KeyError, IndexError):
            constraints = {}
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

    return PathResult(
        path=path if path else None,
        optimal_path=[current_position, goal],
        optimal_length=len(path) - 1 if path else 0,
        tokens_generated=tokens,
        latency_ms=latency,
        raw_output=str(constraints),
    )
