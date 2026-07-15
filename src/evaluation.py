"""Path-validity primitives shared by eval.py and train_grpo.py's reward
function -- the ONE place that defines what counts as a valid/optimal path,
used identically for both scoring and training so the reward signal and the
reported metric are never subtly different things.
"""

from typing import List, Tuple

import numpy as np


def _is_collision_free(path: List[Tuple[int, int]], grid: np.ndarray) -> bool:
    """Check no path cell intersects an obstacle. Returns False for OOB coords.

    Requires len(path) >= 2: `all()` over a 0- or 1-element path is vacuously
    True, which previously let degenerate single-coordinate extractions (e.g.
    a stray number pulled from LaTeX the model emitted instead of the
    requested format) get graded as a valid path. See _is_valid_steps.
    """
    if path is None or len(path) < 2:
        return False
    h, w = grid.shape
    return all(0 <= y < h and 0 <= x < w and grid[y, x] == 0 for x, y in path)


def _is_in_bounds(path: List[Tuple[int, int]], shape: Tuple[int, int]) -> bool:
    """Check all path cells within grid bounds. Requires len(path) >= 2 (see _is_valid_steps)."""
    if path is None or len(path) < 2:
        return False
    h, w = shape
    return all(0 <= x < w and 0 <= y < h for x, y in path)


def _is_valid_steps(path: List[Tuple[int, int]]) -> bool:
    """Check all steps are unit-length (no diagonal moves).

    Requires len(path) >= 2. A 0- or 1-element path has no consecutive pairs
    to check, so `range(len(path) - 1)` is empty and the loop below would
    vacuously return True -- i.e. a single stray coordinate (never reaching
    the goal, not even a real path) would pass as "valid". Concretely hit in
    practice: a model emits `\\boxed{[(3,1)]}` instead of the requested
    output format, the coordinate regex still extracts `(3,1)`, and without
    this guard that 1-element list was graded a valid path.
    """
    if path is None or len(path) < 2:
        return False
    for i in range(len(path) - 1):
        dx = abs(path[i][0] - path[i+1][0])
        dy = abs(path[i][1] - path[i+1][1])
        if dx + dy != 1:
            return False
    return True
