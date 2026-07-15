"""Token-based maze representation -- the "token format" half of the cross-format
transfer experiment (the other half is GridRoute's NL format, see grid_generator.py).

This is deliberately OUR OWN fully-specified token grammar, not a byte-for-byte
reproduction of AlphaMaze/Menlo's proprietary tokenizer layout. It reuses their
published move/coordinate/wall token *vocabulary* (documented in their public
inference code) so it stays visually/textually close to what an AlphaMaze
checkpoint has already seen, but defines its own precise per-cell layout rule.
Reverse-engineering their exact undocumented row layout from a handful of
sample outputs risked subtly-wrong mazes that would confound results for
reasons unrelated to the actual research question. What the cross-format
experiment needs is two genuinely different, well-specified representations of
the *same* underlying grid that we can encode and decode correctly -- not an
exact clone of a third party's tokenizer.

Real AlphaMaze checkpoint + real Maze-Bench-v0.2 data/format are still used
as-is (loaded directly from the HF dataset, no conversion needed) for the
Phase 1 replication check -- that step exists specifically to validate the
harness against a known published number, so it must use their real artifacts
unmodified. This module is only used for Phase 2's own single/mixed/
consistency GRPO training, where both formats are ours to define.
"""

from typing import List, Optional, Tuple

import numpy as np

DIRECTION_DELTAS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}
# Fixed emission order for combining multiple blocked directions into one
# wall-token name, e.g. blocked={down, left} -> "<|down_left_wall|>".
_WALL_ORDER = ["up", "down", "left", "right"]

TASK_INSTRUCTION = (
    "You are a helpful assistant that solves mazes. You will be given a maze "
    "represented by a series of tokens, one row at a time. Each free cell shows "
    "a coordinate token <|row-col|> (e.g. <|0-0|>), optionally <|origin|> or "
    "<|target|>, and a wall token describing which sides are blocked "
    "(<|no_wall|> = open on all four sides, <|up_wall|> = blocked above, "
    "<|up_down_wall|> = blocked above and below, etc.). Blocked cells show "
    "<|blocked|>. Your task is to output the sequence of movements "
    "(<|up|>, <|down|>, <|left|>, <|right|>) required to navigate from the "
    "origin to the target. Output only the move tokens, separated by spaces."
)


def _wall_token(grid: np.ndarray, x: int, y: int) -> str:
    h, w = grid.shape
    blocked = []
    for name in _WALL_ORDER:
        dx, dy = DIRECTION_DELTAS[name]
        nx, ny = x + dx, y + dy
        if not (0 <= nx < w and 0 <= ny < h) or grid[ny, nx] != 0:
            blocked.append(name)
    if not blocked:
        return "<|no_wall|>"
    return f"<|{'_'.join(blocked)}_wall|>"


def grid_to_token_maze(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> str:
    """Encode a grid (0=free, 1=obstacle) + start/goal as a token-maze prompt body.

    Coordinates are (x, y) = (col, row) throughout this codebase, matching
    grid_generator.py / evaluation.py. Emits one line per row, one
    coordinate+[marker]+wall token triple per free cell, left to right.
    """
    h, w = grid.shape
    lines = []
    for y in range(h):
        cells = []
        for x in range(w):
            if grid[y, x] != 0:
                cells.append("<|blocked|>")
                continue
            token = f"<|{y}-{x}|>"
            if (x, y) == tuple(start):
                token += "<|origin|>"
            if (x, y) == tuple(goal):
                token += "<|target|>"
            token += _wall_token(grid, x, y)
            cells.append(token)
        lines.append("".join(cells))
    return "\n".join(lines)


def path_to_moves(path: List[Tuple[int, int]]) -> str:
    """Ground-truth move-token string for an (x,y) coordinate path, for SFT targets."""
    moves = []
    for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
        dx, dy = x1 - x0, y1 - y0
        for name, (ddx, ddy) in DIRECTION_DELTAS.items():
            if (dx, dy) == (ddx, ddy):
                moves.append(f"<|{name}|>")
                break
    return " ".join(moves)


def parse_move_tokens(text: str) -> List[str]:
    """Extract the predicted move sequence from raw model output as plain
    direction names (e.g. "up", not "<|up|>") -- a capturing group, so this
    can't accidentally match inside a wall token like <|up_down_wall|> (that
    token has no literal "|>" right after "up", so the anchored pattern
    below never matches a substring of it either way)."""
    import re
    return re.findall(r"<\|(up|down|left|right)\|>", text)


def moves_to_path(
    moves: List[str], start: Tuple[int, int], grid: np.ndarray, goal: Optional[Tuple[int, int]] = None
) -> List[Tuple[int, int]]:
    """Simulate a predicted move-token sequence from `start`, one step at a time.

    Stops at the first move that would leave the grid or enter an obstacle
    (the resulting path just ends there, mirroring a real derailed trajectory),
    or as soon as `goal` is reached (ignoring any trailing tokens after that).
    Returns the resulting (x,y) coordinate path -- feed this into
    evaluation.py's _is_in_bounds/_is_collision_free/_is_valid_steps/optimal-
    length check, the same primitives used for NL-format paths. This is what
    lets both formats share one scoring backend instead of maintaining two
    separate notions of "correct".
    """
    h, w = grid.shape
    path = [tuple(start)]
    cur = tuple(start)
    if goal is not None and cur == tuple(goal):
        return path
    for mv in moves:
        name = mv.strip("<|>")
        delta = DIRECTION_DELTAS.get(name)
        if delta is None:
            break
        nx, ny = cur[0] + delta[0], cur[1] + delta[1]
        if not (0 <= nx < w and 0 <= ny < h) or grid[ny, nx] != 0:
            break
        cur = (nx, ny)
        path.append(cur)
        if goal is not None and cur == tuple(goal):
            break
    return path
