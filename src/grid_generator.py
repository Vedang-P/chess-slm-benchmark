"""Grid world environment and dataset generation.

Implements GridRoute (Li et al., 2025) compatible grid generation:
- NxN grids with square obstacles
- 4-directional cardinal movement
- Start/end pairs with minimum distance constraints
- Dijkstra reference paths for ground truth

Also generates mazes compatible with Lost in Aggregation (Jiang et al., 2026):
- Tree-structured mazes via DFS randomized generation
- Per-cell topology annotation (corridor, junction, dead-end)
- Shortest unique path computation
"""

import numpy as np
import json
import heapq
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


@dataclass
class GridMap:
    """A single grid map with obstacles."""
    size: int
    obstacles: List[Tuple[int, int, int, int]]  # (x, y, w, h) for each obstacle
    grid: np.ndarray  # 0 = free, 1 = obstacle


@dataclass
class GridRouteTask:
    """A single pathfinding task."""
    task_id: str
    grid: np.ndarray
    size: int
    start: Tuple[int, int]
    goal: Tuple[int, int]
    obstacles_rect: List[Tuple[int, int, int, int]]
    optimal_path: List[Tuple[int, int]] = field(default_factory=list)
    optimal_length: int = 0
    nl_variants: Dict[str, str] = field(default_factory=dict)


DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 4-directional
DIR_NAMES = ["right", "left", "down", "up"]


GRIDROUTE_CONFIGS = [
    {"size": 10, "obstacle_size": 3, "num_obstacles": 2, "label": "size_10"},
    {"size": 20, "obstacle_size": 4, "num_obstacles": 3, "label": "size_20"},
    {"size": 30, "obstacle_size": 5, "num_obstacles": 4, "label": "size_30"},
]


def generate_gridroute_maps(
    size: int = 10,
    obstacle_size: int = 3,
    num_obstacles: int = 2,
    num_maps: int = 100,
    pairs_per_map: int = 5,
    seed: int = 42,
) -> List[GridRouteTask]:
    """Generate GridRoute-style benchmark data.

    Configs (from paper):
    - {N=10, n=2, s=3}, {N=20, n=3, s=4}, {N=30, n=4, s=5}
    - Min Euclidean distance: 30% of grid diagonal
    - Verified connectivity via BFS
    """
    rng = np.random.RandomState(seed)
    tasks = []
    min_dist = 0.3 * np.sqrt(2 * size**2)
    task_counter = 0

    for map_idx in range(num_maps):
        grid = np.zeros((size, size), dtype=np.int8)
        obstacles_rect = []
        for _ in range(num_obstacles):
            x = rng.randint(0, size - obstacle_size)
            y = rng.randint(0, size - obstacle_size)
            obstacles_rect.append((x, y, obstacle_size, obstacle_size))
            grid[y:y+obstacle_size, x:x+obstacle_size] = 1

        for _ in range(pairs_per_map):
            attempts = 0
            while True:
                start = (rng.randint(0, size), rng.randint(0, size))
                goal = (rng.randint(0, size), rng.randint(0, size))
                if (grid[start[1], start[0]] == 0 and
                    grid[goal[1], goal[0]] == 0 and
                    np.linalg.norm(np.array(start) - np.array(goal)) >= min_dist and
                    bfs_connected(grid, start, goal)):
                    break
                attempts += 1
                if attempts > 100:
                    break

            path = dijkstra(grid, start, goal)
            task_id = f"GRID_{size}x{size}_m{map_idx:03d}_p{_}"
            nl_vars = grid_to_nl_variants(grid, start, goal)
            tasks.append(GridRouteTask(
                task_id=task_id,
                grid=grid.copy(),
                size=size,
                start=start,
                goal=goal,
                obstacles_rect=obstacles_rect,
                optimal_path=path,
                optimal_length=len(path) - 1 if path else -1,
                nl_variants=nl_vars,
            ))
            task_counter += 1

    return tasks


def gridroute_defaults(grid_size: int) -> Tuple[int, int]:
    """(obstacle_size, num_obstacles) scaled for a given grid size -- shared by
    every training/eval script so the auto-scaling rule lives in exactly one
    place instead of being copy-pasted (and able to silently drift) across
    train_sft.py, train_grpo.py and eval.py."""
    return (1, 1) if grid_size == 5 else (3, 2)


def generate_all_gridroute(seed: int = 42) -> List[GridRouteTask]:
    """Generate all 3 GridRoute configs (1,500 total tasks)."""
    all_tasks = []
    for cfg in GRIDROUTE_CONFIGS:
        tasks = generate_gridroute_maps(
            size=cfg["size"],
            obstacle_size=cfg["obstacle_size"],
            num_obstacles=cfg["num_obstacles"],
            num_maps=100,
            pairs_per_map=5,
            seed=seed,
        )
        all_tasks.extend(tasks)
        print(f"  {cfg['label']}: {len(tasks)} tasks")
    return all_tasks


def bfs_connected(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
    """Check if start and goal are connected via BFS."""
    from collections import deque
    h, w = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    q = deque([start])
    visited[start[1], start[0]] = True
    while q:
        x, y = q.popleft()
        if (x, y) == goal:
            return True
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx] and grid[ny, nx] == 0:
                visited[ny, nx] = True
                q.append((nx, ny))
    return False


def dijkstra(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Dijkstra shortest path with 4-directional movement."""
    h, w = grid.shape
    dist = {start: 0}
    prev = {}
    pq = [(0, start)]
    visited = set()

    while pq:
        d, (x, y) = heapq.heappop(pq)
        if (x, y) in visited:
            continue
        visited.add((x, y))
        if (x, y) == goal:
            break
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and grid[ny, nx] == 0:
                nd = d + 1
                if (nx, ny) not in dist or nd < dist[(nx, ny)]:
                    dist[(nx, ny)] = nd
                    prev[(nx, ny)] = (x, y)
                    heapq.heappush(pq, (nd, (nx, ny)))

    if goal not in prev:
        return []

    path = [goal]
    cur = goal
    while cur in prev:
        cur = prev[cur]
        path.append(cur)
    return path[::-1]


def grid_to_text(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> str:
    """Convert grid to natural language description for LLM input."""
    h, w = grid.shape
    obstacles = [(x, y) for y in range(h) for x in range(w) if grid[y, x] == 1]
    desc = f"The environment is a {w}x{h} grid. "
    desc += f"Start at ({start[0]}, {start[1]}). Goal at ({goal[0]}, {goal[1]}). "
    if obstacles:
        desc += f"Obstacles occupy cells: {obstacles}. "
    desc += "Movement is in 4 cardinal directions (up, down, left, right) with unit steps."
    return desc


# The ONE shared instruction for how a model should report its GridRoute NL
# answer -- imported by eval.py, train_sft.py, and train_grpo.py so SFT
# training, GRPO training, and evaluation all teach/expect the exact same
# reporting convention. An earlier version of this project had SFT teach one
# phrasing ("Output the path as a Python list...") while GRPO/eval expected a
# different one ("FINAL ANSWER:"), which undermines the whole point of asking
# for a clean, parseable final answer -- the model was never actually trained
# on the convention being scored against.
GRIDROUTE_NL_ANSWER_SUFFIX = (
    "\n\nThink step by step about the path. After your thinking, output your final answer "
    "on a new line starting with exactly 'FINAL ANSWER:' followed by a Python list of "
    "(row,col) tuples, like: FINAL ANSWER: [(0,0), (0,1), (1,1)]"
)


def grid_to_nl_variants(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Dict[str, str]:
    """Generate 3 NL instruction variants for robustness testing."""
    h, w = grid.shape
    obstacles = [(x, y) for y in range(h) for x in range(w) if grid[y, x] == 1]
    obs_str = ", ".join([f"({x},{y})" for x, y in obstacles]) if obstacles else "none"

    a_direct = {
        "direct": (
            f"Navigate from ({start[0]},{start[1]}) to ({goal[0]},{goal[1]}) "
            f"on a {w}x{h} grid with obstacles at [{obs_str}]. Move up/down/left/right."
        ),
    }
    b_descriptive = {
        "descriptive": (
            f"Find a path from the start marker at position ({start[0]},{start[1]}) "
            f"to the goal at ({goal[0]},{goal[1]}) through the {w}x{h} grid world. "
            f"Cells marked as obstacles at [{obs_str}] cannot be entered. "
            f"You can only move in cardinal directions."
        ),
    }
    c_constrained = {
        "constrained": (
            f"Plan the shortest route from A=({start[0]},{start[1]}) to B=({goal[0]},{goal[1]}) "
            f"in a {w}x{h} environment. Blocked cells: [{obs_str}]. "
            f"Allowed moves: up, down, left, right (unit steps only). "
            f"Return the full path coordinates."
        ),
    }
    return {**a_direct, **b_descriptive, **c_constrained}


def generate_maze(size: int, seed: int = 42) -> np.ndarray:
    """Generate a tree-structured maze via randomized DFS.

    Returns grid where 0 = path, 1 = wall.
    Compatible with Lost in Aggregation format.
    """
    rng = np.random.RandomState(seed)
    grid = np.ones((size, size), dtype=np.int8)
    visited = np.zeros((size, size), dtype=bool)

    def carve(x, y):
        visited[y, x] = True
        grid[y, x] = 0
        dirs = DIRECTIONS.copy()
        rng.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = x + 2*dx, y + 2*dy
            if 0 <= nx < size and 0 <= ny < size and not visited[ny, nx]:
                grid[y + dy, x + dx] = 0
                carve(nx, ny)

    carve(1, 1)
    return grid


# Test
if __name__ == "__main__":
    tasks = generate_gridroute_maps(size=10, obstacle_size=3, num_obstacles=2, num_maps=1, pairs_per_map=3)
    for t in tasks:
        desc = grid_to_text(t.grid.grid, t.start, t.goal)
        print(desc)
        print(f"  Optimal path length: {t.optimal_length}")
