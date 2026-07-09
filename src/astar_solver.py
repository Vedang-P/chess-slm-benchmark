"""A* pathfinding solver for grid environments.

Standard A* implementation with:
- 4-directional cardinal movement
- Manhattan heuristic
- Octile heuristic option (for compatibility)
"""

import heapq
from typing import List, Tuple, Optional, Callable
import numpy as np


DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def octile(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + (np.sqrt(2) - 1) * min(dx, dy)


def astar(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    heuristic: Callable = manhattan,
) -> List[Tuple[int, int]]:
    """A* shortest path on grid (0=free, 1=obstacle).

    Returns list of (x,y) coordinates from start to goal inclusive.
    Returns empty list if no path exists.
    """
    h, w = grid.shape
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    came_from = {}
    open_set = [(f_score[start], start)]
    closed_set = set()

    while open_set:
        _, current = heapq.heappop(open_set)
        if current in closed_set:
            continue
        closed_set.add(current)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for dx, dy in DIRECTIONS:
            nx, ny = current[0] + dx, current[1] + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if grid[ny, nx] != 0:
                continue
            neighbor = (nx, ny)
            tentative_g = g_score[current] + 1

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []


def astar_with_replan(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    updated_obstacles: List[Tuple[int, int]] = None,
    heuristic: Callable = manhattan,
) -> Tuple[List[Tuple[int, int]], bool]:
    """A* with dynamic replanning.

    If updated_obstacles is provided, temporarily marks those cells
    as blocked and re-runs A*. Returns (path, was_replanned).
    """
    if updated_obstacles is None:
        return astar(grid, start, goal, heuristic), False

    modified_grid = grid.copy()
    for x, y in updated_obstacles:
        if 0 <= x < modified_grid.shape[1] and 0 <= y < modified_grid.shape[0]:
            modified_grid[y, x] = 1

    return astar(modified_grid, start, goal, heuristic), True


# Test
if __name__ == "__main__":
    from grid_generator import generate_gridroute_maps
    tasks = generate_gridroute_maps(size=10, obstacle_size=3, num_obstacles=2, num_maps=1, pairs_per_map=1)
    t = tasks[0]
    path = astar(t.grid.grid, t.start, t.goal)
    print(f"A* path length: {len(path)-1 if path else -1}")
    print(f"Dijkstra optimal: {t.optimal_length}")
    print(f"Match: {len(path)-1 == t.optimal_length}")
