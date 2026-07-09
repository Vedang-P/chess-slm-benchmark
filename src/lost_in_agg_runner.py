"""Lost in Aggregation benchmark runner.

Evaluates all methods on the Lost in Aggregation maze dataset.
Metrics: SR (Success Rate), VMR (Valid Move Ratio)
"""

import json
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np

from .gemma4_env import Gemma4Env
from .baselines import pure_slm_planner, pure_astar_baseline
from .neuro_symbolic_pipeline import neuro_symbolic_plan
from .grid_generator import grid_to_nl_variants
from .evaluation import PathResult, compute_metrics, print_report


DATA_DIR = Path("data/lost_in_aggregation")
MAZE_SIZES = [3, 5, 7, 10, 15]


def load_lost_in_agg_mazes(size: int, data_dir: Path = DATA_DIR) -> list:
    """Load Lost in Aggregation mazes for a given size."""
    path = data_dir / f"size_{size}_150.json"
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return []
    with open(path) as f:
        mazes = json.load(f)
    return mazes


def maze_to_grid(maze: dict) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """Convert Lost in Aggregation maze to grid, start, goal.

    Maze format: grid[row][col], start=[row,col], goal=[row,col]
    Our format: grid[y][x], start=(x,y), goal=(x,y)
    """
    grid_data = maze["grid"]
    grid = np.array(grid_data, dtype=np.int8)
    start_rc = maze["start"]
    goal_rc = maze["goal"]
    start = (start_rc[1], start_rc[0])
    goal = (goal_rc[1], goal_rc[0])
    return grid, start, goal


def generate_nl_instruction(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> str:
    """Generate NL instruction for a maze."""
    h, w = grid.shape
    wall_cells = [(x, y) for y in range(h) for x in range(w) if grid[y, x] == 1]
    wall_str = ", ".join([f"({x},{y})" for x, y in wall_cells[:50]])
    if len(wall_cells) > 50:
        wall_str += f" ... and {len(wall_cells) - 50} more walls"
    return (
        f"Navigate from ({start[0]},{start[1]}) to ({goal[0]},{goal[1]}) "
        f"through a {w}x{h} maze. "
        f"Passable cells are 0, walls (cells with value 1) at [{wall_str}] are blocked. "
        f"Move only up, down, left, right. "
        f"Return the full path as coordinate pairs."
    )


def run_lost_in_agg_benchmark(
    gemma: Gemma4Env,
    methods: List[str] = None,
    sizes: List[int] = None,
) -> Dict:
    """Run benchmark on Lost in Aggregation mazes."""
    if methods is None:
        methods = ["pure_slm", "neuro_symbolic"]
    if sizes is None:
        sizes = MAZE_SIZES

    all_results = {}

    for size in sizes:
        mazes = load_lost_in_agg_mazes(size)
        if not mazes:
            continue

        print(f"\n{'='*60}")
        print(f"Lost in Aggregation: size {size}x{size} ({len(mazes)} mazes)")
        print(f"{'='*60}")

        batch_results: Dict[str, List[PathResult]] = {m: [] for m in methods}

        for i, maze in enumerate(mazes):
            grid, start, goal = maze_to_grid(maze)
            nl = generate_nl_instruction(grid, start, goal)

            for method in methods:
                try:
                    if method == "pure_slm":
                        result = pure_slm_planner(gemma, grid, start, goal, nl)
                    elif method == "neuro_symbolic":
                        result = neuro_symbolic_plan(gemma, grid, nl)
                    elif method == "pure_astar":
                        result = pure_astar_baseline(grid, start, goal)
                    else:
                        continue

                    true_path = pure_astar_baseline(grid, start, goal)
                    result.optimal_path = true_path.path if true_path.path else [start, goal]
                    result.optimal_length = len(result.optimal_path) - 1
                    batch_results[method].append(result)
                except Exception as e:
                    batch_results[method].append(PathResult(
                        path=None,
                        optimal_path=[start, goal],
                        optimal_length=0,
                        failure_type="runtime_error",
                    ))

            if (i + 1) % 25 == 0:
                print(f"  Progress: {i+1}/{len(mazes)}")

        label = f"size_{size}"
        all_results[label] = {}
        for method in methods:
            if batch_results[method]:
                if isinstance(batch_results[method][0].optimal_path, list):
                    r = compute_metrics(batch_results[method], grid)
                else:
                    r = compute_metrics(batch_results[method], grid)
                all_results[label][method] = r.__dict__
                print_report(r, f"{label} - {method}")

    return all_results


if __name__ == "__main__":
    env = Gemma4Env(load_in_4bit=True, enable_thinking=True)
    env.load()
    results = run_lost_in_agg_benchmark(env)
    output_dir = Path("data/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "lost_in_agg_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
