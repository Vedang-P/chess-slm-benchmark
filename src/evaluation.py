"""Evaluation metrics for pathfinding benchmarks.

Implements all GridRoute (Li et al., 2025) metrics:
- CR: Compliance Ratio
- FR: Feasibility Ratio
- OR: Optimal Ratio
- GM: Geometric Mean
- MSE: Mean Square Error
- RT: Run Time

Plus Lost in Aggregation metrics:
- SR: Success Rate
- VMR: Valid Move Ratio

Plus custom metrics:
- Tokens: Total generated tokens
- VRAM: Peak GPU memory
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import numpy as np
from dataclasses import dataclass, field


@dataclass
class PathResult:
    """Result of a single pathfinding query."""
    path: Optional[List[Tuple[int, int]]]
    optimal_path: List[Tuple[int, int]]
    optimal_length: int
    tokens_generated: int = 0
    latency_ms: float = 0.0
    vram_peak_gb: float = 0.0
    failure_type: str = ""
    extraction_ok: bool = True
    raw_output: str = ""
    task_id: str = ""


@dataclass
class EvaluationReport:
    """Aggregate evaluation metrics."""
    compliance_ratio: float = 0.0
    feasibility_ratio: float = 0.0
    optimal_ratio: float = 0.0
    geometric_mean: float = 0.0
    mse: float = 0.0
    avg_runtime_ms: float = 0.0
    success_rate: float = 0.0
    valid_move_ratio: float = 0.0
    avg_tokens: float = 0.0
    avg_vram_gb: float = 0.0
    failure_types: dict = field(default_factory=dict)
    num_tasks: int = 0
    num_compliant: int = 0
    num_feasible: int = 0
    num_optimal: int = 0


def compute_metrics(results: List[PathResult], grid: Optional[np.ndarray] = None,
                    grids: Optional[List[np.ndarray]] = None) -> EvaluationReport:
    """Compute aggregate metrics from a list of path results.

    Args:
        results: List of path results.
        grid: Single grid to check all paths against (for shared-grid benchmarks).
        grids: Per-task grids (one per result). Takes precedence over `grid`.
    """
    n = len(results)
    if n == 0:
        return EvaluationReport(num_tasks=0)

    feasible_paths = []
    error_counts = {
        "no_output": 0,
        "start_end_mismatch": 0,
        "obstacle_collision": 0,
        "out_of_bounds": 0,
        "invalid_step": 0,
    }

    for i, r in enumerate(results):
        g = grids[i] if grids is not None else grid
        if g is None:
            continue
        if r.path is None or len(r.path) < 2:
            error_counts["no_output"] += 1
            continue
        if r.path[0] != r.optimal_path[0] or r.path[-1] != r.optimal_path[-1]:
            error_counts["start_end_mismatch"] += 1
            continue
        if not _is_collision_free(r.path, g):
            error_counts["obstacle_collision"] += 1
            continue
        if not _is_in_bounds(r.path, g.shape):
            error_counts["out_of_bounds"] += 1
            continue
        if not _is_valid_steps(r.path):
            error_counts["invalid_step"] += 1
            continue
        feasible_paths.append(r)

    feasible = len(feasible_paths)
    compliant = max(feasible, 1)

    optimal_ratios = [
        r.optimal_length / max(1, len(r.path) - 1)
        if len(r.path) - 1 > 0 else 1.0
        for r in feasible_paths
    ]
    gms = [
        np.log(max(1, len(r.path) - 1)) - np.log(max(1, r.optimal_length))
        for r in feasible_paths
    ]
    mses = [
        (len(r.path) - 1 - r.optimal_length) ** 2
        for r in feasible_paths
    ]
    num_optimal = sum(1 for r in feasible_paths
                      if len(r.path) - 1 == r.optimal_length)

    return EvaluationReport(
        compliance_ratio=compliant / n if n > 0 else 0,
        feasibility_ratio=feasible / n if n > 0 else 0,
        optimal_ratio=num_optimal / n if n > 0 else 0,
        geometric_mean=np.exp(np.mean(gms)) if gms else 0,
        mse=np.mean(mses) if mses else 0,
        avg_runtime_ms=np.mean([r.latency_ms for r in results if r.latency_ms > 0]) if results else 0,
        success_rate=feasible / n if n > 0 else 0,
        valid_move_ratio=0.0,
        avg_tokens=np.mean([r.tokens_generated for r in results if r.tokens_generated > 0]) if results else 0,
        avg_vram_gb=np.mean([r.vram_peak_gb for r in results if r.vram_peak_gb > 0]) if results else 0,
        failure_types=error_counts,
        num_tasks=n,
        num_compliant=compliant,
        num_feasible=feasible,
        num_optimal=num_optimal,
    )


def _is_collision_free(path: List[Tuple[int, int]], grid: np.ndarray) -> bool:
    """Check no path cell intersects an obstacle."""
    return all(grid[y, x] == 0 for x, y in path)


def _is_in_bounds(path: List[Tuple[int, int]], shape: Tuple[int, int]) -> bool:
    """Check all path cells within grid bounds."""
    h, w = shape
    return all(0 <= x < w and 0 <= y < h for x, y in path)


def _is_valid_steps(path: List[Tuple[int, int]]) -> bool:
    """Check all steps are unit-length (no diagonal moves)."""
    for i in range(len(path) - 1):
        dx = abs(path[i][0] - path[i+1][0])
        dy = abs(path[i][1] - path[i+1][1])
        if dx + dy != 1:
            return False
    return True


def print_report(report: EvaluationReport, name: str = ""):
    """Pretty-print an evaluation report."""
    print(f"\n{'='*50}")
    print(f"Evaluation Report: {name}")
    print(f"{'='*50}")
    print(f"Tasks:           {report.num_tasks}")
    print(f"Compliance (CR): {report.compliance_ratio:.2%}")
    print(f"Feasibility (FR):{report.feasibility_ratio:.2%}")
    print(f"Optimal (OR):    {report.optimal_ratio:.2%}")
    print(f"Geometric Mean:  {report.geometric_mean:.3f}")
    print(f"MSE:             {report.mse:.1f}")
    print(f"Avg Runtime:     {report.avg_runtime_ms:.0f}ms")
    print(f"Avg Tokens:      {report.avg_tokens:.0f}")
    print(f"Avg VRAM:        {report.avg_vram_gb:.2f}GB")
    print(f"Failure Types:   {report.failure_types}")
