"""Builds the multilingual GridRoute instruction set for the cross-lingual
navigation experiment: sample tasks, fill per-language templates.
"""

import json
from pathlib import Path
from typing import List

import numpy as np

from src.grid_generator import generate_gridroute_maps, GridRouteTask

TEMPLATE_FILE = Path(__file__).parent / "data" / "multilingual" / "gridroute_direct_template.json"


def load_templates() -> dict:
    with open(TEMPLATE_FILE) as f:
        data = json.load(f)
    langs = {"en": {"template": data["source_en"]["template"], "resource_tier": "reference"}}
    for code, entry in data["languages"].items():
        langs[code] = {"template": entry["template"], "resource_tier": entry["resource_tier"]}
    return langs


def fill_template(template: str, task: GridRouteTask) -> str:
    obstacles = [(x, y) for y in range(task.grid.shape[0])
                 for x in range(task.grid.shape[1]) if task.grid[y, x] == 1]
    obs_str = ", ".join(f"({x},{y})" for x, y in obstacles) if obstacles else "none"
    return template.format(
        start_x=task.start[0], start_y=task.start[1],
        goal_x=task.goal[0], goal_y=task.goal[1],
        w=task.size, h=task.size, obstacles=obs_str,
    )


def sample_tasks(n: int, seed: int = 42) -> List[GridRouteTask]:
    """Stratified sample across the map/pair structure, not just the first N."""
    all_tasks = generate_gridroute_maps(
        size=10, obstacle_size=3, num_obstacles=2,
        num_maps=100, pairs_per_map=5, seed=seed,
    )
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(all_tasks), size=min(n, len(all_tasks)), replace=False)
    return [all_tasks[i] for i in sorted(idx)]


def build_instances(n_tasks: int, seed: int = 42):
    """Returns list of dicts: {task_id, lang, resource_tier, instruction, task}."""
    tasks = sample_tasks(n_tasks, seed=seed)
    templates = load_templates()
    instances = []
    for task in tasks:
        for lang, meta in templates.items():
            instances.append({
                "task_id": task.task_id,
                "lang": lang,
                "resource_tier": meta["resource_tier"],
                "instruction": fill_template(meta["template"], task),
                "task": task,
            })
    return instances
