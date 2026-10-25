"""Keep only reproducible anchor checkpoints and the RL resume window on HF.

The policy names historical runs explicitly; unknown future runs are untouched.
Evaluation summaries and label data are never deleted.  GitHub Actions applies
one bounded batch per tick, so cleanup does not hold up the GPU supervisor.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

from huggingface_hub import CommitOperationDelete

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.kaggle_checkpoint import api as make_api
REPO = "vedangfake/chess-slm-benchmark"
RL_RUNS = {"ccgavn-rl-root-v1-seed0", "ccgavn-rl-root-v2-seed0",
           "ccgavn-rl-root-v3-seed0"}

# Endpoints with archived benchmark use. The selected 76.20% baseline and its
# earlier warm-start anchors are pinned by exact step, not inferred by recency.
KEEP = {
    "ccgavn-5m-seed0-v2": {3_896_658},
    "ccgavn-5m-seed0": {320_000, 1_620_000, 1_715_000},
    "account1-baseline-5m-seed0": {55_000},
    "account1-gavn-3m-seed0": {105_000},
    "account2-gavn-3m-seed1": {110_000},
    "account2-gavn-5m-seed0": {160_000},
    "account3-gavn-5m-geometry": {160_000},
    "account3-gavn-5m-geometry-slim": {105_000},
    "account3-gavn-5m-loss": {160_000},
}
RETIRED = {
    "caveman-sft", "caveman-sft-pretest", "ccgavn-refutation-r0-seed0",
    "ccgavn-refutation-r0-seed0-v2", "gavn-3m", "noexplain-slice",
    "rlvr-gate-base5k", "rlvr-pretest", "rlvr-smoke",
    "smoke-ccgavn-disposable", "smoke-gavn-disposable",
    "smoke-gavn-slim-disposable", "smoke-gavn-slim-local-disposable",
    "smoke-student-disposable", "student-5m",
}


def checkpoint_folders(files: set[str]) -> dict[str, dict[int, set[str]]]:
    result: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    for path in files:
        parts = path.split("/")
        if len(parts) < 3 or not parts[1].startswith("checkpoint-"):
            continue
        try:
            step = int(parts[1].removeprefix("checkpoint-"))
        except ValueError:
            continue
        result[parts[0]][step].add("/".join(parts[2:]))
    return result


def plan(files: set[str]) -> tuple[list[str], list[str]]:
    folders = checkpoint_folders(files)
    to_delete, preserved = [], []
    for run, steps in sorted(folders.items()):
        if run in RL_RUNS:
            complete = sorted(step for step, contents in steps.items()
                              if {"state.pt", "config.json"} <= contents)
            keep = set(complete[-2:]) | ({0} if 0 in steps else set())
            if 20_000 in steps:
                keep.add(20_000)
            # An incomplete upload may still be active. Only prune complete
            # older checkpoints after at least two newer complete ones exist.
            candidates = [step for step in complete if step not in keep]
        elif run in KEEP:
            keep = KEEP[run]
            candidates = [step for step in steps if step not in keep]
        elif run in RETIRED:
            keep = set()
            candidates = list(steps)
        else:
            continue
        preserved.extend(f"{run}/checkpoint-{step}" for step in sorted(steps) if step not in candidates)
        to_delete.extend(f"{run}/checkpoint-{step}" for step in sorted(candidates))
    return to_delete, preserved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-folders", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.max_folders <= 200:
        raise ValueError("max-folders must be within 1..200")
    api = make_api(ROOT)
    files = set(api.list_repo_files(REPO, repo_type="dataset"))
    to_delete, preserved = plan(files)
    print(f"[retention] {len(to_delete)} stale checkpoint folders; "
          f"{len(preserved)} protected anchors/resume checkpoints", flush=True)
    print("[retention] anchors: " + ", ".join(preserved[:18]), flush=True)
    if not args.execute:
        print("[retention] dry run; first stale folders: " + ", ".join(to_delete[:10]),
              flush=True)
        return
    batch = to_delete[:args.max_folders]
    if not batch:
        return
    api.create_commit(repo_id=REPO, repo_type="dataset",
                      operations=[CommitOperationDelete(path_in_repo=path, is_folder=True)
                                  for path in batch],
                      commit_message=f"Prune {len(batch)} stale checkpoints; retain baselines and RL resume")
    print(f"[retention] removed {len(batch)} folders; {len(to_delete)-len(batch)} remain",
          flush=True)


if __name__ == "__main__":
    main()
