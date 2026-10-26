"""Retain tactical milestones and the latest two complete resume points.

Only the new tactical run prefix is eligible. The starting baseline and all
historical runs are outside this script's scope.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from huggingface_hub import CommitOperationDelete

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.kaggle_checkpoint import api as make_hf_api
from scripts.tactical_config import HF_REPO, RUN, milestones


def stale_folders(files: set[str]) -> list[str]:
    contents: dict[int, set[str]] = {}
    for path in files:
        match = re.fullmatch(rf"{re.escape(RUN)}/checkpoint-(\d+)/([^/]+)", path)
        if match:
            contents.setdefault(int(match.group(1)), set()).add(match.group(2))
    complete = sorted(step for step, names in contents.items()
                      if {"state.pt", "config.json"} <= names)
    if len(complete) < 3:
        return []
    keep = set(complete[-2:]) | set(milestones())
    return [f"{RUN}/checkpoint-{step}" for step in complete[:-2]
            if step not in keep]


def main() -> None:
    client = make_hf_api(ROOT)
    files = set(client.list_repo_files(HF_REPO, repo_type="dataset"))
    stale = stale_folders(files)[:8]
    if not stale:
        print("[retention] no tactical checkpoints to prune", flush=True)
        return
    client.create_commit(
        repo_id=HF_REPO, repo_type="dataset",
        operations=[CommitOperationDelete(path_in_repo=path, is_folder=True)
                    for path in stale],
        commit_message=f"Retain tactical milestones and resume window; prune {len(stale)}",
    )
    print(f"[retention] pruned {len(stale)} tactical checkpoints", flush=True)


if __name__ == "__main__":
    main()
