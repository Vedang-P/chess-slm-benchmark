"""Print a complete frozen puzzle evaluation's accuracy for Autoresearch.

The command fails closed: a partial run, a wrong checkpoint, or a changed
benchmark size is not a comparable measurement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def score(summary: dict, checkpoint: str) -> float:
    if summary.get("checkpoint") != checkpoint:
        raise ValueError("evaluation checkpoint identity does not match")
    if summary.get("complete") is not True or summary.get("returncode") != 0:
        raise ValueError("evaluation is incomplete or failed")
    puzzles = summary.get("puzzles") or {}
    mate = summary.get("mate") or {}
    if puzzles.get("total") != 10_000 or puzzles.get("expected") != 10_000:
        raise ValueError("evaluation did not score the frozen 10,000 puzzles")
    if mate.get("total") != 4_000 or mate.get("expected") != 4_000:
        raise ValueError("evaluation did not score the frozen 4,000 MATE rows")
    solved = puzzles.get("solved")
    if not isinstance(solved, int) or not 0 <= solved <= 10_000:
        raise ValueError("invalid puzzle solved count")
    return solved / 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True,
                        help="local JSON path or path in the Hugging Face dataset")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hf-repo", default="vedangfake/chess-slm-benchmark")
    args = parser.parse_args()
    path = Path(args.summary)
    if not path.is_file():
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(args.hf_repo, args.summary,
                                    repo_type="dataset"))
    print(f"{score(json.loads(path.read_text(encoding='utf-8')), args.checkpoint):.2f}")


if __name__ == "__main__":
    main()
