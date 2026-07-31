"""Build the pure-legality capability task from standard chess positions.

Positions sampled deterministically from the kagisearch 1000-puzzle CSV
(lichess-sourced, CC0). No oracle needed: the cap task only measures
parse rate + legal-move rate -- the model is told to make ANY legal move.

Usage:
    python scripts/build_cap_positions.py [--n 40]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmarks.games.fen import parse_fen  # noqa: E402

SRC = Path("data/external/kagi_puzzles.csv")
OUT = Path("data/positions/cap-legal-8x8.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(SRC)))
    step = max(1, len(rows) // (args.n * 4))  # spread picks across the file
    out = []
    for i in range(0, len(rows), step):
        if len(out) >= args.n:
            break
        row = rows[i]
        try:
            board = parse_fen(row["FEN"])
        except Exception:
            continue
        if not board.legal_moves():
            continue
        out.append({
            "id": f"cap-{row['PuzzleId']}",
            "source": "lichess-via-kagi",
            "puzzle_id": row["PuzzleId"],
            "n": 8,
            "turn": board.turn,
            "value": "cap",
            "fen": row["FEN"],
            "pieces": [
                {"sq": f"{chr(ord('a') + c)}{r + 1}", "color": color, "kind": kind}
                for (r, c), (color, kind) in sorted(board.pieces.items())
            ],
            "win_moves": [],
            "lose_moves": [],
            "over_budget": False,
        })
    OUT.write_text(json.dumps(out, indent=1))
    print(f"built {len(out)} cap positions -> {OUT}")


if __name__ == "__main__":
    main()
