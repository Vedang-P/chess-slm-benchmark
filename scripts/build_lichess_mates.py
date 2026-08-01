"""Build lichess-derived tactic task sets (mate-in-1 / mate-in-2).

Source: lichess puzzle database (CC0), stream-filtered by theme.
Lichess convention: FEN is the position BEFORE the solver's opponent moves;
the position presented to the solver is FEN + first move. For mate-in-N the
solver's solution begins at the presented position; all solution moves are
'only moves' (exceptions: any checkmating move wins a mate-in-1).

Outputs (committed):
    data/positions/mate1-lichess.json   (288 positions, engine-verified mates)
    data/positions/mate2-lichess.json   (positions with a unique best first move)

Usage:
    python scripts/build_lichess_mates.py --fetch        # download + filter raw
    python scripts/build_lichess_mates.py                # build both task sets
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import urllib.request
import zstandard
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmarks.games import oracles as O  # noqa: E402
from src.benchmarks.games.fen import fen_of_board, parse_fen  # noqa: E402
from src.benchmarks.games.rules import algebraic_to_sq  # noqa: E402

RAW_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
RAW_PATH = Path("data/external/lichess_mates_raw.json")
OUT_1 = Path("data/positions/mate1-lichess.json")
OUT_2 = Path("data/positions/mate2-lichess.json")

THEMES = ("mateIn1", "mateIn2")


def fetch_raw(max_keep: int = 500) -> None:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    dctx = zstandard.ZstdDecompressor()
    kept = []
    with urllib.request.urlopen(RAW_URL) as resp:
        cr = csv.DictReader(io.TextIOWrapper(dctx.stream_reader(resp), encoding="utf-8"))
        for row in cr:
            if any(t in row["Themes"] for t in THEMES):
                kept.append({
                    "id": row["PuzzleId"], "fen": row["FEN"], "moves": row["Moves"],
                    "rating": int(row["Rating"]), "themes": row["Themes"],
                })
                if len(kept) >= max_keep:
                    break
    RAW_PATH.write_text(json.dumps(kept, indent=1))
    print(f"kept {len(kept)} raw puzzles -> {RAW_PATH}")


def _uci_to_sq(uci: str):
    return algebraic_to_sq(uci[:2]), algebraic_to_sq(uci[2:4])


def _presented(p: dict):
    fen = p["fen"]
    first = p["moves"].split()[0]
    fr, to = _uci_to_sq(first)
    b0 = parse_fen(fen)
    legal0 = {m.uci for m in b0.legal_moves()}
    if first not in legal0:
        return None
    m0 = next(m for m in b0.legal_moves() if m.uci == first)
    presented = b0.apply(m0)
    if not presented.king_square(presented.turn):
        return None
    terminal, _ = presented.outcome()
    if terminal:
        return None
    return presented, first


def _record(presented, first, rec_id, puzzle_id: str, solution_start: str, rating: int) -> dict:
    return {
        "id": rec_id,
        "source": "lichess",
        "puzzle_id": puzzle_id,
        "rating": rating,
        "n": 8,
        "turn": presented.turn,
        "value": "cap",
        "fen": None,
        "presented_fen": fen_of_board(presented),
        "pieces": [
            {"sq": f"{chr(ord('a') + c)}{r + 1}", "color": color, "kind": kind}
            for (r, c), (color, kind) in sorted(presented.pieces.items())
        ],
        "win_moves": [],
        "lose_moves": [],
        "over_budget": False,
        "task_extra": {"first_move": solution_start, "presented_after": first},
    }


def build() -> None:
    O.clear_cache()
    raw = json.loads(RAW_PATH.read_text())
    out1, out2 = [], []
    skipped = {"mate1_no_mate": 0, "mate1_vacuous": 0, "bad": 0, "mate2_no_line": 0}
    for p in raw:
        try:
            parsed = _presented(p)
            if parsed is None:
                skipped["bad"] += 1
                continue
            presented, first = parsed
            moves = p["moves"].split()
            sol = moves[1:]  # solver's solution from the presented position
            if "mateIn1" in p["themes"]:
                mates = O.checkmate_moves(presented)
                if not mates:
                    skipped["mate1_no_mate"] += 1
                    continue
                if len(mates) == len(presented.legal_moves()):
                    skipped["mate1_vacuous"] += 1
                    continue
                rec = _record(presented, first, f"lichess-{p['id']}", p["id"],
                              sol[0] if sol else mates[0].uci, p["rating"])
                rec["task_extra"]["mate_moves"] = [m.uci for m in mates]
                rec["fen"] = p["fen"]
                out1.append(rec)
            if "mateIn2" in p["themes"]:
                if len(sol) < 1 or sol[0] not in {m.uci for m in presented.legal_moves()}:
                    skipped["mate2_no_line"] += 1
                    continue
                if len(presented.legal_moves()) < 2:
                    skipped["mate2_no_line"] += 1
                    continue
                rec = _record(presented, first, f"lichess2-{p['id']}", p["id"],
                              sol[0], p["rating"])
                rec["fen"] = p["fen"]
                out2.append(rec)
        except Exception:
            skipped["bad"] += 1
    OUT_1.write_text(json.dumps(out1, indent=1))
    OUT_2.write_text(json.dumps(out2, indent=1))
    print(f"built mate1={len(out1)} mate2={len(out2)} -> data/positions/")
    print(f"skipped: {skipped}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    if args.fetch:
        fetch_raw()
    build()
