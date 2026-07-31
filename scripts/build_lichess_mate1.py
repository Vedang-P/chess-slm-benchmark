"""Build the lichess-derived mate-in-1 task set.

Source: lichess puzzle database (CC0), streamed and filtered by the
`mateIn1` theme into data/external/lichess_mate1_raw.json by this script's
--fetch mode (or a previous run).

Lichess convention: FEN is the position BEFORE the solver's opponent moves;
the position presented to the solver is FEN + first move. The solver's
solution is the remaining moves, ending in mate. For mate-in-1 the mating
move is verified by OUR engine, and every record also requires at least one
non-mate legal move (so the LOSE condition is non-vacuous).

Usage:
    python scripts/build_lichess_mate1.py --fetch        # download + filter raw
    python scripts/build_lichess_mate1.py                # build the task set
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
from src.benchmarks.games.fen import parse_fen  # noqa: E402
from src.benchmarks.games.rules import algebraic_to_sq  # noqa: E402

RAW_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
RAW_PATH = Path("data/external/lichess_mate1_raw.json")
OUT_PATH = Path("data/positions/mate1-lichess.json")


def fetch_raw(max_keep: int = 300) -> None:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    dctx = zstandard.ZstdDecompressor()
    kept = []
    with urllib.request.urlopen(RAW_URL) as resp:
        reader = dctx.stream_reader(resp)
        cr = csv.DictReader(io.TextIOWrapper(reader, encoding="utf-8"))
        for row in cr:
            if "mateIn1" in row["Themes"]:
                kept.append({
                    "id": row["PuzzleId"], "fen": row["FEN"], "moves": row["Moves"],
                    "rating": int(row["Rating"]),
                })
                if len(kept) >= max_keep:
                    break
    RAW_PATH.write_text(json.dumps(kept, indent=1))
    print(f"kept {len(kept)} mateIn1 raw puzzles -> {RAW_PATH}")


def _uci_to_sq(uci: str):
    return algebraic_to_sq(uci[:2]), algebraic_to_sq(uci[2:4])


def build() -> None:
    O.clear_cache()
    raw = json.loads(RAW_PATH.read_text())
    out = []
    skipped = {"non_mate_verified": 0, "vacuous": 0, "bad": 0}
    for p in raw:
        try:
            fen = p["fen"]
            first = p["moves"].split()[0]
            fr, to = _uci_to_sq(first)
            b0 = parse_fen(fen)
            legal0 = {m.uci for m in b0.legal_moves()}
            if first not in legal0:
                skipped["bad"] += 1
                continue
            m0 = next(m for m in b0.legal_moves() if m.uci == first)
            presented = b0.apply(m0)
            if presented.turn not in ("w", "b") or not presented.king_square(presented.turn):
                skipped["bad"] += 1
                continue
            terminal, _ = presented.outcome()
            if terminal:
                skipped["bad"] += 1
                continue
            mates = O.checkmate_moves(presented)
            if not mates:
                skipped["non_mate_verified"] += 1
                continue
            if len(mates) == len(presented.legal_moves()):
                skipped["vacuous"] += 1
                continue
            rec = {
                "id": f"lichess-{p['id']}",
                "source": "lichess",
                "puzzle_id": p["id"],
                "rating": p["rating"],
                "n": 8,
                "turn": presented.turn,
                "value": "win",
                "fen": fen,
                "presented_after": first,
                "pieces": [
                    {"sq": f"{chr(ord('a') + c)}{r + 1}", "color": color, "kind": kind}
                    for (r, c), (color, kind) in sorted(presented.pieces.items())
                ],
                "win_moves": [m.uci for m in mates],
                "lose_moves": [],
                "over_budget": False,
                "task_extra": {"mate_moves": [m.uci for m in mates]},
            }
            out.append(rec)
        except Exception as e:  # defensive: skip malformed puzzles
            skipped["bad"] += 1
            if skipped["bad"] < 3:
                print(f"  skip {p['id']}: {e}", flush=True)
    OUT_PATH.write_text(json.dumps(out, indent=1))
    print(f"built {len(out)} mate-in-1 lichess positions -> {OUT_PATH}")
    print(f"skipped: {skipped}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="download + filter the raw DB first")
    args = ap.parse_args()
    if args.fetch:
        fetch_raw()
    build()
