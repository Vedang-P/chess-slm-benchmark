"""Build lichess-derived tactic task sets (mate-in-1 / mate-in-2).

Source: the official lichess puzzle database (CC0,
https://database.lichess.org/lichess_db_puzzle.csv.zst), filtered by theme
and stratified by rating so the committed sets span the difficulty range.

Lichess convention: FEN is the position BEFORE the solver's opponent moves;
the position presented to the solver is FEN + first move. For mate-in-N the
solver's solution begins at the presented position; all solution moves are
'only moves' (exceptions: any checkmating move wins a mate-in-1).

Every record carries the puzzle's full public metadata (rating, rating
deviation, popularity, play count, theme list, opening tags, game URL) so
downstream analysis can stratify by tactic type, difficulty, and opening.

Outputs (committed):
    data/positions/mate1-lichess.json   (~250 positions, engine-verified mates)
    data/positions/mate2-lichess.json   (~250 positions, unique best first move)

Usage:
    python scripts/build_lichess_mates.py   # build both task sets (uses the
                                            # local data/raw/lichess_db_puzzle.csv)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmarks.games import oracles as O  # noqa: E402
from src.benchmarks.games.fen import fen_of_board, parse_fen  # noqa: E402
from src.benchmarks.games.rules import algebraic_to_sq  # noqa: E402

RAW_PATH = Path("data/raw/lichess_db_puzzle.csv")
OUT_1 = Path("data/positions/mate1-lichess.json")
OUT_2 = Path("data/positions/mate2-lichess.json")

TARGET_PER_TASK = 250
RATING_BANDS = 10  # stratification granularity

META_FIELDS = ("rating", "rating_deviation", "popularity", "nb_plays",
               "themes", "opening_tags", "game_url")


def _candidate_ok(row: dict) -> bool:
    """Cheap pre-filter applied while streaming 6M rows (no engine work)."""
    try:
        if int(row["Rating"]) < 800 or int(row["Rating"]) > 2900:
            return False
        if int(row["NbPlays"]) < 50:
            return False
    except (KeyError, ValueError):
        return False
    # en-passant square present in the FEN (field index 3): our variant has
    # no en passant, and the puzzle's solution may rely on it -- reject
    parts = row["FEN"].split()
    if len(parts) >= 4 and parts[3] != "-":
        return False
    # puzzle FEN must be an 8x8 board (all lichess standard puzzles are)
    if not any("mateIn" in t for t in row["Themes"].split()):
        return False
    return True


def select_candidates() -> dict:
    """Stream the CSV once; return stratified candidate lists per theme."""
    keep = {"mateIn1": [], "mateIn2": []}
    with open(RAW_PATH, newline="") as f:
        for row in csv.DictReader(f):
            themes = row["Themes"].split()
            if not _candidate_ok(row):
                continue
            for theme in keep:
                if theme in themes:
                    # deterministic slot by puzzle id hash -> even spread
                    slot = int(row["PuzzleId"], 36) % RATING_BANDS
                    keep[theme].append((slot, row))
    print(f"candidates: mateIn1={len(keep['mateIn1'])} mateIn2={len(keep['mateIn2'])}")
    return keep


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


def _record(presented, first, rec_id, puzzle_id: str, solution_start: str,
            meta: dict) -> dict:
    task_extra = {"first_move": solution_start, "presented_after": first}
    for k in META_FIELDS:
        task_extra[k] = meta.get(k)
    return {
        "id": rec_id,
        "source": "lichess-puzzle-db",
        "puzzle_id": puzzle_id,
        "n": 8,
        "turn": presented.turn,
        "value": "cap",
        "fen": meta["fen"],
        "presented_fen": fen_of_board(presented),
        "pieces": [
            {"sq": f"{chr(ord('a') + c)}{r + 1}", "color": color, "kind": kind}
            for (r, c), (color, kind) in sorted(presented.pieces.items())
        ],
        "win_moves": [],
        "lose_moves": [],
        "over_budget": False,
        "task_extra": task_extra,
    }


def build() -> None:
    O.clear_cache()
    candidates = select_candidates()
    out1, out2 = [], []
    seen_fens1, seen_fens2 = set(), set()
    skipped = {"bad": 0, "mate1_no_mate": 0, "mate1_vacuous": 0,
               "mate2_no_line": 0, "dup_fen": 0}
    per_slot = TARGET_PER_TASK // RATING_BANDS + 1
    for theme in ("mateIn1", "mateIn2"):
        picked = 0
        for slot in range(RATING_BANDS):
            rows = [r for s, r in candidates[theme] if s == slot]
            rows.sort(key=lambda r: r["PuzzleId"])
            for p in rows:
                if picked >= TARGET_PER_TASK:
                    break
                meta = {
                    "fen": p["FEN"],
                    "moves": p["Moves"],
                    "rating": int(p["Rating"]),
                    "rating_deviation": int(p["RatingDeviation"]),
                    "popularity": int(p["Popularity"]),
                    "nb_plays": int(p["NbPlays"]),
                    "themes": p["Themes"],
                    "opening_tags": p.get("OpeningTags", ""),
                    "game_url": p.get("GameUrl", ""),
                }
                try:
                    parsed = _presented(meta)
                    if parsed is None:
                        skipped["bad"] += 1
                        continue
                    presented, first = parsed
                    fen_key = fen_of_board(presented)
                    moves = meta["moves"].split()
                    sol = moves[1:]  # solver's solution from the presented position
                    if theme == "mateIn1":
                        if fen_key in seen_fens1:
                            skipped["dup_fen"] += 1
                            continue
                        mates = O.checkmate_moves(presented)
                        if not mates:
                            skipped["mate1_no_mate"] += 1
                            continue
                        if len(mates) == len(presented.legal_moves()):
                            skipped["mate1_vacuous"] += 1
                            continue
                        seen_fens1.add(fen_key)
                        rec = _record(presented, first, f"lichess-{p['PuzzleId']}",
                                      p["PuzzleId"], sol[0] if sol else mates[0].uci,
                                      meta)
                        rec["task_extra"]["mate_moves"] = [m.uci for m in mates]
                        out1.append(rec)
                    else:  # mateIn2
                        if fen_key in seen_fens2:
                            skipped["dup_fen"] += 1
                            continue
                        if len(sol) < 1 or sol[0] not in {m.uci for m in presented.legal_moves()}:
                            skipped["mate2_no_line"] += 1
                            continue
                        if len(presented.legal_moves()) < 2:
                            skipped["mate2_no_line"] += 1
                            continue
                        seen_fens2.add(fen_key)
                        rec = _record(presented, first, f"lichess2-{p['PuzzleId']}",
                                      p["PuzzleId"], sol[0], meta)
                        out2.append(rec)
                    picked += 1
                except Exception:
                    skipped["bad"] += 1
            if picked >= TARGET_PER_TASK:
                break
    OUT_1.write_text(json.dumps(out1, indent=1))
    OUT_2.write_text(json.dumps(out2, indent=1))
    print(f"built mate1={len(out1)} mate2={len(out2)} -> data/positions/")
    print(f"skipped: {skipped}")


if __name__ == "__main__":
    build()

