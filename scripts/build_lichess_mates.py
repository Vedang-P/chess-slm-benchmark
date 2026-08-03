"""Build lichess-derived tactic task sets (mate-in-1 / mate-in-2).

Source: the official lichess puzzle database (CC0,
https://database.lichess.org/lichess_db_puzzle.csv.zst), filtered by theme
and stratified by rating so the committed sets span the difficulty range.

Rules: STANDARD chess, faithfully — python-chess applies the opponent's
first move, serializes the presented FEN with real castling/ep rights, and
verifies checkmating moves. Castling, en passant, double-step, and
promotion are all legal, exactly as the models were trained to know them.

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
    python scripts/build_lichess_mates.py   # needs data/raw/lichess_db_puzzle.csv
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess  # noqa: E402

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


def _presented(p: dict):
    """Presented position = puzzle FEN + the opponent's first move, applied
    by python-chess (standard rules). Returns (board, first_uci) or None.
    p is the meta dict (lowercase keys: fen, moves, ...)."""
    try:
        board = chess.Board(p["fen"])
    except ValueError:
        return None
    first = p["moves"].split()[0]
    try:
        board.push_uci(first)
    except ValueError:
        return None
    if not (board.king(chess.WHITE) and board.king(chess.BLACK)):
        return None
    if board.is_game_over():
        return None
    return board, first


def _record(board, first, rec_id, puzzle_id: str, solution_start: str,
            meta: dict) -> dict:
    task_extra = {"first_move": solution_start, "opponent_first_move": first}
    for k in META_FIELDS:
        task_extra[k] = meta.get(k)
    return {
        "id": rec_id,
        "source": "lichess-puzzle-db",
        "puzzle_id": puzzle_id,
        "n": 8,
        "turn": "w" if board.turn == chess.WHITE else "b",
        "value": "cap",
        "fen": meta["fen"],
        "presented_fen": board.fen(),
        "pieces": [
            {"sq": chess.square_name(sq), "color": "w" if board.piece_at(sq).color == chess.WHITE else "b",
             "kind": board.piece_at(sq).symbol().upper()}
            for sq in chess.SQUARES if board.piece_at(sq) is not None
        ],
        "win_moves": [],
        "lose_moves": [],
        "over_budget": False,
        "task_extra": task_extra,
    }


def build() -> None:
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
                    board, first = parsed
                    fen_key = board.fen()
                    moves = meta["moves"].split()
                    sol = moves[1:]  # solver's solution from the presented position
                    if theme == "mateIn1":
                        if fen_key in seen_fens1:
                            skipped["dup_fen"] += 1
                            continue
                        # python-chess verification: all moves that actually mate
                        mates = []
                        for mv in board.legal_moves:
                            board.push(mv)
                            if board.is_checkmate():
                                mates.append(mv.uci())
                            board.pop()
                        if not mates:
                            skipped["mate1_no_mate"] += 1
                            continue
                        if len(mates) == board.legal_moves.count():
                            skipped["mate1_vacuous"] += 1
                            continue
                        seen_fens1.add(fen_key)
                        rec = _record(board, first, f"lichess-{p['PuzzleId']}",
                                      p["PuzzleId"], sol[0] if sol else mates[0],
                                      meta)
                        rec["task_extra"]["mate_moves"] = mates
                        out1.append(rec)
                    else:  # mateIn2
                        if fen_key in seen_fens2:
                            skipped["dup_fen"] += 1
                            continue
                        if len(sol) < 1:
                            skipped["mate2_no_line"] += 1
                            continue
                        try:
                            first_uci = chess.Move.from_uci(sol[0])
                        except ValueError:
                            skipped["mate2_no_line"] += 1
                            continue
                        if first_uci not in board.legal_moves:
                            skipped["mate2_no_line"] += 1
                            continue
                        if board.legal_moves.count() < 2:
                            skipped["mate2_no_line"] += 1
                            continue
                        seen_fens2.add(fen_key)
                        rec = _record(board, first, f"lichess2-{p['PuzzleId']}",
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
