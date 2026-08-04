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

Sampling: candidates are bucketed into equal-width RATING bands and an equal
quota is taken from each, so the committed sets span the difficulty range
rather than inheriting the puzzle DB's heavy skew toward easy mates. Within a
band, selection is by puzzle-id order (deterministic and rating-independent).

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
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess  # noqa: E402

RAW_PATH = Path("data/raw/lichess_db_puzzle.csv")
OUT_1 = Path("data/positions/mate1-lichess.json")
OUT_2 = Path("data/positions/mate2-lichess.json")

TARGET_PER_TASK = 250
# Rating stratification: equal-width bands over [RATING_MIN, RATING_MAX], with
# an equal quota per band, so the committed set spans the difficulty range
# instead of inheriting the puzzle DB's skew toward easy mates.
RATING_MIN, RATING_MAX = 800, 2900
RATING_BANDS = 10
PER_BAND = TARGET_PER_TASK // RATING_BANDS  # 25 per band
SEED = 2026  # fixed output shuffle so records[:n] is a fair rating spread

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


def _rating_band(rating: int) -> int:
    """Which equal-width rating band a puzzle falls into (0..RATING_BANDS-1)."""
    width = (RATING_MAX - RATING_MIN) / RATING_BANDS
    return min(RATING_BANDS - 1, max(0, int((rating - RATING_MIN) // width)))


def select_candidates() -> dict:
    """Stream the CSV once; return candidates keyed by (theme, rating band).

    The band is derived from the puzzle's RATING. It used to be
    `int(PuzzleId, 36) % RATING_BANDS` — a hash of the id, unrelated to
    difficulty despite the RATING_BANDS name — and the picking loop had no
    per-band quota, so it drained band 0 and stopped. Every committed record
    came from one hash bucket and the "stratified by rating" claim was not
    implemented.
    """
    keep = {"mateIn1": {}, "mateIn2": {}}
    with open(RAW_PATH, newline="") as f:
        for row in csv.DictReader(f):
            themes = row["Themes"].split()
            if not _candidate_ok(row):
                continue
            band = _rating_band(int(row["Rating"]))
            for theme in keep:
                if theme in themes:
                    keep[theme].setdefault(band, []).append(row)
    for theme, bands in keep.items():
        total = sum(len(v) for v in bands.values())
        print(f"candidates: {theme}={total} "
              f"per band {{{', '.join(f'{b}:{len(bands.get(b, []))}' for b in range(RATING_BANDS))}}}")
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


def _try_row(p, theme, seen_fens, skipped):
    """Turn one CSV row into a committed record, or None with a skip reason."""
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
            return None
        board, first = parsed
        fen_key = board.fen()
        sol = meta["moves"].split()[1:]  # solver's solution from the presented position
        if fen_key in seen_fens:
            skipped["dup_fen"] += 1
            return None
        if theme == "mateIn1":
            # python-chess verification: every move that actually mates
            mates = []
            for mv in board.legal_moves:
                board.push(mv)
                if board.is_checkmate():
                    mates.append(mv.uci())
                board.pop()
            if not mates:
                skipped["mate1_no_mate"] += 1
                return None
            if len(mates) == board.legal_moves.count():
                skipped["mate1_vacuous"] += 1
                return None
            rec = _record(board, first, f"lichess-{p['PuzzleId']}",
                          p["PuzzleId"], sol[0] if sol else mates[0], meta)
            rec["task_extra"]["mate_moves"] = mates
        else:  # mateIn2
            if len(sol) < 1:
                skipped["mate2_no_line"] += 1
                return None
            try:
                first_uci = chess.Move.from_uci(sol[0])
            except ValueError:
                skipped["mate2_no_line"] += 1
                return None
            if first_uci not in board.legal_moves:
                skipped["mate2_no_line"] += 1
                return None
            if board.legal_moves.count() < 2:
                skipped["mate2_no_line"] += 1
                return None
            rec = _record(board, first, f"lichess2-{p['PuzzleId']}",
                          p["PuzzleId"], sol[0], meta)
        seen_fens.add(fen_key)
        return rec
    except Exception:
        skipped["bad"] += 1
        return None


def build() -> None:
    candidates = select_candidates()
    skipped = {"bad": 0, "mate1_no_mate": 0, "mate1_vacuous": 0,
               "mate2_no_line": 0, "dup_fen": 0}
    outputs = {}
    for theme in ("mateIn1", "mateIn2"):
        out, seen_fens, taken = [], set(), set()
        # Pass 1 fills an equal quota per rating band; pass 2 tops up from
        # whatever is left, so sparse high-rating bands cannot shrink the set
        # below TARGET_PER_TASK while still spending the quota on them first.
        for quota in (PER_BAND, TARGET_PER_TASK):
            for band in range(RATING_BANDS):
                rows = sorted(candidates[theme].get(band, []),
                              key=lambda r: r["PuzzleId"])
                in_band = 0
                for p in rows:
                    if len(out) >= TARGET_PER_TASK or in_band >= quota:
                        break
                    if p["PuzzleId"] in taken:
                        continue
                    rec = _try_row(p, theme, seen_fens, skipped)
                    if rec is None:
                        continue
                    out.append(rec)
                    taken.add(p["PuzzleId"])
                    in_band += 1
            if len(out) >= TARGET_PER_TASK:
                break
        # The picking loop emits band 0 first, band 9 last. Runners take
        # records[:n], so a band-ordered file would hand every short run the
        # easiest puzzles only. Shuffle once with a fixed seed: any prefix is
        # then a fair spread across the rating range, reproducibly.
        random.Random(SEED).shuffle(out)
        outputs[theme] = out

    OUT_1.write_text(json.dumps(outputs["mateIn1"], indent=1))
    OUT_2.write_text(json.dumps(outputs["mateIn2"], indent=1))
    print(f"built mate1={len(outputs['mateIn1'])} mate2={len(outputs['mateIn2'])} "
          f"-> data/positions/")
    print(f"skipped: {skipped}")
    for name, theme in (("mate1", "mateIn1"), ("mate2", "mateIn2")):
        bands = {}
        for r in outputs[theme]:
            b = _rating_band(r["task_extra"]["rating"])
            bands[b] = bands.get(b, 0) + 1
        print(f"{name} rating bands: "
              + ", ".join(f"{b}:{bands.get(b, 0)}" for b in range(RATING_BANDS)))


if __name__ == "__main__":
    build()
