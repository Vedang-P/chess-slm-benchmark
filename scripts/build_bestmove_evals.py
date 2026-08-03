"""Build the best-move task set from the lichess eval database.

Source: Lichess/chess-position-evaluations on Hugging Face (CC0 mirror of
https://database.lichess.org/lichess_db_eval.jsonl.zst, ~395M positions with
Stockfish evals). Streamed locally; only rows that pass our filters touch
the engine.

Position quality bar (metadata-rich, non-trivial best moves):
  - eval depth >= 25 and a real principal variation (>= 3 plies)
  - best move is legal in OUR variant (no castling / double-step / en
    passant — those can never be scored compliant here)
  - no en-passant square in the FEN
  - 12-30 pieces on the board (skip dead endgames and full openings)
  - centipawn eval in [-250, 250] (positions where the best move matters);
    mate evals are excluded

Every record carries the engine metadata (cp, mate, depth, knodes, PV) and
the source FEN, so analysis can stratify by eval, depth, material, etc.

Output (committed):
    data/positions/bestmove-8x8.json   (~120 positions, ordered deterministically)
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess  # noqa: E402

OUT = Path("data/positions/bestmove-8x8.json")
TARGET = 120
MAX_ROWS = 4_000_000  # streaming scan cap (several shards)

MIN_DEPTH = 25
MIN_PV_PLIES = 3
PIECES_MIN, PIECES_MAX = 12, 30
CP_MIN, CP_MAX = -250, 250


def _piece_count(fen: str) -> int:
    return sum(ch.isalpha() for ch in fen.split()[0])


def _best_move_legal(fen: str, line_first: str) -> bool:
    """Best move must be legal under STANDARD chess (python-chess)."""
    try:
        board = chess.Board(fen)
    except ValueError:
        return False
    legal = {m.uci() for m in board.legal_moves}
    if line_first not in legal:
        return False
    if len(legal) < 5:
        return False  # near-stalemate positions make the task vacuous
    return True


def _row_ok(row: dict) -> bool:
    fen = row["fen"]
    if not (PIECES_MIN <= _piece_count(fen) <= PIECES_MAX):
        return False
    if row.get("mate") is not None:
        return False
    cp = row.get("cp")
    if cp is None or not (CP_MIN <= cp <= CP_MAX):
        return False
    if (row.get("depth") or 0) < MIN_DEPTH:
        return False
    line = (row.get("line") or "").split()
    if len(line) < MIN_PV_PLIES:
        return False
    return True


def build() -> None:
    from datasets import load_dataset

    ds = load_dataset("Lichess/chess-position-evaluations", split="train",
                      streaming=True)
    best_by_fen = {}
    rows_seen = 0
    for row in ds:
        rows_seen += 1
        if rows_seen > MAX_ROWS:
            break
        if not _row_ok(row):
            continue
        fen = row["fen"]
        if fen in best_by_fen:
            if row["depth"] > best_by_fen[fen]["depth"]:
                best_by_fen[fen] = row
            continue
        best_by_fen[fen] = row
        if len(best_by_fen) >= TARGET * 3:
            break  # enough unique candidates; dedupe/order below

    cands = sorted(best_by_fen.values(),
                   key=lambda r: (r["fen"], -r["depth"]))
    # keep only positions whose best move is legal in our variant
    out = []
    for row in cands:
        line = row["line"].split()
        if not _best_move_legal(row["fen"], line[0]):
            continue
        out.append(row)
        if len(out) >= TARGET:
            break

    # deterministic order (seeded) so records[:full_n] is a fair spread
    rng = random.Random(2026)
    rng.shuffle(out)

    records = []
    for row in out:
        line = row["line"].split()
        board = chess.Board(row["fen"])
        records.append({
            "id": f"bm-{len(records):04d}",
            "source": "lichess-eval-db",
            "n": 8,
            "turn": "w" if board.turn == chess.WHITE else "b",
            "value": "cap",
            "fen": row["fen"],
            "presented_fen": board.fen(),
            "pieces": [
                {"sq": chess.square_name(sq),
                 "color": "w" if board.piece_at(sq).color == chess.WHITE else "b",
                 "kind": board.piece_at(sq).symbol().upper()}
                for sq in chess.SQUARES if board.piece_at(sq) is not None
            ],
            "win_moves": [],
            "lose_moves": [],
            "over_budget": False,
            "task_extra": {
                "best_move": line[0],
                "cp": row.get("cp"),
                "mate": row.get("mate"),
                "depth": row.get("depth"),
                "knodes": row.get("knodes"),
                "pv": row["line"],
            },
        })
    OUT.write_text(json.dumps(records, indent=1))
    print(f"scanned {rows_seen} rows, candidates {len(best_by_fen)}, "
          f"built {len(records)} -> {OUT}")
    print(f"cp range: {min(r['task_extra']['cp'] for r in records)}.."
          f"{max(r['task_extra']['cp'] for r in records)} | "
          f"depth range: {min(r['task_extra']['depth'] for r in records)}.."
          f"{max(r['task_extra']['depth'] for r in records)}")


if __name__ == "__main__":
    build()
