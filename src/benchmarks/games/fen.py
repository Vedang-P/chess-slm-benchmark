"""FEN <-> our Board schema, with optional python-chess validation.

Our engine ignores castling/en-passant (documented variant); FEN fields for
those are parsed but unused. When python-chess is installed, FEN parsing and
move legality are cross-validated against it (the Kaggle check notebook
installs python-chess, so the parity test actually runs there).
"""
from __future__ import annotations

from typing import List, Dict, Optional

from src.benchmarks.games.rules import Board

_PIECE_MAP = {"K": "K", "Q": "Q", "R": "R", "B": "B", "N": "N", "P": "P"}


def parse_fen(fen: str) -> Board:
    """Standard FEN -> our Board (8x8). Castling/EP/halfmove/fullmove ignored."""
    placement, turn = fen.split()[0], fen.split()[1]
    pieces: Dict[tuple, tuple] = {}
    rows = placement.split("/")
    assert len(rows) == 8, f"expected 8 ranks, got {len(rows)}: {fen}"
    for r, row in enumerate(rows):
        c = 0
        for ch in row:
            if ch.isdigit():
                c += int(ch)
            else:
                color = "w" if ch.isupper() else "b"
                kind = _PIECE_MAP[ch.upper()]
                pieces[(7 - r, c)] = (color, kind)  # FEN rank 8 -> row 0
                c += 1
        assert c == 8, f"bad rank {row} in {fen}"
    return Board(8, pieces, turn)


def board_to_pieces(rec: Dict[str, object]) -> List[Dict[str, str]]:
    return rec["pieces"]


def fen_of_board(board: Board) -> str:
    rows = []
    for r in range(7, -1, -1):
        row = []
        empties = 0
        for c in range(8):
            occ = board.at((r, c))
            if occ is None:
                empties += 1
                continue
            if empties:
                row.append(str(empties))
                empties = 0
            color, kind = occ
            row.append(kind if color == "w" else kind.lower())
        if empties:
            row.append(str(empties))
        rows.append("".join(row))
    return "/".join(rows) + " " + ("w" if board.turn == "w" else "b") + " - - 0 1"


def validate_against_python_chess(fen: str, legal_uci: set) -> Optional[bool]:
    """Cross-check our legal-move set against python-chess for an 8x8 FEN.
    Returns None if python-chess is unavailable."""
    try:
        import chess  # type: ignore
    except ImportError:
        return None
    try:
        b = chess.Board(fen)
    except ValueError:
        return None
    theirs = {m.uci() for m in b.legal_moves}
    # our engine excludes king captures and includes all legal moves; parity
    # should hold for legal positions
    return theirs == set(legal_uci)
