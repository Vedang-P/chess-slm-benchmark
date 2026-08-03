"""Prompt construction for the chess study.

Single representation (FEN) and single condition (win), per the Phase-1
finding and the study design. The position is always the PRESENTED
position (lichess records store the pre-move FEN separately; showing the
pre-move FEN made the board inconsistent with the turn text — a real
formatting bug found during debugging).
"""
from __future__ import annotations

from typing import Dict, List

RULES_SUMMARY = (
    "Standard chess rules apply on this {n}x{n} board: castling, en passant, "
    "double-step pawn moves, and pawn promotion are all legal. It is illegal "
    "to leave your own king in check."
)

OBJECTIVES = {
    "win": (
        "YOUR OBJECTIVE: WIN. Choose the single move that is most likely to "
        "lead YOU to victory. Play the strongest move you can find."
    ),
}

MATE1_OBJECTIVES = {
    "win": ("YOUR OBJECTIVE: WIN. Deliver CHECKMATE in exactly one move."),
}

OUTPUT_SPEC = (
    "Output ONLY a single line of the form: MOVE: <from><to> using algebraic "
    "notation (e.g. 'MOVE: e2e4'). Do not output anything else."
)


def _render_board(pieces: List[Dict[str, str]], n: int) -> str:
    grid = {p["sq"]: p for p in pieces}
    lines = ["   " + " ".join(chr(ord("a") + c) for c in range(n))]
    for r in range(n - 1, -1, -1):
        row = []
        for c in range(n):
            sq = f"{chr(ord('a') + c)}{r + 1}"
            p = grid.get(sq)
            row.append(f"{p['color']}{p['kind']}" if p else "..")
        lines.append(f"{r + 1:2d} " + " ".join(row))
    return "\n".join(lines)


def _fen_of(rec: Dict[str, object]) -> str:
    """FEN for an 8x8 record. Prefers the PRESENTED position: lichess records
    store the pre-move FEN in 'fen' and the solver's actual position in
    'presented_fen' — showing the pre-move FEN made the board inconsistent
    with the turn text (a real formatting bug found during debugging)."""
    if rec.get("presented_fen"):
        return rec["presented_fen"]
    if rec.get("fen"):
        return rec["fen"]
    import chess

    board = chess.Board(None)
    for p in rec.get("pieces") or []:
        board.set_piece_at(chess.parse_square(p["sq"]),
                           chess.Piece.from_symbol(p["kind"] if p["color"] == "w"
                                                   else p["kind"].lower()))
    board.turn = chess.WHITE if rec["turn"] == "w" else chess.BLACK
    return board.fen()


def _bitboards(rec: Dict[str, object]) -> str:
    """64-bit bitboards per piece type, a1 = LSB, rendered as 8 rows of 8 bits
    (rank 8 first) — the standard engine representation."""
    import chess

    board = chess.Board(rec.get("presented_fen") or rec.get("fen"))
    out = []
    for color in ("w", "b"):
        for kind in ("K", "Q", "R", "B", "N", "P"):
            bits = 0
            for sq in chess.SQUARES:
                pc = board.piece_at(sq)
                if pc is not None and (pc.color == chess.WHITE) == (color == "w") \
                        and pc.symbol().upper() == kind:
                    bits |= 1 << sq
            rows = []
            for rank in range(7, -1, -1):
                row = "".join("1" if bits & (1 << (rank * 8 + file)) else "0"
                              for file in range(8))
                rows.append(row)
            out.append(f"{color}{kind}: " + " ".join(rows))
    return "\n".join(out)


def _piece_list(rec: Dict[str, object]) -> str:
    """Plain piece list, e.g. 'White: Ke1, Qd1 ...' — no board geometry."""
    groups = {"w": [], "b": []}
    for p in rec["pieces"]:
        groups[p["color"]].append(f"{p['kind']}{p['sq']}")
    return (f"White pieces: {', '.join(groups['w']) or 'none'}.\n"
            f"Black pieces: {', '.join(groups['b']) or 'none'}.")


def _presentation(rec: Dict[str, object], variant: str) -> str:
    n = rec["n"]
    if variant == "fen":
        if n != 8:
            raise ValueError(f"FEN variant requires 8x8, got {n}x{n}")
        return f"The position in FEN notation: {_fen_of(rec)}"
    if variant == "bitboard":
        if n != 8:
            raise ValueError(f"bitboard variant requires 8x8, got {n}x{n}")
        return ("The position as bitboards (one 8x8 grid of bits per piece "
                "type, a1 is the first bit of the first row, rank 8 first; "
                "1 = a piece of that type stands on the square):\n"
                + _bitboards(rec))
    if variant == "list":
        return _piece_list(rec)
    return ("Here is the current position (each cell shows the piece or '..'):\n"
            + _render_board(rec["pieces"], n))


def _turn_line(rec: Dict[str, object]) -> str:
    return f"It is {'White' if rec['turn'] == 'w' else 'Black'} to move."


def build_single_move_prompt(rec: Dict[str, object], condition: str,
                             variant: str = "grid") -> str:
    # fixed instruction block FIRST, position LAST: the gateway's automatic
    # prompt caching then reuses the identical prefix across all positions
    # of a cell (cache hit = cheaper + faster)
    return "\n".join([
        f"You are playing chess on a {rec['n']}x{rec['n']} board.",
        RULES_SUMMARY.format(n=rec["n"]),
        OBJECTIVES[condition],
        OUTPUT_SPEC,
        _presentation(rec, variant),
        _turn_line(rec),
    ])


def build_mate1_prompt(rec: Dict[str, object], condition: str,
                       variant: str = "grid") -> str:
    return "\n".join([
        f"You are playing chess on a {rec['n']}x{rec['n']} board.",
        RULES_SUMMARY.format(n=rec["n"]),
        MATE1_OBJECTIVES[condition],
        OUTPUT_SPEC,
        _presentation(rec, variant),
        _turn_line(rec),
    ])


def build_mobility_prompt(rec: Dict[str, object], condition: str,
                          variant: str = "grid") -> str:
    return "\n".join([
        f"You are playing chess on a {rec['n']}x{rec['n']} board.",
        RULES_SUMMARY.format(n=rec["n"]),
        MOBILITY_OBJECTIVES[condition],
        OUTPUT_SPEC,
        _presentation(rec, variant),
        _turn_line(rec),
    ])


def build_cap_prompt(rec: Dict[str, object], condition: str,
                     variant: str = "grid") -> str:
    """Pure-legality probe: same board presentation, no objective pressure."""
    return "\n".join([
        f"You are playing chess on a {rec['n']}x{rec['n']} board.",
        RULES_SUMMARY.format(n=rec["n"]),
        CAP_OBJECTIVE,
        OUTPUT_SPEC,
        _presentation(rec, variant),
        _turn_line(rec),
    ])
