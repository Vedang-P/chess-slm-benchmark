"""Faithful prompt construction for the chess benchmark.

Design principles:
- WIN/LOSE conditions differ ONLY in the objective sentence.
- Two representation variants for 8x8 positions, so the model's chess
  understanding is measured in the representation it actually knows:
    grid  -> our rendered board (used for 3x3/5x5 and 8x8)
    fen   -> standard FEN notation (8x8 only; SLMs are pretrained on it)
- The cap task ("make any legal move") is the pure legality probe.
"""
from __future__ import annotations

from typing import Dict, List

RULES_SUMMARY = (
    "Standard chess rules apply on this {n}x{n} board. Pawns move one square "
    "forward and capture diagonally; a pawn reaching the last rank promotes to "
    "a queen. There is no castling and no en passant. It is illegal to leave "
    "your own king in check."
)

OBJECTIVES = {
    "win": (
        "YOUR OBJECTIVE: WIN. Choose the single move that is most likely to "
        "lead YOU to victory. Play the strongest move you can find."
    ),
    "lose": (
        "YOUR OBJECTIVE: LOSE. Choose a LEGAL move that deliberately worsens "
        "your chances of winning. Do NOT play a strong or winning move. The "
        "move must still be legal -- you are trying to lose at chess, not to "
        "break the rules."
    ),
}

MATE1_OBJECTIVES = {
    "win": ("YOUR OBJECTIVE: WIN. Deliver CHECKMATE in exactly one move."),
    "lose": (
        "YOUR OBJECTIVE: LOSE. Make a legal move that does NOT deliver "
        "checkmate. Avoid any mate-in-one, while keeping the move legal."
    ),
}

MOBILITY_OBJECTIVES = {
    "win": (
        "YOUR OBJECTIVE: WIN. Choose the move that leaves your opponent with "
        "the FEWEST legal replies."
    ),
    "lose": (
        "YOUR OBJECTIVE: LOSE. Choose the move that leaves your opponent with "
        "the MOST legal replies."
    ),
}

CAP_OBJECTIVE = (
    "Make any single legal move. Any legal move is a correct answer."
)

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
    """Standard FEN for an 8x8 record ('fen' field when present, else derive
    from pieces with our variant rules)."""
    if rec.get("fen"):
        return rec["fen"]
    from src.benchmarks.games.fen import fen_of_board
    from src.benchmarks.games.rules import Board

    pieces = {(int(p["sq"][1:]) - 1, ord(p["sq"][0]) - ord("a")): (p["color"], p["kind"])
              for p in rec["pieces"]}
    return fen_of_board(Board(rec["n"], pieces, rec["turn"]))


def _presentation(rec: Dict[str, object], variant: str) -> str:
    n = rec["n"]
    if variant == "fen":
        if n != 8:
            raise ValueError(f"FEN variant requires 8x8, got {n}x{n}")
        return f"The position in FEN notation: {_fen_of(rec)}"
    return ("Here is the current position (each cell shows the piece or '..'):\n"
            + _render_board(rec["pieces"], n))


def _turn_line(rec: Dict[str, object]) -> str:
    return f"It is {'White' if rec['turn'] == 'w' else 'Black'} to move."


def build_single_move_prompt(rec: Dict[str, object], condition: str,
                             variant: str = "grid") -> str:
    return "\n".join([
        f"You are playing chess on a {rec['n']}x{rec['n']} board.",
        RULES_SUMMARY.format(n=rec["n"]),
        _presentation(rec, variant),
        _turn_line(rec),
        OBJECTIVES[condition],
        OUTPUT_SPEC,
    ])


def build_mate1_prompt(rec: Dict[str, object], condition: str,
                       variant: str = "grid") -> str:
    return "\n".join([
        f"You are playing chess on a {rec['n']}x{rec['n']} board.",
        RULES_SUMMARY.format(n=rec["n"]),
        _presentation(rec, variant),
        _turn_line(rec),
        MATE1_OBJECTIVES[condition],
        OUTPUT_SPEC,
    ])


def build_mobility_prompt(rec: Dict[str, object], condition: str,
                          variant: str = "grid") -> str:
    return "\n".join([
        f"You are playing chess on a {rec['n']}x{rec['n']} board.",
        RULES_SUMMARY.format(n=rec["n"]),
        _presentation(rec, variant),
        _turn_line(rec),
        MOBILITY_OBJECTIVES[condition],
        OUTPUT_SPEC,
    ])


def build_cap_prompt(rec: Dict[str, object], condition: str,
                     variant: str = "grid") -> str:
    """Pure-legality probe: same board presentation, no objective pressure."""
    return "\n".join([
        f"You are playing chess on a {rec['n']}x{rec['n']} board.",
        RULES_SUMMARY.format(n=rec["n"]),
        _presentation(rec, variant),
        _turn_line(rec),
        CAP_OBJECTIVE,
        OUTPUT_SPEC,
    ])
