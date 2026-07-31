"""Faithful prompt construction for the anti-goal benchmark.

Design principles (documented in the paper):
- The WIN and LOSE conditions differ ONLY in the objective sentence.
- Every prompt explains the board format, the rule variant, and the exact
  output format -- a model that fails is failing at reasoning, not at
  understanding the interface.
- Output is a single line "MOVE: <from><to>" in algebraic notation.
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
    "win": (
        "YOUR OBJECTIVE: WIN. Deliver CHECKMATE in exactly one move."
    ),
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

OUTPUT_SPEC = (
    "Output ONLY a single line of the form: MOVE: <from><to> using algebraic "
    "notation (e.g. 'MOVE: e2e4'). Do not output anything else."
)


def _render_board(pieces: List[Dict[str, str]], n: int) -> str:
    """Algebraic grid, one row per rank, piece or '.' per file."""
    grid = {f"{p['sq']}": p for p in pieces}
    lines = []
    header = "   " + " ".join(chr(ord("a") + c) for c in range(n))
    lines.append(header)
    for r in range(n - 1, -1, -1):
        row = []
        for c in range(n):
            sq = f"{chr(ord('a') + c)}{r + 1}"
            p = grid.get(sq)
            row.append(f"{p['color']}{p['kind']}" if p else "..")
        lines.append(f"{r + 1:2d} " + " ".join(row))
    return "\n".join(lines)


def build_single_move_prompt(rec: Dict[str, object], condition: str) -> str:
    return "\n".join([
        f"You are playing chess on a {rec['n']}x{rec['n']} board.",
        RULES_SUMMARY.format(n=rec["n"]),
        "Here is the current position (each cell shows the piece or '..'):",
        _render_board(rec["pieces"], rec["n"]),
        f"It is {'White' if rec['turn'] == 'w' else 'Black'} to move.",
        OBJECTIVES[condition],
        OUTPUT_SPEC,
    ])


def build_mate1_prompt(rec: Dict[str, object], condition: str) -> str:
    return "\n".join([
        f"You are playing chess on a {rec['n']}x{rec['n']} board.",
        RULES_SUMMARY.format(n=rec["n"]),
        "Here is the current position (each cell shows the piece or '..'):",
        _render_board(rec["pieces"], rec["n"]),
        f"It is {'White' if rec['turn'] == 'w' else 'Black'} to move.",
        MATE1_OBJECTIVES[condition],
        OUTPUT_SPEC,
    ])


def build_mobility_prompt(rec: Dict[str, object], condition: str) -> str:
    return "\n".join([
        f"You are playing chess on a {rec['n']}x{rec['n']} board.",
        RULES_SUMMARY.format(n=rec["n"]),
        "Here is the current position (each cell shows the piece or '..'):",
        _render_board(rec["pieces"], rec["n"]),
        f"It is {'White' if rec['turn'] == 'w' else 'Black'} to move.",
        MOBILITY_OBJECTIVES[condition],
        OUTPUT_SPEC,
    ])
