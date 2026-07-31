"""Simplified chess rules for NxN boards (3x3, 5x5 primary; 8x8 puzzle tasks).

Deliberately a small, self-contained engine -- no python-chess dependency for
the tiny boards. Rule variant (documented in the paper):

- Standard piece movement (K, Q, R, B, N, P).
- Pawns: single-step forward, diagonal capture, promote to queen on the last
  rank. NO double-step, NO en passant.
- NO castling. NO check-permitting moves (standard legality).
- Kings never move into check; opponent king is never capturable (checkmate
  ends the game first).
- Terminal states: checkmate, stalemate, insufficient material (bare kings or
  K vs K+P with no progress is handled by the 3-fold-ish repetition cap in the
  oracle, not here).

Coordinate convention: (row, col), 0-indexed, row 0 = White's back rank.
Algebraic squares: file = chr(ord('a') + col), rank = str(row + 1) -- a1 is
(0, 0) from White's perspective.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

WHITE = "w"
BLACK = "b"
COLORS = (WHITE, BLACK)

PIECES = ("K", "Q", "R", "B", "N", "P")

Square = Tuple[int, int]

_KNIGHT_DELTAS = [
    (-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1),
]
_BISHOP_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
_ROOK_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_QUEEN_DIRS = _BISHOP_DIRS + _ROOK_DIRS
_KING_DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

PAWN_DIR = {WHITE: 1, BLACK: -1}  # White moves toward increasing row (rank 1 -> N)


@dataclass(frozen=True)
class Move:
    fr: Square
    to: Square
    piece: str
    captured: Optional[str] = None
    promote: bool = False

    @property
    def uci(self) -> str:
        return sq_to_algebraic(self.fr) + sq_to_algebraic(self.to)


def sq_to_algebraic(sq: Square) -> str:
    r, c = sq
    return chr(ord("a") + c) + str(r + 1)


def algebraic_to_sq(alg: str) -> Optional[Square]:
    alg = alg.strip().lower()
    if len(alg) < 2:
        return None
    col = ord(alg[0]) - ord("a")
    if not alg[1].isdigit():
        return None
    row = int(alg[1]) - 1
    if row < 0 or col < 0:
        return None
    return (row, col)


class Board:
    """Immutable-by-convention position: never mutate in place; use apply()."""

    __slots__ = ("n", "pieces", "turn")

    def __init__(self, n: int, pieces: Dict[Square, Tuple[str, str]], turn: str):
        assert n >= 3, "boards smaller than 3x3 make no sense"
        assert turn in COLORS
        self.n = n
        # pieces: {(row, col): (color, kind)}
        self.pieces = dict(pieces)
        self.turn = turn

    # ------------------------------------------------------------------ #
    # basics
    # ------------------------------------------------------------------ #
    def in_bounds(self, sq: Square) -> bool:
        r, c = sq
        return 0 <= r < self.n and 0 <= c < self.n

    def at(self, sq: Square) -> Optional[Tuple[str, str]]:
        return self.pieces.get(sq)

    def king_square(self, color: str) -> Optional[Square]:
        for sq, (c, kind) in self.pieces.items():
            if c == color and kind == "K":
                return sq
        return None

    def is_attacked(self, sq: Square, by_color: str) -> bool:
        """Is `sq` attacked by any piece of `by_color`? (No king-adjacency
        shortcut here: attack generation is per-piece below, and kings are
        checked explicitly so adjacent kings are handled correctly.)"""
        for (pr, pc), (c, kind) in self.pieces.items():
            if c != by_color:
                continue
            dr, dc = sq[0] - pr, sq[1] - pc
            if kind == "P":
                if dr == PAWN_DIR[c] and abs(dc) == 1:
                    return True
                continue
            if kind == "N":
                if (abs(dr), abs(dc)) in ((1, 2), (2, 1)):
                    return True
                continue
            if kind == "K":
                if max(abs(dr), abs(dc)) == 1:
                    return True
                continue
            # sliding pieces
            adr, adc = abs(dr), abs(dc)
            if kind == "R" and not (dr == 0 or dc == 0):
                continue
            if kind == "B" and adr != adc:
                continue
            if kind == "Q" and not (dr == 0 or dc == 0 or adr == adc):
                continue
            step_r = (dr > 0) - (dr < 0)
            step_c = (dc > 0) - (dc < 0)
            blocked = False
            r, c = pr + step_r, pc + step_c
            while (r, c) != sq:
                if not self.in_bounds((r, c)):
                    blocked = True
                    break
                if (r, c) in self.pieces:
                    blocked = True
                    break
                r += step_r
                c += step_c
            if not blocked:
                return True
        return False

    def in_check(self, color: str) -> bool:
        ks = self.king_square(color)
        if ks is None:
            return False  # king-less position (shouldn't happen in our sets)
        return self.is_attacked(ks, "w" if color == "b" else "b")

    # ------------------------------------------------------------------ #
    # move generation
    # ------------------------------------------------------------------ #
    def _sliding_targets(self, fr: Square, kind: str, dirs) -> List[Square]:
        out = []
        for dr, dc in dirs:
            r, c = fr[0] + dr, fr[1] + dc
            while self.in_bounds((r, c)):
                occ = self.at((r, c))
                if occ is None:
                    out.append((r, c))
                else:
                    if occ[0] != self.turn:
                        out.append((r, c))
                    break
                r += dr
                c += dc
        return out

    def _pseudo_legal(self) -> List[Move]:
        """All moves ignoring the own-king-in-check filter (capturing the
        enemy king is not allowed -- mate ends the game first)."""
        out = []
        for (r, c), (color, kind) in self.pieces.items():
            if color != self.turn:
                continue
            fr = (r, c)
            if kind == "P":
                fwd = PAWN_DIR[color]
                one = (r + fwd, c)
                if self.in_bounds(one) and self.at(one) is None:
                    if one[0] in (0, self.n - 1):
                        out.append(Move(fr, one, "P", promote=True))
                    else:
                        out.append(Move(fr, one, "P"))
                for dc in (-1, 1):
                    cap = (r + fwd, c + dc)
                    if self.in_bounds(cap):
                        occ = self.at(cap)
                        if occ is not None and occ[0] != color:
                            out.append(Move(fr, cap, "P", captured=occ[1]))
            elif kind == "N":
                for dr, dc in _KNIGHT_DELTAS:
                    to = (r + dr, c + dc)
                    if self.in_bounds(to):
                        occ = self.at(to)
                        if occ is None or occ[0] != color:
                            out.append(
                                Move(fr, to, "N", captured=occ[1] if occ else None)
                            )
            elif kind == "K":
                for dr, dc in _KING_DIRS:
                    to = (r + dr, c + dc)
                    if self.in_bounds(to):
                        occ = self.at(to)
                        if occ is None or occ[0] != color:
                            out.append(
                                Move(fr, to, "K", captured=occ[1] if occ else None)
                            )
            elif kind == "R":
                for to in self._sliding_targets(fr, "R", _ROOK_DIRS):
                    occ = self.at(to)
                    out.append(Move(fr, to, "R", captured=occ[1] if occ else None))
            elif kind == "B":
                for to in self._sliding_targets(fr, "B", _BISHOP_DIRS):
                    occ = self.at(to)
                    out.append(Move(fr, to, "B", captured=occ[1] if occ else None))
            elif kind == "Q":
                for to in self._sliding_targets(fr, "Q", _QUEEN_DIRS):
                    occ = self.at(to)
                    out.append(Move(fr, to, "Q", captured=occ[1] if occ else None))
        return out

    def legal_moves(self) -> List[Move]:
        out = []
        for m in self._pseudo_legal():
            if m.captured == "K":
                continue  # never generate king captures; mate ends the game first
            after = self.apply(m)
            if not after.in_check(self.turn):
                out.append(m)
        return out

    def has_legal_moves(self) -> bool:
        for m in self._pseudo_legal():
            if m.captured == "K":
                continue
            if not self.apply(m).in_check(self.turn):
                return True
        return False

    def apply(self, m: Move) -> "Board":
        pieces = dict(self.pieces)
        del pieces[m.fr]
        if m.promote:
            pieces[m.to] = (self.turn, "Q")
        else:
            pieces[m.to] = (self.turn, m.piece)
        return Board(self.n, pieces, "b" if self.turn == "w" else "w")

    # ------------------------------------------------------------------ #
    # terminal states
    # ------------------------------------------------------------------ #
    def outcome(self) -> Tuple[bool, Optional[str]]:
        """(is_terminal, winner). winner in {w, b, None=draw}."""
        if not self.has_legal_moves():
            if self.in_check(self.turn):
                return (True, "b" if self.turn == "w" else "w")
            return (True, None)  # stalemate
        if self._insufficient_material():
            return (True, None)
        return (False, None)

    def _insufficient_material(self) -> bool:
        nonkings = [(c, k) for (c, k) in self.pieces.values() if k != "K"]
        if len(nonkings) == 0:
            return True
        if len(nonkings) == 1:
            return nonkings[0][1] in ("B", "N")  # K+B or K+N vs K
        if len(nonkings) == 2:
            # K+B vs K+B (same-color bishops) is drawn; keep it simple: only
            # bare kings + single minor end as draws.
            return False
        return False

    # ------------------------------------------------------------------ #
    # rendering / serialization
    # ------------------------------------------------------------------ #
    def render(self) -> List[str]:
        """ASCII board, one string per row, for prompts. 'wQ' = white queen."""
        rows = []
        for r in range(self.n):
            row = []
            for c in range(self.n):
                occ = self.pieces.get((r, c))
                row.append(occ[0] + occ[1] if occ else "..")
            rows.append(" ".join(row))
        return rows

    def key(self) -> str:
        return ",".join(
            f"{sq_to_algebraic(sq)}:{c}{k}" for sq, (c, k) in sorted(self.pieces.items())
        ) + "|" + self.turn

    @classmethod
    def from_key(cls, key: str, n: int) -> "Board":
        placement, turn = key.split("|")
        pieces = {}
        for tok in placement.split(","):
            if not tok:
                continue
            sq_alg, color_kind = tok.split(":")
            pieces[algebraic_to_sq(sq_alg)] = (color_kind[0], color_kind[1])
        return cls(n, pieces, turn)

    def __repr__(self):
        return f"Board({self.n}x{self.n}, {self.turn} to move)"
