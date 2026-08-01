"""Game environments for full-game playouts (model vs random opponent).

Three environments, all text-driven:
- Chess5x5   : our NxN engine (5x5), model plays White, algebraic from-to moves
- TicTacToe  : 3x3, model is X, single-square moves (e.g. "b2")
- Connect4   : 6x7 drop game, model is first player, column moves ("3")

Each env exposes the same interface so the playout loop in run_chess.py is
game-agnostic: start/over/legal_moves/apply/random_move/prompt/parse.
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

from src.benchmarks.games.rules import Board, algebraic_to_sq


def _sg(sq) -> str:
    return f"{chr(ord('a') + sq[1])}{sq[0] + 1}"


class Chess5x5:
    name = "playout-5x5"
    n = 5

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.start_board = Board(5, {
            (0, 0): ("w", "K"), (0, 1): ("w", "Q"), (0, 2): ("w", "B"), (0, 3): ("w", "B"),
            (1, 0): ("w", "P"), (1, 1): ("w", "P"), (1, 2): ("w", "P"),
            (4, 4): ("b", "K"), (4, 3): ("b", "Q"), (4, 2): ("b", "R"),
            (4, 1): ("b", "P"), (4, 0): ("b", "P"), (3, 4): ("b", "P"),
        }, "w")

    def start(self):
        return self.start_board

    def over(self, board: Board) -> Tuple[bool, Optional[str]]:
        terminal, winner = board.outcome()
        if not terminal:
            return (False, None)
        return (True, "model" if winner == "w" else "opp" if winner == "b" else None)

    def legal_moves(self, board: Board) -> List[str]:
        return [m.uci for m in board.legal_moves()]

    def apply(self, board: Board, move: str) -> Board:
        m = next(m for m in board.legal_moves() if m.uci == move)
        return board.apply(m)

    def random_move(self, board: Board) -> str:
        return self.rng.choice(self.legal_moves(board))

    def prompt(self, board: Board) -> str:
        rows = []
        for r in range(4, -1, -1):
            row = []
            for c in range(5):
                occ = board.at((r, c))
                row.append(f"{occ[0]}{occ[1]}" if occ else "..")
            rows.append(f"{r + 1:2d} " + " ".join(row))
        return (
            "You are playing chess on a 5x5 board. Standard chess rules apply "
            "(pawns move forward and capture diagonally, promote to queen on the last "
            "rank; no castling, no en passant). You are WHITE. Current position:\n"
            + "\n".join(rows)
            + "\nIt is White to move. Play the strongest move you can find.\n"
            + "Output ONLY a single line: MOVE: <from><to> in algebraic notation "
              "(e.g. 'MOVE: e2e4')."
        )

    def parse(self, text: str) -> Optional[str]:
        import re

        m = re.search(r"([a-e][1-5])\s*[- ]?\s*([a-e][1-5])", text or "")
        if not m:
            return None
        fr, to = algebraic_to_sq(m.group(1)), algebraic_to_sq(m.group(2))
        if fr is None or to is None:
            return None
        return _sg(fr) + _sg(to)


class TicTacToe:
    name = "ttt"
    n = 3
    EMPTY, X, O = 0, 1, 2

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def start(self):
        return [[self.EMPTY] * 3 for _ in range(3)]

    def _winner(self, b) -> Optional[int]:
        lines = [b[r] for r in range(3)] + [[b[r][c] for r in range(3)] for c in range(3)] \
            + [[b[0][0], b[1][1], b[2][2]], [b[0][2], b[1][1], b[2][0]]]
        for ln in lines:
            if ln[0] != self.EMPTY and ln[0] == ln[1] == ln[2]:
                return ln[0]
        return None

    def over(self, b) -> Tuple[bool, Optional[str]]:
        w = self._winner(b)
        if w:
            return (True, "model" if w == self.X else "opp")
        if all(x != self.EMPTY for row in b for x in row):
            return (True, None)
        return (False, None)

    def legal_moves(self, b) -> List[str]:
        return [_sg((r, c)) for r in range(3) for c in range(3) if b[r][c] == self.EMPTY]

    def apply(self, b, move: str) -> List[List[int]]:
        sq = algebraic_to_sq(move)
        nb = [row[:] for row in b]
        nb[sq[0]][sq[1]] = self.X if sum(x != self.EMPTY for row in b for x in row) % 2 == 0 else self.O
        return nb

    def random_move(self, b) -> str:
        return self.rng.choice(self.legal_moves(b))

    def prompt(self, b) -> str:
        glyph = {self.EMPTY: ".", self.X: "X", self.O: "O"}
        rows = []
        for r in range(2, -1, -1):
            rows.append(f"{r + 1} " + " ".join(glyph[x] for x in b[r]))
        return (
            "You are playing tic-tac-toe on a 3x3 board. You are X and it is your turn. "
            "Board (row 1 is the top row):\n   a b c\n" + "\n".join(rows)
            + "\nPlay one move. Output ONLY: MOVE: <square> (e.g. 'MOVE: b2')."
        )

    def parse(self, text: str) -> Optional[str]:
        import re

        m = re.search(r"([a-c][1-3])", text or "")
        return m.group(1) if m else None


class Connect4:
    name = "c4"
    ROWS, COLS = 6, 7
    EMPTY, P1, P2 = 0, 1, 2

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def start(self):
        return [[self.EMPTY] * self.COLS for _ in range(self.ROWS)]

    def _winner(self, b) -> Optional[int]:
        for r in range(self.ROWS):
            for c in range(self.COLS):
                v = b[r][c]
                if v == self.EMPTY:
                    continue
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    if all(0 <= r + dr * k < self.ROWS and 0 <= c + dc * k < self.COLS
                           and b[r + dr * k][c + dc * k] == v for k in range(4)):
                        return v
        return None

    def over(self, b) -> Tuple[bool, Optional[str]]:
        w = self._winner(b)
        if w:
            return (True, "model" if w == self.P1 else "opp")
        if all(b[0][c] != self.EMPTY for c in range(self.COLS)):
            return (True, None)
        return (False, None)

    def legal_moves(self, b) -> List[str]:
        return [str(c + 1) for c in range(self.COLS) if b[0][c] == self.EMPTY]

    def apply(self, b, move: str) -> List[List[int]]:
        col = int(move) - 1
        piece = self.P1 if sum(x != self.EMPTY for row in b for x in row) % 2 == 0 else self.P2
        nb = [row[:] for row in b]
        for r in range(self.ROWS - 1, -1, -1):
            if nb[r][col] == self.EMPTY:
                nb[r][col] = piece
                break
        return nb

    def random_move(self, b) -> str:
        return self.rng.choice(self.legal_moves(b))

    def prompt(self, b) -> str:
        glyph = {self.EMPTY: ".", self.P1: "X", self.P2: "O"}
        rows = ["   " + " ".join(str(c + 1) for c in range(self.COLS))]
        for r in range(self.ROWS):
            rows.append(f"{r + 1:2d} " + " ".join(glyph[x] for x in b[r]))
        return (
            "You are playing Connect Four. You are X (first player) and it is your turn. "
            "Board (top row is row 1):\n" + "\n".join(rows)
            + "\nPlay one move by dropping a piece into a column. "
              "Output ONLY: MOVE: <column> (e.g. 'MOVE: 3')."
        )

    def parse(self, text: str) -> Optional[str]:
        import re

        m = re.search(r"\b([1-7])\b", text or "")
        return m.group(1) if m else None


ENVS = {env.name: env for env in (Chess5x5(), TicTacToe(), Connect4())}
