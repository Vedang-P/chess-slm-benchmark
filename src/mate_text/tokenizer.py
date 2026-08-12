"""Tokenizer + vocab for the MATE text transformer.

Two input views are fused in the model:
  1. board tokens — 64 squares, each a piece token (or EMPTY), plus
     side-to-move and castling/en-passant metadata tokens
  2. text tokens — MoveA:<uci> / MoveB:<uci>, task framing, answer

The MATE selection task is BINARY, so the model ends with a 2-way head
over the final hidden state (MoveA vs MoveB). This tokenizer produces
token-id sequences for board + text, and the text is also kept for the
text tower's embedding.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import chess

# --- vocabulary -------------------------------------------------------------

PIECE_TOKENS = {
    None: 0,  # empty square
    chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3,
    chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6,
}

SIDE_TO_MOVE = {chess.WHITE: 7, chess.BLACK: 8}

SPECIAL = {
    "<pad>": 9,
    "<cls>": 10,
    "<sep>": 11,
    "<mside>": 12,   # marker: side-to-move follows
    "<castle>": 13,  # marker: castling string follows
    "<ep>": 14,      # marker: en-passant square follows
    "<text>": 15,    # marker: candidate-move text follows
    "<ansA>": 16,    # marker: answer is MoveA
    "<ansB>": 17,    # marker: answer is MoveB
    "<unk>": 18,
}

# text tokens: "MoveA", "MoveB", ":", a-h, 1-8, piece chars for promotion
TEXT_BASE = 19


def _text_vocab() -> dict[str, int]:
    v = {}
    n = TEXT_BASE
    for tok in ("MoveA", "MoveB", ":"):
        v[tok] = n
        n += 1
    for ch in "abcdefgh":
        v[ch] = n
        n += 1
    for ch in "12345678":
        v[ch] = n
        n += 1
    for ch in "qrbn":
        v[ch] = n
        n += 1
    return v


def _special_id(name: str) -> int:
    return SPECIAL[name]


@dataclass
class MateTokenizer:
    vocab_size: int = field(default_factory=lambda: 0)
    text_tokens: dict[str, int] = field(default_factory=_text_vocab)

    def __post_init__(self):
        n = TEXT_BASE
        for tok in ("MoveA", "MoveB", ":"):
            n += 1
        for ch in "abcdefgh12345678qrbn":
            n += 1
        self.vocab_size = n

    # -- board -----------------------------------------------------------------

    def board_ids(self, board: chess.Board) -> list[int]:
        ids = []
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p is None:
                ids.append(PIECE_TOKENS[None])
            else:
                # color-agnostic: black pieces offset to disambiguate
                base = PIECE_TOKENS[p.piece_type]
                if p.color == chess.BLACK:
                    base += 6
                ids.append(base)
        ids.append(_special_id("<mside>"))
        ids.append(SIDE_TO_MOVE[board.turn])
        castle = (("K" if board.has_kingside_castling_rights(chess.WHITE) else "")
                  + ("Q" if board.has_queenside_castling_rights(chess.WHITE) else "")
                  + ("k" if board.has_kingside_castling_rights(chess.BLACK) else "")
                  + ("q" if board.has_queenside_castling_rights(chess.BLACK) else "")
                  or "-")
        ids.append(_special_id("<castle>"))
        ids.extend(self._castle_ids(castle))
        ep = board.ep_square
        ids.append(_special_id("<ep>"))
        if ep is not None:
            ids.extend(self._square_ids(chess.square_name(ep)))
        else:
            ids.append(self._char_id("-"))
        return ids

    def _castle_ids(self, castle: str) -> list[int]:
        return [self._char_id(c) for c in castle]

    def _square_ids(self, square: str) -> list[int]:
        return [self._char_id(c) for c in square]

    def _char_id(self, ch: str) -> int:
        if ch in self.text_tokens:
            return self.text_tokens[ch]
        return _special_id("<unk>")

    # -- text -----------------------------------------------------------------

    def text_ids(self, candidate_a: str, candidate_b: str) -> list[int]:
        ids = [_special_id("<text>")]
        ids.append(self.text_tokens["MoveA"])
        ids.append(self.text_tokens[":"])
        ids.extend(self._uci_ids(candidate_a))
        ids.append(self.text_tokens["MoveB"])
        ids.append(self.text_tokens[":"])
        ids.extend(self._uci_ids(candidate_b))
        return ids

    def _uci_ids(self, uci: str) -> list[int]:
        return [self._char_id(c) for c in uci]

    def answer_id(self, truth: str) -> int:
        return _special_id("<ansA>" if truth == "A" else "<ansB>")

    def answer_token_count(self) -> int:
        return 2

    def special_pad(self) -> int:
        return _special_id("<pad>")

    # -- serialization ----------------------------------------------------------

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"vocab_size": self.vocab_size,
                       "text_tokens": self.text_tokens}, f)

    @classmethod
    def load(cls, path: str) -> "MateTokenizer":
        with open(path) as f:
            d = json.load(f)
        tok = cls()
        tok.vocab_size = d["vocab_size"]
        tok.text_tokens = d["text_tokens"]
        return tok
