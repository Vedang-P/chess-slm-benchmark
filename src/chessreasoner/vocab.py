"""Fixed chess vocabulary for ChessReasoner.

Every chess entity is an *atomic* token reserved before BPE ever sees the
corpus, so BPE can neither split ``f3`` into ``f`` + ``3`` nor merge ``f3g5``
into one opaque unit. Surface forms are angle-bracketed (``<f3>``, ``<N>``) and
the generators guarantee prose never contains ``<``, so the two id spaces cannot
collide.

Layout (ids are stable; append only, never reorder):

    0    - 15    specials
    16   - 79    64 square tokens, ``chess.SQUARE_NAMES`` order (a1 -> h8)
    80   - 92    12 piece tokens + empty
    93   - 100   flags: side to move, yes/no, promotion pieces
    101  - ...   prose BPE, offset by ``CHESS_VOCAB_SIZE``
"""
from __future__ import annotations

import chess

# --- specials -------------------------------------------------------------
PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
FEN_BEGIN, FEN_END = "<FEN>", "</FEN>"
THINK_BEGIN, THINK_END = "<THINK>", "</THINK>"
LINE_BEGIN, LINE_END = "<LINE>", "</LINE>"
CAND_BEGIN, CAND_END = "<CAND>", "</CAND>"
MOVE_BEGIN, MOVE_END = "<MOVE>", "</MOVE>"
EVAL, NL = "<EVAL>", "<nl>"

SPECIALS = [
    PAD, BOS, EOS, UNK,
    FEN_BEGIN, FEN_END,
    THINK_BEGIN, THINK_END,
    LINE_BEGIN, LINE_END,
    CAND_BEGIN, CAND_END,
    MOVE_BEGIN, MOVE_END,
    EVAL, NL,
]

# --- board entities -------------------------------------------------------
SQUARE_TOKENS = [f"<{name}>" for name in chess.SQUARE_NAMES]  # a1 .. h8
assert len(SQUARE_TOKENS) == 64

PIECE_LETTERS = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
PIECE_TOKENS = [f"<{c}>" for c in PIECE_LETTERS]
EMPTY_TOKEN = "<.>"
BOARD_CONTENT_TOKENS = PIECE_TOKENS + [EMPTY_TOKEN]  # 13 classes for the board head

# --- flags ----------------------------------------------------------------
# Castling rights are emitted positionally (4 fixed slots: WK, WQ, BK, BQ), so
# each slot only needs yes/no. Using the letters K/Q/k/q here would collide with
# the piece tokens of the same name.
WHITE_TO_MOVE, BLACK_TO_MOVE = "<stm:w>", "<stm:b>"
YES, NO = "<yes>", "<no>"
PROMO_TOKENS = {
    chess.QUEEN: "<=q>", chess.ROOK: "<=r>",
    chess.BISHOP: "<=b>", chess.KNIGHT: "<=n>",
}
FLAG_TOKENS = [WHITE_TO_MOVE, BLACK_TO_MOVE, YES, NO] + list(PROMO_TOKENS.values())

CHESS_TOKENS = SPECIALS + SQUARE_TOKENS + BOARD_CONTENT_TOKENS + FLAG_TOKENS
CHESS_VOCAB_SIZE = len(CHESS_TOKENS)

CHESS_TOKEN_TO_ID = {tok: i for i, tok in enumerate(CHESS_TOKENS)}
assert len(CHESS_TOKEN_TO_ID) == len(CHESS_TOKENS), "duplicate chess token surface form"

SQUARE_TO_TOKEN = {sq: SQUARE_TOKENS[sq] for sq in chess.SQUARES}
TOKEN_TO_SQUARE = {tok: sq for sq, tok in SQUARE_TO_TOKEN.items()}

# Index of each board-content token within the 13-way board-head target.
BOARD_CLASS_INDEX = {tok: i for i, tok in enumerate(BOARD_CONTENT_TOKENS)}


def piece_token(piece: chess.Piece | None) -> str:
    """Board-content token for a square's occupant (``<.>`` when empty)."""
    if piece is None:
        return EMPTY_TOKEN
    return f"<{piece.symbol()}>"


def build_move_vocabulary() -> list[tuple[int, int, int | None]]:
    """Every ``(from, to, promotion)`` triple reachable by some piece.

    Derived rather than hardcoded so the policy head can never silently
    desynchronize from the move encoding. Returns 1968 entries for standard
    chess: all sliding and knight from-to pairs, plus the four promotion choices
    on each single-step or capturing pawn advance to the last rank.
    """
    moves: list[tuple[int, int, int | None]] = []
    seen = set()
    for frm in chess.SQUARES:
        fr, ff = chess.square_rank(frm), chess.square_file(frm)
        for to in chess.SQUARES:
            if frm == to:
                continue
            tr, tf = chess.square_rank(to), chess.square_file(to)
            dr, df = abs(tr - fr), abs(tf - ff)
            sliding = dr == 0 or df == 0 or dr == df
            knight = sorted((dr, df)) == [1, 2]
            if not (sliding or knight):
                continue
            key = (frm, to, None)
            if key not in seen:
                seen.add(key)
                moves.append(key)
            # promotions: a pawn advancing onto the last rank, straight or capturing
            if ((fr == 6 and tr == 7) or (fr == 1 and tr == 0)) and df <= 1 and dr == 1:
                for promo in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
                    key = (frm, to, promo)
                    if key not in seen:
                        seen.add(key)
                        moves.append(key)
    return moves


MOVE_VOCABULARY = build_move_vocabulary()
MOVE_VOCAB_SIZE = len(MOVE_VOCABULARY)
MOVE_TO_POLICY_INDEX = {m: i for i, m in enumerate(MOVE_VOCABULARY)}


def policy_index(move: chess.Move) -> int:
    """Index of ``move`` in the policy head's output space."""
    return MOVE_TO_POLICY_INDEX[(move.from_square, move.to_square, move.promotion)]
