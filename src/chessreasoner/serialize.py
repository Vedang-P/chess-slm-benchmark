"""Board and move serialization.

The board is emitted as a **fixed-length 72-token span**:

    <FEN> [64 square-content tokens, a1 -> h8] [stm] [WK][WQ][BK][BQ] [ep] </FEN>

Fixed length is the whole point. Square ``i`` always sits at offset ``1 + i``
from ``<FEN>``, so file (+1), rank (+8) and diagonal (+9 / +7) neighbours are at
constant relative offsets and attention can learn them as strides. Run-length
FEN (``2r2r2/2qbbppk/...``) makes every square's position depend on decoding the
digits before it, which is the representation the Gemma baseline had to work
from.
"""
from __future__ import annotations

import chess

from .vocab import (
    BLACK_TO_MOVE,
    FEN_BEGIN,
    FEN_END,
    MOVE_BEGIN,
    MOVE_END,
    NO,
    PROMO_TOKENS,
    SQUARE_TO_TOKEN,
    TOKEN_TO_SQUARE,
    WHITE_TO_MOVE,
    YES,
    piece_token,
)

BOARD_SPAN_LEN = 72
"""Every board span is exactly this long: 1 + 64 + 1 + 4 + 1 + 1."""

_CASTLING_SLOTS = (
    (chess.WHITE, chess.BB_H1),  # White king-side
    (chess.WHITE, chess.BB_A1),  # White queen-side
    (chess.BLACK, chess.BB_H8),  # Black king-side
    (chess.BLACK, chess.BB_A8),  # Black queen-side
)


def board_to_parts(board: chess.Board) -> list[str]:
    """Serialize a position to its fixed 72-token span."""
    parts = [FEN_BEGIN]
    for sq in chess.SQUARES:  # chess.SQUARES is a1..h8, matching SQUARE_TOKENS
        parts.append(piece_token(board.piece_at(sq)))
    parts.append(WHITE_TO_MOVE if board.turn == chess.WHITE else BLACK_TO_MOVE)
    for color, rook_bb in _CASTLING_SLOTS:
        parts.append(YES if board.castling_rights & rook_bb else NO)
    parts.append(SQUARE_TO_TOKEN[board.ep_square] if board.ep_square is not None else NO)
    parts.append(FEN_END)
    assert len(parts) == BOARD_SPAN_LEN, f"board span was {len(parts)}, expected {BOARD_SPAN_LEN}"
    return parts


def parts_to_board(parts: list[str]) -> chess.Board:
    """Inverse of :func:`board_to_parts`.

    Halfmove and fullmove clocks are not carried in the span, so they come back
    as 0 and 1. Everything that affects legality -- occupancy, side to move,
    castling rights, en-passant square -- round-trips exactly.
    """
    if len(parts) != BOARD_SPAN_LEN or parts[0] != FEN_BEGIN or parts[-1] != FEN_END:
        raise ValueError("not a board span")
    board = chess.Board(None)  # empty board, no pieces, no castling rights
    for i, sq in enumerate(chess.SQUARES):
        tok = parts[1 + i]
        if tok == "<.>":
            continue
        board.set_piece_at(sq, chess.Piece.from_symbol(tok[1:-1]))
    board.turn = chess.WHITE if parts[65] == WHITE_TO_MOVE else chess.BLACK
    rights = chess.BB_EMPTY
    for slot, (_color, rook_bb) in enumerate(_CASTLING_SLOTS):
        if parts[66 + slot] == YES:
            rights |= rook_bb
    board.castling_rights = rights
    ep = parts[70]
    board.ep_square = None if ep == NO else TOKEN_TO_SQUARE[ep]
    return board


def move_to_parts(move: chess.Move, wrap: bool = False) -> list[str]:
    """``(from, to[, promo])`` -- 2 or 3 tokens, optionally in a ``<MOVE>`` span.

    A move can never be confused with a single square: ``f3g5`` is
    ``[<f3>, <g5>]``, not a subword sequence that starts with a valid square.
    """
    parts = [SQUARE_TO_TOKEN[move.from_square], SQUARE_TO_TOKEN[move.to_square]]
    if move.promotion is not None:
        parts.append(PROMO_TOKENS[move.promotion])
    return [MOVE_BEGIN, *parts, MOVE_END] if wrap else parts


def parts_to_move(parts: list[str]) -> chess.Move:
    """Inverse of :func:`move_to_parts`, accepting the wrapped or bare form."""
    if parts and parts[0] == MOVE_BEGIN:
        parts = parts[1:-1]
    if len(parts) not in (2, 3):
        raise ValueError(f"expected 2 or 3 move tokens, got {len(parts)}")
    promo = None
    if len(parts) == 3:
        promo = next(p for p, tok in PROMO_TOKENS.items() if tok == parts[2])
    return chess.Move(TOKEN_TO_SQUARE[parts[0]], TOKEN_TO_SQUARE[parts[1]], promotion=promo)
