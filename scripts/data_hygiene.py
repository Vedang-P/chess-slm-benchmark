"""Position-level exclusions shared by training, augmentation, and evaluation."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np

EMPTY = 30


def position_tokens(tokens):
    """Ignore clocks; canonicalize the unordered FEN castling-rights field."""
    out = np.asarray(tokens[:, :71], dtype=np.uint8).copy()
    out[:, 65:69].sort(axis=1)
    return out


def reflected_positions(tokens):
    out = position_tokens(tokens)
    safe = np.all(out[:, 65:69] == EMPTY, axis=1)
    out[safe, 1:65] = out[safe, 1:65].reshape(-1, 8, 8)[:, :, ::-1].reshape(-1, 64)
    ep = out[:, 69]
    mask = safe & (ep >= 10) & (ep <= 17)
    out[mask, 69] = 27 - ep[mask]
    return out


def position_hashes(tokens):
    """Stable 64-bit hash of the reflection orbit, not the action or clocks."""
    def hash_rows(rows):
        h = np.full(len(rows), np.uint64(1469598103934665603), dtype=np.uint64)
        for col in range(rows.shape[1]):
            h ^= rows[:, col].astype(np.uint64)
            h *= np.uint64(1099511628211)
        return h
    normal = position_tokens(tokens)
    return np.minimum(hash_rows(normal), hash_rows(reflected_positions(normal)))


def eligible_mask(tokens, excluded_hashes, modulus, fold, chunk_size=250_000):
    """Bounded-memory, global train mask; hash collisions exclude conservatively."""
    result = np.empty(len(tokens), dtype=bool)
    for start in range(0, len(tokens), chunk_size):
        hashes = position_hashes(tokens[start:start + chunk_size])
        result[start:start + chunk_size] = ((hashes % modulus != fold)
                                           & ~np.isin(hashes, excluded_hashes))
    return result


def load_exclusions(mate_dir: Path, puzzles: Path):
    """Exclude all MATE positions and all official puzzle decision positions."""
    import chess
    import chess.pgn
    import pandas as pd
    from scripts.eval_gavn import tokenize_fen

    paths = [mate_dir / name for name in (
        'mate-selection-test.json', 'mate-selection-test-noexplain.json',
        'mate-selection-test-tactic.json', 'mate-selection-test-both.json')]
    digest = hashlib.sha256()
    tokens = []
    for path in paths:
        raw = path.read_bytes(); digest.update(raw)
        rows = json.loads(raw)
        if len(rows) != 1000:
            raise ValueError(f'{path}: expected exactly 1000 MATE rows')
        for row in rows:
            tokens.append(tokenize_fen(row.get('fen') or row['position']))
    digest.update(puzzles.read_bytes())
    table = pd.read_csv(puzzles)
    if len(table) != 10000:
        raise ValueError('exclusions require the exact official 10,000-puzzle file')
    for _, row in table.iterrows():
        game = chess.pgn.read_game(io.StringIO(row['PGN']))
        if game is None or game.errors:
            raise ValueError('invalid official puzzle PGN')
        board = game.end().board()
        tokens.append(tokenize_fen(board.fen()))
        for uci in row['Moves'].split():
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                raise ValueError(f'illegal official puzzle move: {uci}')
            board.push(move)
            tokens.append(tokenize_fen(board.fen()))
    hashes = np.unique(position_hashes(np.asarray(tokens)))
    return hashes, digest.hexdigest()


def set_learning_rate(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr
