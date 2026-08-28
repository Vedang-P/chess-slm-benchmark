"""Build the puzzle-distribution SFT set from the Lichess puzzle DB.

Pipeline:
  1. Load parquet shards (Lichess/chess-puzzles HF mirror).
  2. Normalize FENs via python-chess; EXCLUDE FENs present in the official
     test set (puzzles.csv) — leakage guard.
  3. Dedupe positions, stratify-sample by rating (--n-positions).
  4. Enumerate ALL legal moves per position -> (tokens, action) rows,
     saved as an npz identical in shape to train_set.npz (no winprob yet;
     the teacher provides Q via teacher_label.py + --winprob-source teacher).

Usage (era venv, CPU):
    python3 scripts/build_puzzle_sft_set.py \
        --shards C:/tmp/sl_data/puzzles_db/train-*.parquet \
        --test C:/tmp/searchless_chess/data/puzzles.csv \
        --out C:/tmp/sl_data/puzzle_sft_set.npz --n-positions 50000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, help="glob of parquet shards")
    ap.add_argument("--test", required=True, help="official puzzles.csv (excluded)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-positions", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import chess
    import pandas as pd

    pat = Path(args.shards).expanduser()
    shards = sorted(pat.parent.glob(pat.name)) if "*" in pat.name else [pat]
    frames = []
    for s in shards:
        if s.suffix != ".parquet":
            continue
        df = pd.read_parquet(s, columns=["FEN", "Moves", "Rating"])
        frames.append(df)
        print(f"[sft] {s.name}: {len(df)} rows", flush=True)
    df = pd.concat(frames, ignore_index=True)
    print(f"[sft] total puzzles: {len(df)}", flush=True)

    # exclusion set: official test FENs, normalized
    test_df = pd.read_csv(args.test, usecols=["FEN"])
    excluded = set()
    for fen in test_df["FEN"]:
        try:
            excluded.add(chess.Board(fen).fen())
        except Exception:
            excluded.add(fen)
    print(f"[sft] test exclusion set: {len(excluded)} FENs", flush=True)

    seen = set()
    fens, moves, ratings = [], [], []
    for fen, mv, rating in zip(df["FEN"], df["Moves"], df["Rating"]):
        try:
            nf = chess.Board(fen).fen()
        except Exception:
            continue
        if nf in excluded or nf in seen:
            continue
        seen.add(nf)
        fens.append(nf)
        moves.append(mv)
        ratings.append(int(rating))
    print(f"[sft] unique non-test positions: {len(fens)}", flush=True)

    # stratify by rating deciles, sample n-positions
    rng = np.random.default_rng(args.seed)
    order = np.argsort(ratings)
    positions = []
    n = len(fens)
    per_bucket = max(1, args.n_positions // 10)
    for b in range(10):
        lo, hi = b * n // 10, (b + 1) * n // 10
        idx = order[lo:hi]
        if len(idx) > per_bucket:
            idx = rng.choice(idx, size=per_bucket, replace=False)
        positions.extend(int(i) for i in idx)
    print(f"[sft] selected positions: {len(positions)}", flush=True)

    rows_tok, rows_act = [], []
    sys.path.insert(0, str(Path(args.test).parents[2]))  # package root (C:/tmp)
    from searchless_chess.src import tokenizer, utils

    for i, pi in enumerate(positions):
        board = chess.Board(fens[pi])
        for mv in board.legal_moves:
            rows_tok.append(tokenizer.tokenize(board.fen()))
            rows_act.append(utils.MOVE_TO_ACTION[mv.uci()])
        if (i + 1) % 5000 == 0:
            print(f"[sft] {i+1}/{len(positions)} positions, "
                  f"{len(rows_tok)} rows", flush=True)
    tokens = np.asarray(rows_tok, dtype=np.uint8)
    actions = np.asarray(rows_act, dtype=np.uint16)
    np.savez_compressed(args.out, tokens=tokens, actions=actions)
    with open(Path(args.out).with_suffix(".json"), "w") as f:
        json.dump({"n_positions": len(positions), "n_rows": len(tokens),
                   "ratings": [ratings[p] for p in positions]}, f)
    print(f"[sft] saved -> {args.out} ({len(tokens)} rows)", flush=True)


if __name__ == "__main__":
    main()
