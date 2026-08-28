"""Parse the ChessBench test action-value bag into a compact training set.

Output (npz, saved next to the bag):
  tokens   uint8  [N, 77]   tokenized FEN (official tokenizer)
  actions  uint16 [N]       action index (official MOVE_TO_ACTION)
  winprob  float32 [N]      Stockfish 16 win probability label

Usage (era venv, CPU is fine):
    python3 scripts/build_student_train_set.py \
        --bag C:/tmp/searchless_chess/data/test/action_value_data.bag
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="action_value_data.bag path")
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--out", default="", help="output npz (default: <bag dir>/train_set.npz)")
    args = ap.parse_args()

    import numpy as np

    # package root is the parent of the searchless_chess repo dir
    sys.path.insert(0, str(Path(args.bag).parents[3]))
    from searchless_chess.src import bagz, constants, tokenizer, utils

    reader = bagz.BagFileReader(args.bag)
    n = len(reader) if not args.max_records else min(len(reader), args.max_records)
    print(f"[parse] {len(reader)} records; reading {n}", flush=True)

    tokens = np.zeros((n, 77), dtype=np.uint8)
    actions = np.zeros(n, dtype=np.uint16)
    winprob = np.zeros(n, dtype=np.float32)
    coder = constants.CODERS["action_value"]
    bad = 0
    for i in range(n):
        try:
            fen, move, wp = coder.decode(reader[i])
            tokens[i] = tokenizer.tokenize(fen)
            actions[i] = utils.MOVE_TO_ACTION[move]
            winprob[i] = float(wp)
        except Exception:
            bad += 1
            continue
        if (i + 1) % 200000 == 0:
            print(f"[parse] {i+1}/{n} (bad={bad})", flush=True)
    print(f"[parse] done: {n - bad} good, {bad} bad", flush=True)

    out = args.out or str(Path(args.bag).parent / "train_set.npz")
    np.savez_compressed(out, tokens=tokens, actions=actions, winprob=winprob)
    print(f"[parse] saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
