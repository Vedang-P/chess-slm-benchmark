"""Evaluate google-deepmind/searchless_chess (Ruoss et al.) on MATE-style
2-choice move selection (noexplain-1000).

Reuses the OFFICIAL engine builder (engines/constants.py) so the released
9M/136M/270M checkpoints load with their paper configs. For each MATE
position: score win-prob of candidate A vs B via ActionValueEngine, pick
higher. Answers: does a searchless chess model transfer to expert
2-choice tasks, and how does it compare to gemma-4-E2B's 58.1%?

Usage (Kaggle, from the repo root):
    python3 scripts/eval_searchless_mate.py \
        --model 270M --eval data/positions/mate-selection-test-noexplain.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# searchless_chess cloned at repo root: /kaggle/working/chess-slm-benchmark/searchless_chess
SL_ROOT = ROOT / "searchless_chess"
sys.path.insert(0, str(SL_ROOT))
sys.path.insert(0, str(SL_ROOT / "src"))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="270M",
                    choices=["9M", "136M", "270M"])
    ap.add_argument("--eval", required=True,
                    help="MATE-style eval json (noexplain format)")
    ap.add_argument("--checkpoint-dir", default="",
                    help="dir containing checkpoints/<model> (default: "
                         "repo root, matching the official layout)")
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()

    import numpy as np
    import chess

    # official builder: constants.ENGINE_BUILDERS['270M'] etc.
    import sys as _sys
    if args.checkpoint_dir:
        _sys.path.insert(0, args.checkpoint_dir)
    from engines import constants as sl_constants

    builder = sl_constants.ENGINE_BUILDERS[args.model]
    engine = builder()
    print(f"[sl] {args.model} engine loaded", flush=True)

    # return buckets for win-prob conversion
    from engines import neural_engines, engine as engine_lib
    import utils as sl_utils

    # ---- load MATE eval set ----
    rows = json.load(open(args.eval))
    if args.max_rows:
        rows = rows[:args.max_rows]
    print(f"[sl] {len(rows)} MATE positions", flush=True)

    n_correct = 0
    n_total = 0
    examples = []
    for i, row in enumerate(rows):
        fen = row.get("fen") or row.get("position")
        truth = row.get("truth_label") or row.get("label")
        ca = row.get("candidate_a") or row.get("move_a")
        cb = row.get("candidate_b") or row.get("move_b")
        if not fen or not ca or not cb:
            continue
        board = chess.Board(fen)
        try:
            analysis = engine.analyse(board)
        except Exception as e:
            print(f"[sl] analyse failed row {i}: {e}", flush=True)
            continue
        log_probs = analysis["log_probs"]
        probs = np.exp(log_probs)
        buckets = getattr(engine, "_return_buckets_values", None)
        if buckets is None:
            _, buckets = sl_utils.get_uniform_buckets_edges_values(128)
        win_probs = np.inner(probs, buckets)
        moves = engine_lib.get_ordered_legal_moves(board)
        win = dict(zip([m.uci() for m in moves], win_probs.tolist()))
        wa, wb = win.get(ca), win.get(cb)
        if wa is None or wb is None:
            continue
        pred = "A" if wa > wb else "B"
        n_total += 1
        if pred == truth:
            n_correct += 1
        if i < 5:
            examples.append({"fen": fen[:30], "truth": truth, "pred": pred,
                             "wa": round(wa, 4), "wb": round(wb, 4)})

    acc = n_correct / n_total if n_total else 0.0
    print(f"[sl] MATE 2-choice accuracy: {n_correct}/{n_total} = "
          f"{acc*100:.1f}%  (gemma base = 58.1%)", flush=True)
    print("[sl] first 5:", json.dumps(examples, indent=1), flush=True)


if __name__ == "__main__":
    main()
