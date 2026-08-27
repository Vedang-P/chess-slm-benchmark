"""Evaluate searchless_chess (Ruoss et al.) on MATE 2-choice via the
dbest-isi HF fork's clean loader (modern jax/orbax, no PositionalSharding).

Loads the ORIGINAL DeepMind orbax checkpoints (9M/136M/270M) through the
fork's hf_model.py SearchlessChessModel, scores each MATE position by
win-prob(A) vs win-prob(B) (mean over the 128 return buckets), and
reports per-test-set accuracy.

Usage (Kaggle):
    python3 scripts/eval_searchless_mate.py \
        --model 270M --checkpoint /kaggle/working/checkpoints/270M/6400000 \
        --eval <comma-separated MATE jsons> \
        --sl-code /kaggle/working/chess-slm-benchmark/searchless_chess_code
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="270M", choices=["9M", "136M", "270M"])
    ap.add_argument("--checkpoint", required=True,
                    help="orbax checkpoint dir (…/270M/6400000)")
    ap.add_argument("--eval", required=True,
                    help="comma-separated MATE json paths")
    ap.add_argument("--sl-code", required=True,
                    help="dir with the fork's searchless_chess_code/")
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()

    import numpy as np
    import chess

    sys.path.insert(0, args.sl_code)
    from hf_model import SearchlessChessModel, SearchlessChessConfig

    # arch per model (from official constants.py)
    arch = {"9M": (256, 8, 8), "136M": (1024, 8, 8),
            "270M": (1024, 16, 8)}
    embedding_dim, num_layers, num_heads = arch[args.model]

    cfg = SearchlessChessConfig(
        vocab_size=1968, output_size=128,
        embedding_dim=embedding_dim, num_layers=num_layers,
        num_heads=num_heads, max_sequence_length=79,
        num_return_buckets=128, model_name=args.model)
    model = SearchlessChessModel(cfg)
    model.load_params(args.checkpoint)
    print(f"[sl] {args.model} loaded from {args.checkpoint}", flush=True)

    rows = []
    for path in args.eval.split(","):
        rows += json.load(open(path.strip()))
    if args.max_rows:
        rows = rows[:args.max_rows]
    print(f"[sl] {len(rows)} MATE positions", flush=True)

    n_correct = 0
    n_total = 0
    per_file = {}
    examples = []
    for i, row in enumerate(rows):
        _src = (row.get("source") or "pool")
        per_file.setdefault(_src, [0, 0])
        fen = row.get("fen") or row.get("position")
        truth = row.get("truth_label") or row.get("label")
        ca = row.get("candidate_a") or row.get("move_a")
        cb = row.get("candidate_b") or row.get("move_b")
        te = row.get("task_extra") or {}
        if not ca:
            ca = te.get("candidate_a")
        if not cb:
            cb = te.get("candidate_b")
        if not truth:
            truth = te.get("truth_label")
        if not fen or not ca or not cb:
            continue
        board = chess.Board(fen)
        try:
            result = model.predict(fen, temperature=1.0)
        except Exception as e:
            print(f"[sl] predict failed row {i}: {e}", flush=True)
            continue
        probs = result.get("action_probs")
        if probs is None:
            print(f"[sl] no action_probs row {i}", flush=True)
            continue
        # action_probs: per-action (1968) probability vector
        from utils import MOVE_TO_ACTION
        wa = wb = None
        for uci, idx in MOVE_TO_ACTION.items():
            if uci == ca:
                wa = float(probs[idx])
            if uci == cb:
                wb = float(probs[idx])
        if wa is None or wb is None:
            print(f"[sl] candidate not found row {i}: {ca} {cb}", flush=True)
            continue
        pred = "A" if wa > wb else "B"
        n_total += 1
        per_file[_src][1] += 1
        if pred == truth:
            n_correct += 1
            per_file[_src][0] += 1
        if i < 5:
            examples.append({"fen": fen[:30], "truth": truth, "pred": pred,
                             "wa": round(wa, 4), "wb": round(wb, 4)})

    acc = n_correct / n_total if n_total else 0.0
    print(f"[sl] MATE 2-choice accuracy: {n_correct}/{n_total} = "
          f"{acc*100:.1f}%  (gemma base = 58.1%)", flush=True)
    for src_name, (c, t) in sorted(per_file.items()):
        if t:
            print(f"[sl]   {src_name}: {c}/{t} = {c/t*100:.1f}%", flush=True)
    print("[sl] first 5:", json.dumps(examples, indent=1), flush=True)


if __name__ == "__main__":
    main()
