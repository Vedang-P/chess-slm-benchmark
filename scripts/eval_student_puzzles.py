"""Evaluate a student checkpoint on the OFFICIAL puzzle benchmark.

Mirrors puzzles.py exactly (same dataset, same protocol, same Engine
interface) but loads a student checkpoint from the train_student.py format.

Usage (WSL GPU venv):
    python3 scripts/eval_student_puzzles.py --ckpt results/student/step-15000 \
        --data C:/tmp/searchless_chess/data/puzzles.csv --num-puzzles 10000
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="base path of step-N (no extension)")
    ap.add_argument("--data", required=True, help="official puzzles.csv")
    ap.add_argument("--num-puzzles", type=int, default=10000)
    ap.add_argument("--sl-repo", default="", help="official repo root")
    args = ap.parse_args()

    import os

    ckpt = Path(args.ckpt)
    sl_repo = Path(args.sl_repo or os.environ.get("SL_REPO", "C:/tmp/searchless_chess"))
    sys.path.insert(0, str(sl_repo.parent))

    import jax
    import pandas as pd
    from searchless_chess.src import transformer, utils
    from searchless_chess.src.engines import neural_engines
    from searchless_chess.src.puzzles import evaluate_puzzle_from_pandas_row

    with open(f"{ckpt}.config.json") as f:
        c = json.load(f)
    with open(f"{ckpt}.treedef.pkl", "rb") as f:
        treedef = pickle.load(f)
    data = np.load(f"{ckpt}.npz")
    leaves = [data[k] for k in sorted(data.files, key=lambda x: int(x.split("_")[1]))] \
        if all("_" in k for k in data.files) else [data[k] for k in data.files]
    leaves = [np.asarray(l, dtype=np.float32) for l in leaves]
    params = jax.tree_util.tree_unflatten(treedef, leaves)

    cfg = transformer.TransformerConfig(
        vocab_size=utils.NUM_ACTIONS,
        output_size=128,
        pos_encodings=transformer.PositionalEncodings.LEARNED,
        max_sequence_length=79,
        num_heads=c["heads"],
        num_layers=c["layers"],
        embedding_dim=c["dim"],
        apply_post_ln=True,
        apply_qk_layernorm=False,
        use_causal_mask=False,
    )
    predictor = transformer.build_transformer_predictor(cfg)
    _, bucket_vals = utils.get_uniform_buckets_edges_values(128)
    engine = neural_engines.ActionValueEngine(
        return_buckets_values=np.asarray(bucket_vals, dtype=np.float32),
        predict_fn=neural_engines.wrap_predict_fn(predictor, params))
    print(f"[eval] student step-{c['step']} (dim={c['dim']} L={c['layers']})",
          flush=True)

    puzzles = pd.read_csv(args.data, nrows=args.num_puzzles)
    correct = 0
    for i, row in puzzles.iterrows():
        ok = evaluate_puzzle_from_pandas_row(puzzle=row, engine=engine)
        correct += int(ok)
        if (i + 1) % 500 == 0:
            print(f"[eval] {i+1}/{len(puzzles)} "
                  f"acc={correct/(i+1)*100:.1f}%", flush=True)
    print(f"[eval] FINAL: {correct}/{len(puzzles)} = "
          f"{correct/len(puzzles)*100:.1f}%  (9M ref: 86.1 ours / 88.9 paper)",
          flush=True)


if __name__ == "__main__":
    main()
