"""MATE 2-choice eval for searchless_chess action-value models.

Implements the protocol that produced the Mac's 9M=98.2% noexplain-1000
result: the official ActionValueEngine semantics (per-action 128-bucket
return distribution; win-prob = <p, bucket_values>; compare the two
candidates' win-probs; higher wins).

Loads the ORIGINAL DeepMind orbax-ocdbt checkpoints through the dbest-isi
fork modules (hf_model.py + era orbax 0.5.5 API) — same weights, same
restore mechanism. On Windows, orbax/tensorstore build ocdbt kvstore base
URLs as "file://C:\\..." (drive letter parsed as URL host) or read
leading-slash paths as UNC; both are patched to the valid "file:///C:/..."
form at the final spec choke points.

Usage (era-stack venv: jax 0.4.35, orbax 0.5.5):
    python3 scripts/sl_mate_eval.py \
        --model 9M --checkpoint <root>/9M/6400000/params \
        --sl-code <fork searchless_chess_code dir> \
        --eval data/positions/mate-selection-test.json,<...>,<...> \
        [--max-rows N] [--save results/sl-mate-9M.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ARCH = {"9M": (256, 8, 8), "136M": (1024, 8, 8), "270M": (1024, 16, 8)}


def _fix_base(spec):
    """Rewrite ocdbt kvstore base 'file://C:\\...' -> 'file:///C:/...'."""
    kv = spec.get("kvstore")
    if (isinstance(kv, dict) and isinstance(kv.get("base"), str)
            and re.match(r"file://[A-Za-z]:", kv["base"])):
        kv["base"] = "file:///" + kv["base"][7:].replace("\\", "/")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["9M", "136M", "270M"])
    ap.add_argument("--checkpoint", required=True,
                    help="orbax checkpoint dir (e.g. .../9M/6400000/params)")
    ap.add_argument("--eval", required=True,
                    help="comma-separated MATE json paths")
    ap.add_argument("--sl-code", required=True,
                    help="dir with the fork's searchless_chess_code/")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    import numpy as np
    import chess

    sys.path.insert(0, args.sl_code)
    import tokenizer as sl_tokenizer
    import utils as sl_utils
    from hf_model import SearchlessChessModel, SearchlessChessConfig

    # --- Windows ocdbt spec patches (no-op on POSIX) ---
    import orbax.checkpoint as ocp
    import orbax.checkpoint.type_handlers as th
    import jax as _jax
    import jax.experimental.array_serialization.serialization as _ser

    _orig_ts_spec = th.get_tensorstore_spec

    def _win_ts_spec(directory, *a, **k):
        spec = _orig_ts_spec(directory, *a, **k)
        _fix_base(spec)
        return spec

    th.get_tensorstore_spec = _win_ts_spec

    _orig_ad = _ser.async_deserialize

    def _win_ad(sharding, tspec, **kw):
        _fix_base(tspec)
        return _orig_ad(sharding, tspec, **kw)

    _ser.async_deserialize = _win_ad

    # Leading-slash paths are read as UNC network paths by tensorstore on
    # Windows — normalize to a drive-letter absolute path first.
    args.checkpoint = os.path.abspath(args.checkpoint)

    embedding_dim, num_layers, num_heads = ARCH[args.model]
    cfg = SearchlessChessConfig(
        vocab_size=1968, output_size=128,
        embedding_dim=embedding_dim, num_layers=num_layers,
        num_heads=num_heads, max_sequence_length=79,
        num_return_buckets=128, model_name=args.model)
    model = SearchlessChessModel(cfg)
    dummy = model.predictor.initial_params(
        rng=_jax.random.PRNGKey(0),
        targets=np.ones((1, 1), dtype=np.uint32))
    _ckptr = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    model.params = _ckptr.restore(args.checkpoint, item=dummy)
    print(f"[sl] {args.model} loaded from {args.checkpoint}", flush=True)
    print(f"[sl] return bucket values: [{model.return_buckets_values[0]:.4f},"
          f" ..., {model.return_buckets_values[-1]:.4f}]", flush=True)

    rows = []
    for path in args.eval.split(","):
        rows += json.load(open(path.strip()))
    if args.max_rows:
        rows = rows[:args.max_rows]
    print(f"[sl] {len(rows)} MATE positions", flush=True)

    n_correct = 0
    n_total = 0
    n_ties = 0
    per_file = {}
    results = []
    t0 = time.time()
    for i, row in enumerate(rows):
        _src = (row.get("source") or "pool")
        per_file.setdefault(_src, [0, 0])
        fen = row.get("fen") or row.get("position")
        truth = row.get("truth_label") or row.get("label")
        te = row.get("task_extra") or {}
        ca = row.get("candidate_a") or row.get("move_a") or te.get("candidate_a")
        cb = row.get("candidate_b") or row.get("move_b") or te.get("candidate_b")
        if not truth:
            truth = te.get("truth_label")
        if not fen or not ca or not cb:
            print(f"[sl] skip row {i}: missing fen/candidates", flush=True)
            continue

        # Official ActionValueEngine.analyse: sequence per legal action =
        # [77 FEN tokens, action index, dummy return bucket] -> context 79.
        board = chess.Board(fen)
        fens = [board.fen()] * 2
        actions = np.array(
            [sl_utils.MOVE_TO_ACTION[ca], sl_utils.MOVE_TO_ACTION[cb]],
            dtype=np.int32)
        dummy = np.zeros((2, 1), dtype=np.int32)
        tok = np.stack([sl_tokenizer.tokenize(f) for f in fens]).astype(np.int32)
        sequences = np.concatenate([tok, actions[:, None], dummy], axis=1)

        log_probs = model.predictor.predict(
            params=model.params, targets=sequences, rng=None)
        bucket_probs = np.exp(np.asarray(log_probs[:, -1]))  # [2, 128]
        win_probs = bucket_probs @ model.return_buckets_values  # [2]

        wa, wb = float(win_probs[0]), float(win_probs[1])
        pred = "A" if wa > wb else "B"
        if wa == wb:
            n_ties += 1
        n_total += 1
        per_file[_src][1] += 1
        if pred == truth:
            n_correct += 1
            per_file[_src][0] += 1
        results.append({
            "fen": fen, "truth": truth, "pred": pred,
            "win_a": round(wa, 6), "win_b": round(wb, 6), "set": _src,
        })
        if (i + 1) % 200 == 0:
            print(f"[sl] {i+1}/{len(rows)} "
                  f"({(time.time()-t0)/(i+1):.2f}s/pos)", flush=True)

    acc = n_correct / n_total if n_total else 0.0
    print(f"[sl] MATE 2-choice accuracy: {n_correct}/{n_total} = "
          f"{acc*100:.1f}%  (gemma base = 58.1%; 9M official = 98.2%)",
          flush=True)
    for src_name, (c, t) in sorted(per_file.items()):
        if t:
            print(f"[sl]   {src_name}: {c}/{t} = {c/t*100:.1f}%", flush=True)
    if n_ties:
        print(f"[sl]   ties (wa==wb): {n_ties}", flush=True)
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "w") as f:
            json.dump({
                "model": args.model,
                "checkpoint": args.checkpoint,
                "sets": {k: {"correct": v[0], "total": v[1]}
                         for k, v in sorted(per_file.items())},
                "total": n_total, "correct": n_correct,
                "accuracy": round(acc, 6), "ties": n_ties,
                "rows": results,
            }, f, indent=1)
        print(f"[sl] saved -> {args.save}", flush=True)


if __name__ == "__main__":
    main()
