"""Generate 270M teacher labels for the student training set (GPU).

Reads train_set.npz (tokens/actions) and writes teacher bucket log-probs
(270M, official checkpoint) as fp16 [N, 128].

Usage (WSL GPU venv):
    python3 scripts/teacher_label.py --npz C:/tmp/searchless_chess/data/test/train_set.npz \
        --checkpoint C:/tmp/sl9m/270M/6400000/params --out C:/tmp/sl_data/teacher_logp.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--checkpoint", required=True,
                    help="270M orbax checkpoint dir (…/270M/6400000/params)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=128,
                    help="batch size (512+ OOMs on 6GB GPUs for the 270M)")
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--sl-repo", default="",
                    help="official searchless_chess repo root")
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--start", type=int, default=0,
                    help="first row (chunked runs survive WSL GPU wedges)")
    ap.add_argument("--end", type=int, default=0, help="last row (0 = all)")
    args = ap.parse_args()

    import os

    sl_repo = Path(args.sl_repo or os.environ.get("SL_REPO", "C:/tmp/searchless_chess"))
    sys.path.insert(0, str(sl_repo.parent))

    import numpy as np
    import jax
    import orbax.checkpoint as ocp

    from searchless_chess.src import tokenizer, transformer, utils

    d = np.load(args.npz)
    tokens = d["tokens"]
    actions = d["actions"]
    if args.max_records:
        tokens = tokens[: args.max_records]
        actions = actions[: args.max_records]
    if args.end:
        tokens = tokens[args.start : args.end]
        actions = actions[args.start : args.end]
    elif args.start:
        tokens = tokens[args.start :]
        actions = actions[args.start :]
    n = len(tokens)
    print(f"[teach] {n} triples (start={args.start})", flush=True)

    cfg = transformer.TransformerConfig(
        vocab_size=utils.NUM_ACTIONS,
        output_size=128,
        pos_encodings=transformer.PositionalEncodings.LEARNED,
        max_sequence_length=tokenizer.SEQUENCE_LENGTH + 2,
        num_heads=args.heads,
        num_layers=args.layers,
        embedding_dim=args.dim,
        apply_post_ln=True,
        apply_qk_layernorm=False,
        use_causal_mask=False,
    )
    predictor = transformer.build_transformer_predictor(cfg)
    dummy = predictor.initial_params(
        rng=jax.random.PRNGKey(1),
        targets=np.ones((1, 1), dtype=np.uint32))
    params = ocp.Checkpointer(ocp.StandardCheckpointHandler()).restore(
        args.checkpoint, item=dummy)
    if not args.no_fp16:
        import jax.numpy as jnp
        params = jax.tree_util.tree_map(lambda x: x.astype(jnp.float16), params)
    pred_jit = jax.jit(
        lambda p, seq: predictor.predict(params=p, targets=seq, rng=None))
    print(f"[teach] 270M loaded (fp16={not args.no_fp16})", flush=True)

    out = np.lib.format.open_memmap(
        str(Path(args.out)), mode="w+", dtype=np.float16, shape=(n, 128))
    dummy_bucket = np.zeros((args.batch, 1), dtype=np.int32)
    t0 = time.time()
    for s in range(0, n, args.batch):
        e = min(s + args.batch, n)
        seq = np.concatenate([
            tokens[s:e].astype(np.int32),
            actions[s:e, None].astype(np.int32),
            dummy_bucket[: e - s],
        ], axis=1)
        lp = pred_jit(params, seq)
        out[s:e] = np.asarray(lp[:, -1], dtype=np.float16)
        if (s // args.batch) % 25 == 0:
            print(f"[teach] {e}/{n} ({(time.time()-t0)/max(e,1)*e:.0f}s est)",
                  flush=True)
            out.flush()
    print(f"[teach] done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
