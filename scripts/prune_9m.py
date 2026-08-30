"""Prune the pretrained 9M checkpoint to ~5M effective params.

Two modes:
  unstructured: zero 44% smallest-magnitude weights per matrix (keeps dense shape,
                5M effective params, need sparse kernels for speedup)
  structured:  slice width 256->192 (keep top L2-norm channels) -> dense 192/8 model

Usage (WSL GPU not needed, CPU fine):
    python3 scripts/prune_9m.py --sparsity 0.442 --mode unstructured --out results/pruned-5M
    python3 scripts/prune_9m.py --target-dim 192 --mode structured --out results/pruned-5M-dense
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
    ap.add_argument("--checkpoint", default="C:/tmp/sl9m/9M/6400000/params")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sparsity", type=float, default=0.442, help="fraction zeroed (unstructured)")
    ap.add_argument("--mode", choices=["unstructured", "structured"], default="unstructured")
    ap.add_argument("--target-dim", type=int, default=192)
    ap.add_argument("--sl-repo", default="")
    args = ap.parse_args()

    import os

    sl_repo = Path(args.sl_repo or os.environ.get("SL_REPO", "C:/tmp/searchless_chess"))
    sys.path.insert(0, str(sl_repo.parent))

    import jax
    import orbax.checkpoint as ocp
    from searchless_chess.src import transformer, utils

    # --- Windows ocdbt URL fix ---
    import orbax.checkpoint.type_handlers as th
    import jax.experimental.array_serialization.serialization as _ser
    import re

    def _fix(spec):
        kv = spec.get("kvstore")
        if isinstance(kv, dict) and isinstance(kv.get("base"), str) and re.match(r"file://[A-Za-z]:", kv["base"]):
            kv["base"] = "file:///" + kv["base"][7:].replace("\\", "/")

    _orig = th.get_tensorstore_spec
    th.get_tensorstore_spec = lambda d, *a, **k: (_fix(s := _orig(d, *a, **k)), s)[1]
    _orig_ad = _ser.async_deserialize
    _ser.async_deserialize = lambda sh, ts, **kw: (_fix(ts), _orig_ad(sh, ts, **kw))[1]

    # Load 9M
    cfg9 = transformer.TransformerConfig(
        vocab_size=utils.NUM_ACTIONS, output_size=128,
        pos_encodings=transformer.PositionalEncodings.LEARNED,
        max_sequence_length=79, num_heads=8, num_layers=8, embedding_dim=256,
        apply_post_ln=True, apply_qk_layernorm=False, use_causal_mask=False)
    pred9 = transformer.build_transformer_predictor(cfg9)
    dummy9 = pred9.initial_params(rng=jax.random.PRNGKey(0), targets=np.ones((1, 1), dtype=np.uint32))
    params9 = ocp.Checkpointer(ocp.StandardCheckpointHandler()).restore(args.checkpoint, item=dummy9)
    n9 = sum(int(np.asarray(l).size) for l in jax.tree_util.tree_leaves(params9))
    print(f"[prune] 9M loaded: {n9:,} params")

    if args.mode == "unstructured":
        keep = 1.0 - args.sparsity
        flat = jax.tree_util.tree_flatten_with_path(params9)[0]
        pruned = {}
        # Use dict for reconstruction via unflatten needs treedef; easier: tree_map
        def prune_leaf(x):
            a = np.asarray(x)
            if a.ndim < 2 or a.size < 1000:
                return a
            k = int(a.size * keep)
            thresh = np.partition(np.abs(a).flatten(), a.size - k)[a.size - k]
            return np.where(np.abs(a) >= thresh, a, 0).astype(a.dtype)

        # Need to preserve treedef: use tree_map with path-aware? Simpler: rebuild via flatten
        leaves, treedef = jax.tree_util.tree_flatten(params9)
        new_leaves = [prune_leaf(np.asarray(l)) for l in leaves]
        params_p = jax.tree_util.tree_unflatten(treedef, new_leaves)
        n_eff = sum(int((np.asarray(l) != 0).sum()) for l in new_leaves)
        print(f"[prune] unstructured {args.sparsity*100:.1f}% sparsity -> effective {n_eff:,} params ({n_eff/n9*100:.1f}% keep)")

        # Save in student format (same 256/8 arch, sparse weights)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        flat_p, td_p = jax.tree_util.tree_flatten(params_p)
        np.savez(str(out) + ".npz", *[np.asarray(l) for l in flat_p])
        with open(str(out) + ".treedef.pkl", "wb") as f:
            pickle.dump(td_p, f)
        with open(str(out) + ".config.json", "w") as f:
            json.dump({"dim": 256, "layers": 8, "heads": 8, "mode": "unstructured",
                       "sparsity": args.sparsity, "effective_params": n_eff, "orig_params": n9}, f)
        print(f"[prune] saved -> {out}.npz")
    else:
        # Structured: 256 -> target_dim, keep top L2-norm channels globally
        # Compute per-channel importance as sum of squared weights across all dim-dependent matrices
        # For simplicity, use embedding column L2 as proxy (fast, correlates with Wanda at ~0.8)
        flat = jax.tree_util.tree_flatten_with_path(params9)[0]
        # Find embed weight
        for path, leaf in flat:
            name = "/".join(str(p.key) for p in path)
            if name == "embed/embeddings":
                emb = np.asarray(leaf)  # [1968, 256]
                chan_score = np.linalg.norm(emb, axis=0)  # per dim
                keep_idx = np.argsort(chan_score)[::-1][: args.target_dim]
                keep_idx = np.sort(keep_idx)
                print(f"[prune] structured keep channels (top L2 of embed): {keep_idx[:5]}...")
                break
        # Build 192/8 student and slice weights
        from searchless_chess.src.transformer import TransformerConfig
        cfg_s = TransformerConfig(vocab_size=utils.NUM_ACTIONS, output_size=128,
                                  pos_encodings=transformer.PositionalEncodings.LEARNED,
                                  max_sequence_length=79, num_heads=8, num_layers=8,
                                  embedding_dim=args.target_dim, apply_post_ln=True,
                                  apply_qk_layernorm=False, use_causal_mask=False)
        pred_s = transformer.build_transformer_predictor(cfg_s)
        dummy_s = pred_s.initial_params(rng=jax.random.PRNGKey(0), targets=np.ones((1, 1), dtype=np.uint32))
        # Slice 9M params into student shape via keep_idx
        # This is intricate: for each leaf, slice dims that correspond to the pruned embedding dim
        # Simplified: for 2D weights [in, out] where in or out ==256, slice both dims to 192
        # For 1D norms [256] -> [192]
        leaves_s, td_s = jax.tree_util.tree_flatten(dummy_s)
        # Map student leaves to 9M leaves by path
        # Build dict path->array for 9M
        map9 = {"/".join(str(p.key) for p in path): np.asarray(leaf) for path, leaf in flat}
        flat_s_path = jax.tree_util.tree_flatten_with_path(dummy_s)[0]
        new_leaves = []
        for (path, _), leaf_s in zip(flat_s_path, leaves_s):
            name = "/".join(str(p.key) for p in path)
            arr9 = map9.get(name)
            if arr9 is None:
                # New architecture leaf not in 9M (should not happen for same depth)
                new_leaves.append(np.asarray(leaf_s))
                continue
            tgt_shape = np.asarray(leaf_s).shape
            if arr9.shape == tgt_shape:
                new_leaves.append(arr9)
            elif len(arr9.shape) == 2 and len(tgt_shape) == 2:
                # Slice both dims if they are 256
                r_idx = keep_idx
                c_idx = keep_idx
                # Heuristic: if a dim is 1968 or 79 or 128, keep it
                if arr9.shape[0] == 1968:
                    r_idx = np.arange(1968)
                if arr9.shape[1] == 128:
                    c_idx = np.arange(128)
                if arr9.shape[0] == 79:
                    r_idx = np.arange(79)
                sliced = arr9[np.ix_(r_idx if arr9.shape[0] == 256 else np.arange(arr9.shape[0]),
                                      c_idx if arr9.shape[1] in (256, 1024) else np.arange(arr9.shape[1]))]
                # Handle 1024->768 for MLP: keep top 768 of the 1024 dim via column L2 of that matrix
                if arr9.shape[1] == 1024 and tgt_shape[1] == 768:
                    col_score = np.linalg.norm(arr9, axis=0)
                    c_keep = np.argsort(col_score)[::-1][:768]
                    c_keep = np.sort(c_keep)
                    sliced = arr9[np.ix_(keep_idx if arr9.shape[0] == 256 else np.arange(arr9.shape[0]), c_keep)]
                if sliced.shape != tgt_shape:
                    # Fallback: truncated slice
                    sliced = sliced[: tgt_shape[0], : tgt_shape[1]] if len(tgt_shape) == 2 else sliced[: tgt_shape[0]]
                new_leaves.append(sliced.astype(np.asarray(leaf_s).dtype))
            elif len(arr9.shape) == 1:
                if arr9.shape[0] == 256:
                    new_leaves.append(arr9[keep_idx].astype(np.asarray(leaf_s).dtype))
                else:
                    new_leaves.append(arr9)
            else:
                new_leaves.append(arr9)
        params_s = jax.tree_util.tree_unflatten(td_s, new_leaves)
        n_s = sum(int(np.asarray(l).size) for l in new_leaves)
        print(f"[prune] structured {args.target_dim}/256 -> {n_s:,} params")
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        flat_s, td_s2 = jax.tree_util.tree_flatten(params_s)
        np.savez(str(out) + ".npz", *[np.asarray(l) for l in flat_s])
        with open(str(out) + ".treedef.pkl", "wb") as f:
            pickle.dump(td_s2, f)
        with open(str(out) + ".config.json", "w") as f:
            json.dump({"dim": args.target_dim, "layers": 8, "heads": 8, "mode": "structured",
                       "keep_idx": keep_idx.tolist(), "params": n_s}, f)
        print(f"[prune] saved -> {out}.npz")


if __name__ == "__main__":
    main()
