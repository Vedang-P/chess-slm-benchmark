"""Train the 5M student via distillation from the 270M teacher.

Losses (all switchable):
  L_KL  : KL(student_bucket_dist || teacher_bucket_dist)            [soft targets]
  L_CE  : cross-entropy vs HL-Gauss(Stockfish winprob) labels        [hard targets,
          mirroring the 9M's original objective]
  L_rank: hinge margin on Q(student,a_better) > Q(student,a_worse)   [DeepChess-style
          pairwise ranking, pairs built from multi-move FENs]

Total = L_KL + w_ce*L_CE + w_rank*L_rank.

Adam (hand-rolled, bias-corrected), warmup + cosine LR, EMA(0.999) for eval.
Checkpoints: npz (flattened fp32 params) + pickled treedef + config json, saved as
<outdir>/step-<n>.npz (+ .treedef.pkl + .config.json).

Usage (WSL GPU venv):
    python3 scripts/train_student.py --npz C:/tmp/searchless_chess/data/test/train_set.npz \
        --teacher C:/tmp/sl_data/teacher_logp.npy \
        --outdir C:/Users/vedang/Desktop/Research/chess-slm-benchamrking/results/student
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--teacher", required=True, help="teacher log-probs (.npy [N,128])")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dim", type=int, default=224)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--w-ce", type=float, default=0.5)
    ap.add_argument("--w-rank", type=float, default=0.5)
    ap.add_argument("--rank-margin", type=float, default=0.05)
    ap.add_argument("--rank-min-delta", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--max-records", type=int, default=0,
                    help="truncate dataset (for smoke tests)")
    ap.add_argument("--winprob-source", default="npz",
                    choices=["npz", "teacher"],
                    help="'teacher' = Q(teacher) derived from t_logp "
                         "(for sets without SF winprob labels)")
    ap.add_argument("--init", default="",
                    help="base path of a previous checkpoint to continue from "
                         "(e.g. results/student/step-15000)")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import haiku as hk  # noqa: F401  (imported by transformer module graph)

    sys.path.insert(0, str(Path(args.npz).parents[3]))  # package root
    from searchless_chess.src import tokenizer, transformer, utils

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    d = np.load(args.npz)
    tokens = d["tokens"]
    actions = d["actions"]
    winprob = d["winprob"]
    if args.max_records:
        tokens = tokens[: args.max_records]
        actions = actions[: args.max_records]
        winprob = winprob[: args.max_records]
    n = len(tokens)
    tdata = np.load(args.teacher)
    t_logp = tdata["teacher_logp"] if isinstance(tdata, np.lib.npyio.NpzFile) else tdata
    t_logp = np.asarray(t_logp, dtype=np.float32)
    assert t_logp.shape[0] == n, (t_logp.shape, n)
    if args.winprob_source == "teacher":
        from searchless_chess.src import utils as _u
        z = np.asarray(_u.get_uniform_buckets_edges_values(128)[1], dtype=np.float32)
        p = np.exp(t_logp)
        p /= p.sum(axis=-1, keepdims=True)
        winprob = p @ z
        print("[train] winprob derived from teacher Q", flush=True)
    print(f"[train] N={n} teacher={t_logp.shape}", flush=True)

    # ---- rank pairs: group rows by FEN (72-byte token prefix is unique) ----
    prefix = tokens[:, :72].view(np.uint64)
    _, fen_id = np.unique(prefix, axis=0, return_inverse=True)
    order = np.argsort(fen_id, kind="stable")
    pairs = []
    s = 0
    while s < n:
        e = s + 1
        while e < n and fen_id[order[e]] == fen_id[order[s]]:
            e += 1
        run = order[s:e]
        if len(run) >= 2:
            wp_run = winprob[run]
            for i in range(len(run)):
                for j in range(i + 1, len(run)):
                    if wp_run[i] - wp_run[j] >= args.rank_min_delta:
                        pairs.append((run[i], run[j]))
                    elif wp_run[j] - wp_run[i] >= args.rank_min_delta:
                        pairs.append((run[j], run[i]))
        s = e
    pairs = np.asarray(pairs, dtype=np.int32)
    print(f"[train] rank pairs: {len(pairs)}", flush=True)

    # ---- model ----
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
    params = predictor.initial_params(
        rng=jax.random.PRNGKey(args.seed),
        targets=np.ones((1, 1), dtype=np.uint32))
    n_params = sum(int(p.size) for p in jax.tree_util.tree_leaves(params))
    assert 4.5e6 <= n_params <= 5.9e6, f"expected ~5M, got {n_params}"
    print(f"[train] student params: {n_params:,}", flush=True)
    if args.init:
        base = Path(args.init)
        with open(f"{base}.treedef.pkl", "rb") as f:
            treedef = pickle.load(f)
        data = np.load(f"{base}.npz")
        leaves = [data[k] for k in
                  sorted(data.files, key=lambda x: int(x.split("_")[1]))]
        init_leaves = [np.asarray(l, dtype=np.float32) for l in leaves]
        params = jax.tree_util.tree_unflatten(treedef, init_leaves)
        print(f"[train] continued from {args.init}", flush=True)

    bucket_vals = np.asarray(
        utils.get_uniform_buckets_edges_values(128)[1], dtype=np.float32)
    bucket_vals = jnp.asarray(bucket_vals)

    def hl_gauss_batch(winprob_b: jnp.ndarray) -> jnp.ndarray:
        """HL-Gauss label smoothing, sigma=0.75 bins (paper Section 2.2)."""
        k = 128
        idx = jnp.clip(jnp.round(winprob_b * k).astype(jnp.int32), 0, k - 1)
        bins = jnp.arange(k, dtype=jnp.float32)
        q = jnp.exp(-(bins[None, :] - idx[:, None]) ** 2 / (2 * 0.75**2))
        return q / q.sum(axis=-1, keepdims=True)

    def make_seq(tok, act):
        return jnp.concatenate(
            [tok, act[:, None],
             jnp.zeros((tok.shape[0], 1), dtype=jnp.int32)], axis=1)

    @jax.jit
    def train_step(params, m, v, t, b_tok, b_act, b_hl, b_tp,
                   r1_idx, r2_idx, r1_tok, r2_tok, lr):
        def loss_fn(p):
            lp = predictor.predict(params=p, targets=make_seq(b_tok, b_act),
                                   rng=None)[:, -1]
            lp = jax.nn.log_softmax(lp, axis=-1)
            tp = jax.nn.softmax(b_tp, axis=-1)
            l_kl = jnp.mean(jnp.sum(tp * (jnp.log(tp + 1e-9) - lp), axis=-1))
            l_ce = -jnp.mean(jnp.sum(b_hl * lp, axis=-1))
            l_rank = jnp.asarray(0.0, dtype=jnp.float32)
            if args.w_rank > 0:
                def q_of(seq):
                    lp2 = predictor.predict(params=p, targets=seq, rng=None)[:, -1]
                    return jnp.sum(jax.nn.softmax(lp2, axis=-1) * bucket_vals, axis=-1)
                q1 = q_of(make_seq(r1_tok, r1_idx))
                q2 = q_of(make_seq(r2_tok, r2_idx))
                l_rank = jnp.mean(
                    jnp.maximum(0.0, args.rank_margin - (q1 - q2)))
            return l_kl + args.w_ce * l_ce + args.w_rank * l_rank, (l_kl, l_ce, l_rank)

        (loss, _aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
        m = jax.tree_util.tree_map(lambda a, b: 0.9 * a + 0.1 * b, m, grads)
        v = jax.tree_util.tree_map(lambda a, b: 0.999 * a + 0.001 * b * b, v, grads)
        m_hat = jax.tree_util.tree_map(lambda a: a / (1 - 0.9 ** (t + 1)), m)
        v_hat = jax.tree_util.tree_map(lambda a: a / (1 - 0.999 ** (t + 1)), v)
        new_params = jax.tree_util.tree_map(
            lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + 1e-8),
            params, m_hat, v_hat)
        return new_params, m, v, loss

    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    ema = jax.tree_util.tree_map(jnp.array, params)

    rng_state = np.random.default_rng(args.seed)
    t0 = time.time()
    for step in range(args.steps):
        idx = rng_state.integers(0, n, size=args.batch)
        b_tok = jnp.asarray(tokens[idx], dtype=jnp.int32)
        b_act = jnp.asarray(actions[idx], dtype=jnp.int32)
        b_hl = hl_gauss_batch(jnp.asarray(winprob[idx], dtype=jnp.float32))
        b_tp = jnp.asarray(t_logp[idx])
        n_pair = args.batch if args.w_rank > 0 and len(pairs) else 0
        if n_pair:
            pi = rng_state.integers(0, len(pairs), size=n_pair)
            r1, r2 = pairs[pi, 0], pairs[pi, 1]
            r1_tok = jnp.asarray(tokens[r1], dtype=jnp.int32)
            r1_idx = jnp.asarray(actions[r1], dtype=jnp.int32)
            r2_tok = jnp.asarray(tokens[r2], dtype=jnp.int32)
            r2_idx = jnp.asarray(actions[r2], dtype=jnp.int32)
        else:
            r1_tok = b_tok[:0]
            r1_idx = b_act[:0]
            r2_tok = b_tok[:0]
            r2_idx = b_act[:0]
        lr = args.lr * min(1.0, (step + 1) / args.warmup)
        if step > args.warmup:
            frac = (step - args.warmup) / max(1, args.steps - args.warmup)
            lr = args.lr * 0.5 * (1 + np.cos(np.pi * frac))
        params, m, v, loss = train_step(
            params, m, v, step, b_tok, b_act, b_hl, b_tp,
            r1_idx, r2_idx, r1_tok, r2_tok, jnp.asarray(lr, dtype=jnp.float32))
        ema = jax.tree_util.tree_map(
            lambda a, b: 0.999 * a + 0.001 * b, ema, params)
        if (step + 1) % 100 == 0:
            print(f"[train] step {step+1}/{args.steps} loss={float(loss):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if (step + 1) % args.ckpt_every == 0 or step + 1 == args.steps:
            save_ckpt(outdir, ema, args, step + 1, float(loss))
    print(f"[train] done in {time.time()-t0:.0f}s", flush=True)


def save_ckpt(outdir, ema, args, step, loss):
    import jax
    leaves, treedef = jax.tree_util.tree_flatten(ema)
    np.savez(outdir / f"step-{step}.npz", *[np.asarray(l) for l in leaves])
    with open(outdir / f"step-{step}.treedef.pkl", "wb") as f:
        pickle.dump(treedef, f)
    with open(outdir / f"step-{step}.config.json", "w") as f:
        json.dump({"dim": args.dim, "layers": args.layers, "heads": args.heads,
                   "step": step, "loss": loss}, f)
    print(f"[train] saved ckpt step-{step}", flush=True)


if __name__ == "__main__":
    main()
