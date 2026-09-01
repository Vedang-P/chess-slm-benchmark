"""Train the 5M student via distillation from the 270M teacher.

Losses (all switchable):
  L_KL  : KL(student_bucket_dist || teacher_bucket_dist)            [soft targets]
  L_CE  : cross-entropy vs HL-Gauss(Stockfish winprob) labels        [hard targets,
          mirroring the 9M's original objective]
  L_rank: hinge margin on Q(student,a_better) > Q(student,a_worse)   [DeepChess-style
          pairwise ranking, pairs built from multi-move FENs]

Total = L_KL + w_ce*L_CE + w_rank*L_rank.

The official Searchless Chess predictor already returns log-softmax values.
Keep that representation throughout the loss; applying log_softmax a second
time changes the distillation target and invalidates the controlled baseline.

Adam (hand-rolled, bias-corrected), warmup + cosine LR, EMA(0.999) for eval.
Checkpoints: <outdir>/checkpoint-<n>/ contains EMA weights plus full optimizer,
EMA, RNG, and config state; the directory is uploadable and resumable from HF.

Usage (WSL GPU venv):
    python3 scripts/train_student.py --npz C:/tmp/searchless_chess/data/test/train_set.npz \
        --teacher C:/tmp/sl_data/teacher_logp.npy \
        --outdir C:/Users/vedang/Desktop/Research/chess-slm-benchamrking/results/student
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.kaggle_checkpoint import (  # noqa: E402
    UploadTimer, api as make_hf_api, download_latest, upload_checkpoint,
    write_status,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dim", type=int, default=224)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--w-kl", type=float, default=1.0)
    ap.add_argument("--w-ce", type=float, default=0.5)
    ap.add_argument("--w-rank", type=float, default=0.5)
    ap.add_argument("--rank-margin", type=float, default=0.05)
    ap.add_argument("--rank-min-delta", type=float, default=0.05)
    ap.add_argument("--rank-subsample", type=int, default=0,
                    help="build rank pairs from this many random rows (bounded "
                         "table on full-scale datasets; 0 = all rows)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--npz", required=False)
    ap.add_argument("--teacher", required=False)
    ap.add_argument("--hf-shards", default="",
                    help="HF prefix for 8 shards, e.g. chessbench-full-build")
    ap.add_argument("--hf-repo", default="vedangfake/chess-slm-benchmark")
    ap.add_argument("--sl-repo", default=os.environ.get("SL_REPO", "/kaggle/working/searchless_chess"))
    import jax
    import jax.numpy as jnp
    sl_repo = Path(args.sl_repo)
    sys.path.insert(0, str(sl_repo.parent))
    from searchless_chess.src import tokenizer, transformer, utils

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    hf_client = None
    hf_timer = UploadTimer(args.hf_upload_every)
    hf_resume_dir = None
    if args.resume_from_hf:
        hf_client = make_hf_api(ROOT)
        hf_resume_dir = download_latest(hf_client, args.hf_repo, args.hf_run,
                                         outdir / "hf-resume")

    # ---- data (mmap-backed so the full multi-GB dataset stays out of RAM) ----
    d = np.load(args.npz, mmap_mode="r")
    tokens = d["tokens"]
    actions = d["actions"]
    winprob = d["winprob"] if "winprob" in d else None
    if args.max_records and winprob is not None:
        winprob = winprob[: args.max_records]
    if args.max_records:
        tokens = tokens[: args.max_records]
        actions = actions[: args.max_records]
    n = len(tokens)
    tdata = np.load(args.teacher, mmap_mode="r")
    t_logp = tdata["teacher_logp"] if isinstance(tdata, np.lib.npyio.NpzFile) else tdata
    t_logp = t_logp[:n]
    # ---- data: either single file or HF shards ----
    if args.hf_shards:
        from huggingface_hub import hf_hub_download
        import tempfile
        try:
            hf_client_shard = make_hf_api(ROOT)
            hf_token_shard = hf_client_shard.token
        except Exception:
            hf_client_shard = None
            hf_token_shard = None
        tmp_shard_dir = Path(tempfile.mkdtemp(prefix="hf_shards_"))
        # Use first shard for smoke, real training will stream shards round-robin
        tag = "00000"
        for fname in ("train_set.npz", "teacher_logp.npy"):
            dest = tmp_shard_dir / f"shard-{tag}-{fname}"
            if not dest.exists():
                hf_hub_download(repo_id=args.hf_repo, repo_type="dataset", token=hf_token_shard,
                                filename=f"{args.hf_shards}/shard-{tag}/{fname}", local_dir=str(tmp_shard_dir))
                cached = tmp_shard_dir / args.hf_shards / f"shard-{tag}" / fname
                if cached.exists():
                    import shutil
                    cached.replace(dest)
                    shutil.rmtree(tmp_shard_dir / args.hf_shards, ignore_errors=True)
        d = np.load(str(tmp_shard_dir / f"shard-{tag}-train_set.npz"), mmap_mode="r")
        tokens = d["tokens"]
        actions = d["actions"]
        winprob = d["winprob"] if "winprob" in d else None
        tdata = np.load(str(tmp_shard_dir / f"shard-{tag}-teacher_logp.npy"), mmap_mode="r")
        t_logp = tdata["teacher_logp"] if isinstance(tdata, np.lib.npyio.NpzFile) else tdata
        print(f"[sharded] using shard {tag} {len(tokens)} rows", flush=True)
    else:
        assert args.npz and args.teacher, "--npz/--teacher or --hf-shards required"
        d = np.load(args.npz, mmap_mode="r")
        tokens = d["tokens"]
        actions = d["actions"]
        winprob = d["winprob"] if "winprob" in d else None
        tdata = np.load(args.teacher, mmap_mode="r")
        t_logp = tdata["teacher_logp"] if isinstance(tdata, np.lib.npyio.NpzFile) else tdata
        t_logp = t_logp[:len(tokens)]
    if args.max_records and winprob is not None:
        winprob = winprob[: args.max_records]
    if args.max_records:
        tokens = tokens[: args.max_records]
        actions = actions[: args.max_records]
        t_logp = t_logp[:len(tokens)]
    n = len(tokens)
    assert t_logp.shape[0] == n, (t_logp.shape, n)
    if args.w_rank > 0:
        if args.rank_subsample and args.rank_subsample < n:
            rng_pick = np.random.default_rng(args.seed)
            pick = np.sort(rng_pick.choice(n, size=args.rank_subsample, replace=False))
            pair_tokens, pair_actions, pair_winprob = tokens[pick], actions[pick], winprob[pick]
            print(f"[train] rank pairs from subsample of {args.rank_subsample} rows", flush=True)
        else:
            pair_tokens, pair_actions, pair_winprob = tokens, actions, winprob
        prefix = np.asarray(pair_tokens[:, :72]).view(np.uint64)
        _, fen_id = np.unique(prefix, axis=0, return_inverse=True)
        order = np.argsort(fen_id, kind="stable")
        pairs = []
        s = 0
        m = len(order)
        while s < m:
            e = s + 1
            while e < m and fen_id[order[e]] == fen_id[order[s]]:
                e += 1
            run = order[s:e]
            if len(run) >= 2:
                wp_run = pair_winprob[run]
                for i in range(len(run)):
                    for j in range(i + 1, len(run)):
                        if wp_run[i] - wp_run[j] >= args.rank_min_delta:
                            pairs.append((run[i], run[j]))
                        elif wp_run[j] - wp_run[i] >= args.rank_min_delta:
                            pairs.append((run[j], run[i]))
            s = e
        pairs = np.asarray(pairs, dtype=np.int32)
        print(f"[train] rank pairs: {len(pairs)}", flush=True)
    else:
        pairs = np.zeros((0, 2), dtype=np.int32)
        print("[train] rank pairs: disabled (w_rank=0)", flush=True)

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
    if not args.init:
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
            return args.w_kl * l_kl + args.w_ce * l_ce + args.w_rank * l_rank, (l_kl, l_ce, l_rank)

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
    start_step = 0
    if hf_resume_dir is not None and (hf_resume_dir / "state.npz").exists():
        params, m, v, ema, start_step, rng_state = load_state(
            hf_resume_dir, jax, jnp)
        print(f"[resume] restored full state at step {start_step}", flush=True)
    t0 = time.time()
    for step in range(start_step, args.steps):
        idx = rng_state.integers(0, n, size=args.batch)
        b_tok = jnp.asarray(np.asarray(tokens[idx], dtype=np.int32), dtype=jnp.int32)
        b_act = jnp.asarray(np.asarray(actions[idx], dtype=np.int32), dtype=jnp.int32)
        b_hl = hl_gauss_batch(jnp.asarray(np.asarray(winprob[idx], dtype=np.float32), dtype=jnp.float32))
        b_tp = jnp.asarray(np.asarray(t_logp[idx], dtype=np.float32), dtype=jnp.float32)
        n_pair = args.batch if args.w_rank > 0 and len(pairs) else 0
        if n_pair:
            pi = rng_state.integers(0, len(pairs), size=n_pair)
            r1, r2 = pairs[pi, 0], pairs[pi, 1]
            r1_tok = jnp.asarray(np.asarray(tokens[r1], dtype=np.int32), dtype=jnp.int32)
            r1_idx = jnp.asarray(np.asarray(actions[r1], dtype=np.int32), dtype=jnp.int32)
            r2_tok = jnp.asarray(np.asarray(tokens[r2], dtype=np.int32), dtype=jnp.int32)
            r2_idx = jnp.asarray(np.asarray(actions[r2], dtype=np.int32), dtype=jnp.int32)
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
            checkpoint = save_ckpt(
                outdir, params, m, v, ema, rng_state, args, step + 1,
                float(loss))
            if hf_client is None and os.environ.get("HF_WRITE_TOKEN"):
                hf_client = make_hf_api(ROOT)
            if hf_client is not None and (hf_timer.due() or step + 1 == args.steps):
                try:
                    upload_checkpoint(hf_client, args.hf_repo, outdir,
                                      args.hf_run, checkpoint.name)
                    hf_timer.mark()
                except Exception as exc:
                    print(f"[hf] upload failed; local checkpoint retained: {exc}",
                          flush=True)
    print(f"[train] done in {time.time()-t0:.0f}s", flush=True)


def save_ckpt(outdir, params, m, v, ema, rng_state, args, step, loss):
    import jax
    checkpoint = outdir / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    leaves, treedef = jax.tree_util.tree_flatten(ema)
    np.savez(checkpoint / f"step-{step}.npz", *[np.asarray(l) for l in leaves])
    with open(checkpoint / f"step-{step}.treedef.pkl", "wb") as f:
        pickle.dump(treedef, f)
    full_leaves, full_treedef = jax.tree_util.tree_flatten(
        {"params": params, "m": m, "v": v, "ema": ema})
    np.savez(checkpoint / "state.npz",
             **{f"arr_{i}": np.asarray(x) for i, x in enumerate(full_leaves)})
    with open(checkpoint / "state.treedef.pkl", "wb") as f:
        pickle.dump(full_treedef, f)
    with open(checkpoint / "rng.pkl", "wb") as f:
        pickle.dump(rng_state.bit_generator.state, f)
    with open(checkpoint / "config.json", "w") as f:
        json.dump({"dim": args.dim, "layers": args.layers, "heads": args.heads,
                   "step": step, "loss": loss, "seed": args.seed}, f)
    print(f"[train] saved complete checkpoint-{step}", flush=True)
    return checkpoint


def load_state(checkpoint, jax, jnp):
    with open(checkpoint / "state.treedef.pkl", "rb") as f:
        treedef = pickle.load(f)
    arrays = np.load(checkpoint / "state.npz")
    leaves = [jnp.asarray(arrays[k]) for k in
              sorted(arrays.files, key=lambda x: int(x.split("_")[1]))]
    state = jax.tree_util.tree_unflatten(treedef, leaves)
    with open(checkpoint / "config.json", encoding="utf-8") as f:
        step = int(json.load(f)["step"])
    with open(checkpoint / "rng.pkl", "rb") as f:
        rng_state = np.random.default_rng()
        rng_state.bit_generator.state = pickle.load(f)
    return state["params"], state["m"], state["v"], state["ema"], step, rng_state


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # A Kaggle kernel can disappear before local logs are copied out. Try
        # to leave a short, actionable failure marker in the same HF run.
        try:
            import traceback
            def _flag(name, default):
                flag = f"--{name}"
                return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default
            client = make_hf_api(ROOT)
            write_status(
                client, _flag("hf-repo", "vedangfake/chess-slm-benchmark"),
                _flag("hf-run", "student-5m"),
                type(exc).__name__ + ": " + str(exc) + "\n" +
                traceback.format_exc()[-4000:],
            )
        except Exception as status_exc:
            print(f"[hf] could not upload failure status: {status_exc}", flush=True)
        raise
