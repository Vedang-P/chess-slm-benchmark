"""Conservative CC-GAVN continuation trainer (cont1).

Starts from the verified 3.9M-parameter CC-GAVN checkpoint (76.20% puzzles) and
keeps the architecture. The recipe fixes what broke the previous continuation:

* low constant learning rate with warmup (the old restart sat at 26% of the
  original peak) and an EMA of the weights, which is what gets evaluated;
* every batch mixes the original teacher-distilled broad corpus (replay) with
  whole tactical roots, so tactical specialisation cannot erase general play;
* a listwise loss over each tactical root's legal moves, aimed at the argmax
  that the puzzle metric actually scores;
* held-out validation puzzles scored during training with the exact harness
  rules, plus a regression guard that halts the run instead of burning quota.

Everything a batch contains is a pure function of (seed, step), so a session
that ends at ~11.5 h resumes on the next Kaggle account exactly where it left
off from the Hugging Face checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.cont_data import (TacticalPack, broad_block_tag, draw_rows,  # noqa: E402
                               extra_exclusion_hashes, read_split_csv, step_rng)
from scripts.cont_eval import (BatchedScorer, evaluate_mate, evaluate_puzzles,  # noqa: E402
                               lichess_puzzles, mate_rows, official_puzzles, summarize)
from scripts.data_hygiene import (eligible_mask, load_exclusions,  # noqa: E402
                                  set_learning_rate)
from scripts.kaggle_checkpoint import (UploadTimer, api as make_hf_api,  # noqa: E402
                                       download_latest, upload_checkpoint, write_status)
from scripts.shard_data import ShardManager  # noqa: E402
from scripts.train_ccgavn import (CCGAVN, candidate_relation_types,  # noqa: E402
                                  horizontal_action_map, reflect_horizontal)
from scripts.train_gavn import action_tables  # noqa: E402

RECIPE_FIELDS = ("dim", "layers", "heads", "broad_rows", "tactical_groups", "lr", "warmup",
                 "ema_decay", "weight_decay", "w_dist", "w_ce", "w_ce_tac", "lambda_list",
                 "tau", "reflect_prob", "seed", "block_steps", "decay_start", "decay_steps",
                 "rl_weight", "rl_beta", "init_weights", "ply_groups")
BASE_RUN, BASE_STEP = "ccgavn-5m-v2-functional-seed0", 3_896_658


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--hf-repo", default="vedangfake/chess-slm-benchmark")
    p.add_argument("--hf-run", required=True, help="output prefix for this run")
    p.add_argument("--init-hf-run", default=BASE_RUN)
    p.add_argument("--init-checkpoint", default=f"checkpoint-{BASE_STEP}")
    p.add_argument("--resume-from-hf", action="store_true")
    p.add_argument("--outdir", required=True)
    p.add_argument("--sl-repo", default=os.environ.get("SL_REPO", "/kaggle/working/searchless_chess"))
    p.add_argument("--searchless-commit", default="")
    p.add_argument("--cont-dir", default="", help="local dir holding cont-v1/ (skip HF download)")
    # model
    p.add_argument("--dim", type=int, default=208)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--model-version", default="v2")
    # data
    p.add_argument("--broad-prefix", default="chessbench-full-build")
    p.add_argument("--broad-tags-file", default=str(ROOT / "configs/ccgavn-2b-shard-tags.json"))
    p.add_argument("--broad-rows", type=int, default=1434)
    p.add_argument("--tactical-groups", type=int, default=20)
    p.add_argument("--ply-groups", type=int, default=0,
                   help="extra tactical groups per step from the later-ply pack "
                        "(cont-v2/ply-pack); 0 disables it and leaves every batch unchanged")
    p.add_argument("--ply-dir", default="", help="local dir holding cont-v2/ (skip HF download)")
    p.add_argument("--block-steps", type=int, default=6000)
    p.add_argument("--reflect-prob", type=float, default=0.5)
    p.add_argument("--exclusion-puzzles", required=True, help="official puzzles.csv")
    p.add_argument("--mate-dir", default=str(ROOT / "data/positions"))
    # optimisation
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--decay-start", type=int, default=-1, help="step to begin linear decay to 0")
    p.add_argument("--decay-steps", type=int, default=0)
    p.add_argument("--ema-decay", type=float, default=0.9998)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--w-dist", type=float, default=1.0)
    p.add_argument("--w-ce", type=float, default=0.25)
    p.add_argument("--w-ce-tac", type=float, default=1.0)
    p.add_argument("--lambda-list", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=0.05)
    # stage 2: exact-expectation policy optimisation against a frozen reference
    p.add_argument("--rl-weight", type=float, default=0.0, help="0 disables the RL term")
    p.add_argument("--rl-beta", type=float, default=0.05, help="KL(pi || pi_ref) coefficient")
    p.add_argument("--init-weights", choices=["model", "ema"], default="model",
                   help="which weights of the init checkpoint to start from (and to freeze as "
                        "the RL reference)")
    p.add_argument("--seed", type=int, default=0)
    # cadence / control
    p.add_argument("--max-steps", type=int, default=0, help="stage steps; 0 = until session end")
    p.add_argument("--val-every", type=int, default=5000)
    p.add_argument("--ckpt-every", type=int, default=10000)
    p.add_argument("--hf-upload-every", type=float, default=1200)
    p.add_argument("--milestone-every", type=int, default=100000)
    p.add_argument("--session-seconds", type=float, default=0)
    p.add_argument("--kernel-started-epoch", type=float, default=0)
    p.add_argument("--guard-after", type=int, default=20000)
    p.add_argument("--guard-drop", type=float, default=3.0)
    p.add_argument("--dev-batch", type=int, default=8192)
    p.add_argument("--allow-recipe-change", action="store_true")
    p.add_argument("--no-official-eval", action="store_true")
    p.add_argument("--val-detail", action="store_true",
                   help="also log validation failures per ply and save per-puzzle results")
    return p.parse_args(argv)


# ------------------------------------------------------------------ math helpers
def lr_at(step: int, args) -> float:
    lr = args.lr * min(1.0, (step + 1) / max(1, args.warmup))
    if args.decay_start >= 0 and step >= args.decay_start:
        frac = min(1.0, (step - args.decay_start) / max(1, args.decay_steps))
        lr = args.lr * (1.0 - frac)
    return lr


def hl_gauss(torch, wp, bins: int = 128, sigma: float = 0.75):
    bucket = torch.clamp(torch.ceil(wp * bins).long() - 1, 0, bins - 1)
    idx = torch.arange(bins, device=wp.device, dtype=torch.float32)[None]
    hard = torch.exp(-(idx - bucket[:, None]) ** 2 / (2 * sigma ** 2))
    return hard / hard.sum(-1, keepdim=True)


def listwise_loss(torch, q, wp, gid, lens, tau: float):
    """Cross-entropy of softmax(q/tau) against softmax(wp/tau) within each root.

    Returns (loss, kl, hit, regret): `hit` counts roots whose argmax move is
    (tied for) the best Stockfish value, `regret` is the mean win-prob lost.
    """
    groups, longest = int(lens.numel()), int(lens.max())
    starts = torch.cumsum(lens, 0) - lens
    pos = torch.arange(q.numel(), device=q.device) - starts[gid]
    pad = -1e4
    qq = torch.full((groups, longest), pad, device=q.device)
    tt = torch.full((groups, longest), pad, device=q.device)
    qq[gid, pos] = q / tau
    tt[gid, pos] = wp / tau
    logp = torch.log_softmax(qq, dim=1)
    target = torch.softmax(tt, dim=1)
    ce = -(target * logp).sum(1)
    entropy = -(target * torch.log(target.clamp_min(1e-12))).sum(1)
    pick = qq.argmax(1, keepdim=True)
    got = tt.gather(1, pick).squeeze(1)
    best = tt.max(1).values
    hit = (got >= best - 1e-6 / tau).float().mean()
    regret = ((best - got) * tau).mean()
    return ce.mean(), (ce - entropy).mean(), hit.detach(), regret.detach()


def exact_policy_loss(torch, q, wp, gid, lens, tau: float, q_ref=None, beta: float = 0.0):
    """Negative expected return of the softmax policy over each root's legal moves.

    Every legal move has a Stockfish value, so J = sum_a pi(a) * R(a) is computed
    exactly (no sampled groups, no advantage noise). `beta` penalises KL(pi || pi_ref)
    against a frozen reference so the policy cannot drift into degenerate play.
    Returns (loss, J, kl): the means over roots.
    """
    q, wp = q.float(), wp.float()
    q_ref = None if q_ref is None else q_ref.float()
    groups, longest = int(lens.numel()), int(lens.max())
    starts = torch.cumsum(lens, 0) - lens
    pos = torch.arange(q.numel(), device=q.device) - starts[gid]
    pad = -1e4
    logits = torch.full((groups, longest), pad, device=q.device)
    reward = torch.zeros((groups, longest), device=q.device)
    logits[gid, pos] = q / tau
    reward[gid, pos] = wp
    logp = torch.log_softmax(logits, dim=1)
    pi = logp.exp()
    value = (pi * reward).sum(1)
    kl = torch.zeros_like(value)
    if q_ref is not None and beta:
        ref = torch.full((groups, longest), pad, device=q.device)
        ref[gid, pos] = q_ref / tau
        kl = (pi * (logp - torch.log_softmax(ref, dim=1))).sum(1)
    return -(value - beta * kl).mean(), value.mean().detach(), kl.mean().detach()


def expected_value(torch, logits, bucket_values):
    return torch.softmax(logits.float(), -1) @ bucket_values


def batch_loss(torch, logits, batch, args, bucket_values):
    """Mean per-row loss of a mixed batch (broad rows first, then tactical rows).

    Broad rows: teacher-distribution CE (+ weighted Stockfish HL-Gauss CE).
    Tactical rows: Stockfish HL-Gauss CE, plus the listwise term per root.
    Returns (loss, diagnostics, number_of_rows).
    """
    device = logits.device
    nb = batch["nb"]
    logits = logits.float()
    logp = torch.log_softmax(logits, -1)
    teacher_probs = torch.softmax(torch.as_tensor(batch["bteach"], device=device), -1)
    dist = -(teacher_probs * logp[:nb]).sum(-1)
    ce_b = -(hl_gauss(torch, torch.as_tensor(batch["bw"], device=device)) * logp[:nb]).sum(-1)
    parts = [args.w_dist * dist + args.w_ce * ce_b]
    zero = torch.zeros((), device=device)
    diag = {"dist": dist.mean().detach(), "ce_b": ce_b.mean().detach(), "ce_t": zero,
            "list": zero, "hit": zero, "regret": zero}
    n_tac, list_loss = 0, zero
    if "tt" in batch:
        tw = torch.as_tensor(batch["tw"], device=device)
        ce_t = -(hl_gauss(torch, tw) * logp[nb:]).sum(-1)
        parts.append(args.w_ce_tac * ce_t)
        n_tac = len(ce_t)
        gid = torch.as_tensor(batch["gid"], device=device)
        lens = torch.as_tensor(batch["lens"], device=device)
        q = expected_value(torch, logits[nb:], bucket_values)
        if args.lambda_list:
            list_loss, _, hit, regret = listwise_loss(torch, q, tw, gid, lens, args.tau)
        else:  # monitoring only: the pointwise arm still reports ranking quality
            with torch.no_grad():
                list_loss, _, hit, regret = listwise_loss(torch, q.detach(), tw, gid, lens, args.tau)
        diag.update(ce_t=ce_t.mean().detach(), list=list_loss.detach(), hit=hit, regret=regret)
        if getattr(args, "rl_weight", 0) and "q_ref" in batch:
            rl, value, kl = exact_policy_loss(
                torch, q, tw, gid, lens, args.tau,
                torch.as_tensor(batch["q_ref"], device=device), args.rl_beta)
            diag.update(rl_value=value, rl_kl=kl)
            rl_term = args.rl_weight * (n_tac / (nb + n_tac)) * rl
    total = nb + n_tac
    loss = torch.cat(parts).sum() / total
    if args.lambda_list and n_tac:
        loss = loss + args.lambda_list * (n_tac / total) * list_loss
    if n_tac and "rl_value" in diag:
        loss = loss + rl_term
    return loss, diag, total


# ------------------------------------------------------------------ main
def main(argv=None):
    import torch

    args = parse_args(argv)
    started = time.time()
    deadline = ((args.kernel_started_epoch or started) + args.session_seconds
                if args.session_seconds else None)
    source_commit = "unknown"
    try:
        import subprocess
        source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                                stderr=subprocess.DEVNULL).strip()
    except Exception:
        pass

    src, dst, promo, utils = action_tables(Path(args.sl_repo))
    action_map = horizontal_action_map(utils)
    bucket_np = np.asarray(utils.get_uniform_buckets_edges_values(128)[1], dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bucket_values = torch.as_tensor(bucket_np, device=device)
    torch.manual_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    hf = make_hf_api(ROOT)
    token = hf.token

    # ---- data artifacts
    cont_root = Path(args.cont_dir) if args.cont_dir else outdir / "data"
    if not (cont_root / "cont-v1" / "manifest.json").exists():
        from huggingface_hub import snapshot_download
        snapshot_download(args.hf_repo, repo_type="dataset", token=token,
                          local_dir=str(cont_root), allow_patterns=["cont-v1/*", "cont-v1/pack/*"])
    cont = cont_root / "cont-v1"
    manifest = json.loads((cont / "manifest.json").read_text(encoding="utf-8"))
    val_rows = read_split_csv(cont / "val.csv")
    val_puzzles = lichess_puzzles(val_rows)
    tags_bytes = Path(args.broad_tags_file).read_bytes()
    tags_payload = json.loads(tags_bytes)
    tags = [str(t) for t in (tags_payload["tags"] if isinstance(tags_payload, dict) else tags_payload)]
    tags_digest = hashlib.sha256(tags_bytes).hexdigest()

    official_hashes, exclusion_digest = load_exclusions(Path(args.mate_dir),
                                                        Path(args.exclusion_puzzles))
    split_hashes = extra_exclusion_hashes(val_rows + read_split_csv(cont / "test.csv"))
    excluded = np.unique(np.concatenate([official_hashes, split_hashes]))

    pack = None
    if args.tactical_groups:
        pack = TacticalPack.load([cont / part["file"] for part in manifest["pack_parts"]])
        print(f"[data] tactical roots train={len(pack.train_roots):,} dev={len(pack.dev_roots):,} "
              f"rows={len(pack.actions):,}", flush=True)
    ply_pack, ply_manifest = None, None
    if args.ply_groups:
        from huggingface_hub import snapshot_download
        from scripts.ply_data import PACK_PREFIX, PlyPack
        ply_root = Path(args.ply_dir) if args.ply_dir else outdir / "data"
        if not (ply_root / PACK_PREFIX / "manifest.json").exists():
            snapshot_download(args.hf_repo, repo_type="dataset", token=token,
                              local_dir=str(ply_root), allow_patterns=[f"{PACK_PREFIX}/*"])
        ply_manifest = json.loads((ply_root / PACK_PREFIX / "manifest.json").read_text(encoding="utf-8"))
        ply_pack = PlyPack.load([ply_root / PACK_PREFIX / part["file"]
                                 for part in ply_manifest["pack_parts"]])
        print(f"[data] ply roots train={len(ply_pack.train_roots):,} dev={len(ply_pack.dev_roots):,} "
              f"rows={len(ply_pack.actions):,}", flush=True)
    dev = np.load(cont / "broad-dev.npz")
    dev_tokens = torch.as_tensor(np.asarray(dev["tokens"], dtype=np.int64), device=device)
    dev_actions = torch.as_tensor(np.asarray(dev["actions"], dtype=np.int64), device=device)
    dev_teacher = torch.as_tensor(np.asarray(dev["teacher"], dtype=np.float32), device=device)
    dev_wp = torch.as_tensor(np.asarray(dev["winprob"], dtype=np.float32), device=device)
    dev_pick = np.random.default_rng(args.seed + 9187).choice(
        len(dev_tokens), size=min(args.dev_batch, len(dev_tokens)), replace=False)
    dev_pick = torch.as_tensor(dev_pick, device=device)
    tactical_dev_roots = None
    if pack is not None and len(pack.dev_roots):
        tactical_dev_roots = pack.dev_roots[
            np.random.default_rng(args.seed + 5).choice(
                len(pack.dev_roots), size=min(2000, len(pack.dev_roots)), replace=False)]

    ply_dev_roots = None
    if ply_pack is not None and len(ply_pack.dev_roots):
        ply_dev_roots = ply_pack.dev_roots[
            np.random.default_rng(args.seed + 6).choice(
                len(ply_pack.dev_roots), size=min(2000, len(ply_pack.dev_roots)), replace=False)]

    shard_rows = {}
    for path in (ROOT / "configs/chessbench_rows.json", ROOT / "kernels/build-2b/shard_rows.json"):
        if path.exists():
            shard_rows.update({str(k): int(v) for k, v in json.loads(path.read_text()).items()})
    manager = ShardManager(args.hf_repo, args.broad_prefix,
                           Path(os.environ.get("SHARD_CACHE", "/kaggle/tmp/shards")), token=token,
                           expect_tags=len(tags), tags=tags,
                           row_counts={t: shard_rows[t] for t in tags} if all(t in shard_rows for t in tags) else None)
    manager.count_rows()

    # ---- model, EMA, optimiser
    raw_model = CCGAVN(torch, args.dim, args.layers, args.heads, src, dst, promo,
                       candidate_relation_types(), model_version=args.model_version).to(device)
    params = sum(p.numel() for p in raw_model.parameters())
    if params > 5_000_000:
        raise ValueError(f"model has {params:,} parameters; exceeds the strict 5M budget")
    model = raw_model
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(raw_model)
    ema_model = copy.deepcopy(raw_model).eval()
    for prm in ema_model.parameters():
        prm.requires_grad_(False)
    no_decay, decay = [], []
    for name, prm in raw_model.named_parameters():
        (no_decay if prm.ndim <= 1 or "rel_" in name or "embed" in name else decay).append(prm)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                                   {"params": no_decay, "weight_decay": 0.0}],
                                  lr=args.lr, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema_src = [p.data for p in raw_model.parameters()]
    ema_dst = [p.data for p in ema_model.parameters()]

    def ema_step():
        with torch.no_grad():
            torch._foreach_mul_(ema_dst, args.ema_decay)
            torch._foreach_add_(ema_dst, ema_src, alpha=1 - args.ema_decay)

    def load_init_weights() -> dict:
        """Weights of the init checkpoint (raw or EMA), validated against this model."""
        from huggingface_hub import snapshot_download
        dest = outdir / "init"
        snapshot_download(args.hf_repo, repo_type="dataset", token=token, local_dir=str(dest),
                          allow_patterns=[f"{args.init_hf_run}/{args.init_checkpoint}/*"])
        init_dir = dest / args.init_hf_run / args.init_checkpoint
        init_cfg = json.loads((init_dir / "config.json").read_text(encoding="utf-8"))
        if (init_cfg.get("architecture") != "cc-gavn-v1"
                or init_cfg.get("model_version") != args.model_version
                or init_cfg.get("parameter_count") != params):
            raise ValueError("init checkpoint does not match the requested architecture")
        saved = torch.load(init_dir / "state.pt", map_location=device, weights_only=False)
        if args.init_weights == "ema" and "ema" not in saved:
            raise ValueError("init checkpoint has no EMA weights")
        return saved["ema" if args.init_weights == "ema" else "model"]

    # ---- resume or initialise
    start_step, baseline = 0, None
    resume_dir = (download_latest(hf, args.hf_repo, args.hf_run, outdir / "hf-resume")
                  if args.resume_from_hf else None)
    identity = {"tags_digest": tags_digest, "pack_digest": manifest["manifest_sha256"],
                "split_digest": manifest["val_csv_sha256"], "exclusion_digest": exclusion_digest}
    if ply_manifest is not None:          # only runs that use the ply pack carry its identity
        identity["ply_pack_digest"] = ply_manifest["manifest_sha256"]
    if resume_dir is not None:
        cfg = json.loads((resume_dir / "config.json").read_text(encoding="utf-8"))
        changed = [f"{k}: {cfg.get(k)!r} -> {getattr(args, k)!r}" for k in RECIPE_FIELDS
                   if k in cfg and cfg[k] != getattr(args, k)]
        changed += [f"{k}: changed" for k, v in identity.items() if cfg.get(k) != v]
        if changed and not args.allow_recipe_change:
            raise ValueError("resume would change the recipe or data identity: " + "; ".join(changed)
                             + " (start a new --hf-run, or pass --allow-recipe-change)")
        state = torch.load(resume_dir / "state.pt", map_location=device, weights_only=False)
        raw_model.load_state_dict(state["model"])
        ema_model.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state.get("scaler", {}))
        start_step, baseline = int(state["step"]), cfg.get("baseline")
        print(f"[resume] step={start_step}", flush=True)
    else:
        init_weights = load_init_weights()
        raw_model.load_state_dict(init_weights)
        ema_model.load_state_dict(init_weights)
        print(f"[init] {args.init_hf_run}/{args.init_checkpoint} ({args.init_weights} weights)",
              flush=True)
    ref_model = None
    if args.rl_weight:
        # The frozen reference for the KL anchor is the init weights, on every session.
        ref_model = copy.deepcopy(raw_model).eval()
        ref_model.load_state_dict(load_init_weights())
        for prm in ref_model.parameters():
            prm.requires_grad_(False)
    print(f"[train] device={device} params={params:,} lr={args.lr} ema={args.ema_decay} "
          f"broad_rows={args.broad_rows} tactical_groups={args.tactical_groups}"
          + (f" ply_groups={args.ply_groups}" if args.ply_groups else ""), flush=True)

    # ---- evaluation helpers
    def dev_listwise(pk, roots) -> tuple[float, float, float]:
        """Hit rate, regret and KL of the listwise loss on held-out roots of a pack."""
        toks, acts, wps, gid, lens = pk.gather(roots)
        qs = []
        with torch.inference_mode():
            for s in range(0, len(acts), 8192):
                sl = slice(s, s + 8192)
                qs.append(expected_value(torch, ema_model(
                    torch.as_tensor(toks[sl], dtype=torch.long, device=device),
                    torch.as_tensor(acts[sl], dtype=torch.long, device=device)), bucket_values))
            q = torch.cat(qs)
            _, kl, hit, regret = listwise_loss(
                torch, q, torch.as_tensor(wps, device=device),
                torch.as_tensor(gid, device=device), torch.as_tensor(lens, device=device),
                args.tau)
        return float(hit), float(regret), float(kl)

    def evaluate_dev() -> dict:
        out = {}
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                                    enabled=device.type == "cuda"):
            logp = torch.log_softmax(ema_model(dev_tokens[dev_pick], dev_actions[dev_pick]).float(), -1)
            dist = -(torch.softmax(dev_teacher[dev_pick], -1) * logp).sum(-1).mean()
            ce = -(hl_gauss(torch, dev_wp[dev_pick]) * logp).sum(-1).mean()
        out.update(dev_dist=float(dist), dev_ce=float(ce),
                   dev_loss=float(args.w_dist * dist + args.w_ce * ce))
        if tactical_dev_roots is not None:
            hit, regret, kl = dev_listwise(pack, tactical_dev_roots)
            out.update(tac_dev_hit=hit, tac_dev_regret=regret, tac_dev_kl=kl)
        if ply_dev_roots is not None:
            hit, regret, kl = dev_listwise(ply_pack, ply_dev_roots)
            out.update(ply_dev_hit=hit, ply_dev_regret=regret, ply_dev_kl=kl)
        return out

    val_detail: dict = {}      # latest per-puzzle validation results (only with --val-detail)

    def evaluate_val() -> dict:
        t0 = time.time()
        scorer = BatchedScorer(torch, ema_model, utils, device)
        res = evaluate_puzzles(scorer, val_puzzles)
        summary = summarize(res, val_puzzles)
        summary["seconds"] = round(time.time() - t0, 1)
        if args.val_detail:
            by_ply: dict[int, int] = {}
            for r in res:
                if not r["correct"] and r.get("fail_ply") is not None:
                    by_ply[r["fail_ply"]] = by_ply.get(r["fail_ply"], 0) + 1
            summary["failed_by_ply"] = {str(k): v for k, v in sorted(by_ply.items())}
            val_detail["results"] = res
        return summary

    def run_validation(step: int) -> dict:
        val = evaluate_val()
        metrics = {"val_step": step, "val_solved": val["solved"], "val_total": val["total"],
                   "val_pct": val["pct"],
                   "val_failed_first_decision": val.get("failed_at_first_decision"),
                   "val_failed_later": val.get("failed_later"),
                   **evaluate_dev()}
        if args.val_detail:
            metrics["val_failed_by_ply"] = val["failed_by_ply"]
            val_detail["step"] = step
        print(f"[val] step={step} puzzles={val['solved']}/{val['total']} ({val['pct']:.2f}%) "
              f"dev_loss={metrics['dev_loss']:.4f} dev_dist={metrics['dev_dist']:.4f} "
              f"tac_hit={metrics.get('tac_dev_hit', float('nan')):.3f} in {val['seconds']}s",
              flush=True)
        return metrics

    if baseline is None:
        baseline = run_validation(0)
        baseline["step"] = 0
        print(f"[baseline] {json.dumps(baseline)}", flush=True)

    # ---- batch construction
    state_cache = {"tag": None, "arrays": None, "mask": None, "prefetch": None}

    def load_broad(tag: str):
        if state_cache["tag"] == tag:
            return
        if state_cache["prefetch"] is not None:
            state_cache["prefetch"].join()
            state_cache["prefetch"] = None
        tokens, actions, winprob, teacher = manager.load(tag)
        if teacher.shape != (len(tokens), 128):
            raise ValueError(f"shard {tag}: invalid teacher shape {teacher.shape}")
        norm = np.logaddexp.reduce(np.asarray(teacher[:1024], dtype=np.float32), axis=1)
        if not np.allclose(norm, 0.0, atol=2e-3):
            raise ValueError(f"shard {tag}: teacher matrix is not normalized log-probabilities")
        mask = eligible_mask(tokens, excluded, 100, 0)
        state_cache.update(tag=tag, arrays=(tokens, actions, winprob, teacher), mask=mask)
        print(f"[data] shard {tag} train={mask.mean():.3%}", flush=True)

    def maybe_prefetch(step: int):
        done = step % args.block_steps
        if done < int(0.7 * args.block_steps) or state_cache["prefetch"] is not None:
            return
        nxt = broad_block_tag(step - done + args.block_steps, tags, args.block_steps, args.seed)
        if nxt != state_cache["tag"]:
            thread = threading.Thread(target=manager._download, args=(nxt,), daemon=True)
            thread.start()
            state_cache["prefetch"] = thread

    def make_batch(step: int):
        tag = broad_block_tag(step, tags, args.block_steps, args.seed)
        load_broad(tag)
        maybe_prefetch(step)
        tokens, actions, winprob, teacher = state_cache["arrays"]
        rng = step_rng(args.seed, step, 1)
        idx = np.sort(draw_rows(rng, len(tokens), args.broad_rows, state_cache["mask"]))
        bt = np.asarray(tokens[idx], dtype=np.int64)
        ba = np.asarray(actions[idx], dtype=np.int64)
        reflected = rng.random(len(idx)) < args.reflect_prob
        bt, ba = reflect_horizontal(bt, ba, action_map, reflected)
        batch = {"bt": bt, "ba": ba, "bw": np.asarray(winprob[idx], dtype=np.float32),
                 "bteach": np.asarray(teacher[idx], dtype=np.float32), "nb": len(idx)}
        if pack is not None:
            trng = step_rng(args.seed, step, 2)
            roots = pack.train_roots[trng.integers(0, len(pack.train_roots), size=args.tactical_groups)]
            tt, ta, tw, gid, lens = pack.gather(roots)
            flip = np.repeat(trng.random(len(roots)) < args.reflect_prob, lens)
            tt, ta = reflect_horizontal(tt.astype(np.int64), ta, action_map, flip)
            batch.update(tt=tt, ta=ta, tw=tw, gid=gid, lens=lens)
        if ply_pack is not None:
            # A third stream, so the broad and first-decision draws are untouched.
            prng = step_rng(args.seed, step, 3)
            proots = ply_pack.train_roots[prng.integers(0, len(ply_pack.train_roots),
                                                        size=args.ply_groups)]
            pt, pa, pw, pgid, plens = ply_pack.gather(proots)
            pflip = np.repeat(prng.random(len(proots)) < args.reflect_prob, plens)
            pt, pa = reflect_horizontal(pt.astype(np.int64), pa, action_map, pflip)
            if "tt" in batch:
                batch.update(tt=np.concatenate([batch["tt"], pt]),
                             ta=np.concatenate([batch["ta"], pa]),
                             tw=np.concatenate([batch["tw"], pw]),
                             gid=np.concatenate([batch["gid"], pgid + len(batch["lens"])]),
                             lens=np.concatenate([batch["lens"], plens]))
            else:
                batch.update(tt=pt, ta=pa, tw=pw, gid=pgid, lens=plens)
        return batch

    # ---- checkpointing
    timer = UploadTimer(args.hf_upload_every)
    timer.mark()

    def save_checkpoint(step: int, metrics: dict):
        ckpt = outdir / f"checkpoint-{step}"
        ckpt.mkdir(parents=True, exist_ok=True)
        torch.save({"model": raw_model.state_dict(), "ema": ema_model.state_dict(),
                    "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                    "step": step}, ckpt / "state.pt")
        cfg = {k: getattr(args, k) for k in RECIPE_FIELDS} | identity | {
            "architecture": "cc-gavn-v1", "model_version": args.model_version, "dim": args.dim,
            "layers": args.layers, "heads": args.heads, "parameter_count": params,
            "recipe_version": "cont1", "base_run": args.init_hf_run, "base_step": BASE_STEP,
            "step": step, "source_commit": source_commit,
            "searchless_commit": args.searchless_commit or None, "baseline": baseline,
            "weights_for_eval": "ema", "cli": vars(args)}
        (ckpt / "config.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")
        (ckpt / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
        if args.val_detail and val_detail.get("step") == step:
            results = val_detail["results"]
            (ckpt / "val-results.json").write_text(json.dumps({
                "step": step, "ids": [r["id"] for r in results],
                "correct": [int(bool(r["correct"])) for r in results],
                "fail_ply": [r.get("fail_ply") for r in results]}), encoding="utf-8")
        upload_checkpoint(hf, args.hf_repo, outdir, args.hf_run, ckpt.name)
        for old in sorted(outdir.glob("checkpoint-*"),
                          key=lambda p: int(p.name.rsplit("-", 1)[-1]))[:-2]:
            import shutil
            shutil.rmtree(old, ignore_errors=True)
        timer.mark()

    def official_eval(step: int):
        if args.no_official_eval:
            return
        t0 = time.time()
        scorer = BatchedScorer(torch, ema_model, utils, device)
        puzzles = official_puzzles(Path(args.exclusion_puzzles))
        pres = evaluate_puzzles(scorer, puzzles)
        mres = evaluate_mate(scorer, mate_rows(Path(args.mate_dir)))
        if len(mres) != 4000:
            raise ValueError(f"MATE scored {len(mres)} rows, expected 4000")
        summary = {"checkpoint": f"{args.hf_run}/checkpoint-{step}", "architecture": "cc-gavn-v1",
                   "model_version": args.model_version, "score": "dist", "weights": "ema",
                   "returncode": 0, "complete": True, "partial": False,
                   "mate": {"correct": sum(r["correct"] for r in mres), "total": len(mres),
                            "expected": 4000},
                   "puzzles": {**summarize(pres, puzzles), "expected": 10000},
                   "errors": [], "source_commit": source_commit,
                   "seconds": round(time.time() - t0, 1)}
        folder = outdir / "eval" / f"checkpoint-{step}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "eval-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (folder / "puzzles.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in pres), encoding="utf-8")
        from scripts.hf_upload import retry_hf_call
        retry_hf_call(lambda: hf.upload_folder(
            folder_path=str(folder), path_in_repo=f"eval-results/{args.hf_run}-{step // 1000}k-cont",
            repo_id=args.hf_repo, repo_type="dataset",
            commit_message=f"Official eval {args.hf_run} step {step}"), "upload official eval")
        print(f"[official] step={step} puzzles={summary['puzzles']['solved']}/10000 "
              f"mate={summary['mate']['correct']}/4000", flush=True)

    # ---- training loop
    guard_hits, last_metrics, nonfinite = 0, {}, 0
    step = start_step
    stop_reason = None
    t_loop = time.time()
    rows_seen = 0
    while True:
        if args.max_steps and step >= args.max_steps:
            stop_reason = "max_steps"
            break
        if deadline is not None and time.time() >= deadline:
            stop_reason = "session_end"
            break
        batch = make_batch(step)
        lr = lr_at(step, args)
        set_learning_rate(optimizer, lr)
        nb = batch["nb"]
        tokens_t = torch.as_tensor(batch["bt"], dtype=torch.long, device=device)
        actions_t = torch.as_tensor(batch["ba"], dtype=torch.long, device=device)
        if "tt" in batch:
            tokens_t = torch.cat([tokens_t, torch.as_tensor(batch["tt"], dtype=torch.long, device=device)])
            actions_t = torch.cat([actions_t, torch.as_tensor(batch["ta"], dtype=torch.long, device=device)])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(tokens_t, actions_t)
            ref_logits = None
            if ref_model is not None and "tt" in batch:
                with torch.no_grad():
                    ref_logits = ref_model(tokens_t[nb:], actions_t[nb:])
        if ref_logits is not None:
            batch["q_ref"] = expected_value(torch, ref_logits, bucket_values)   # fp32, no autocast
        loss, diag, total_rows = batch_loss(torch, logits, batch, args, bucket_values)
        if not torch.isfinite(loss):
            nonfinite += 1
            if nonfinite >= 20:
                raise RuntimeError("20 consecutive non-finite losses")
            step += 1
            continue
        nonfinite = 0
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0))
        scaler.step(optimizer)
        scaler.update()
        ema_step()
        step += 1
        rows_seen += total_rows
        val_due = step % args.val_every == 0
        milestone = bool(args.milestone_every and step % args.milestone_every == 0)
        ckpt_due = (step % args.ckpt_every == 0 or timer.due() or val_due or milestone)
        if step % 100 == 0 or val_due or ckpt_due:
            rate = rows_seen / max(1.0, time.time() - t_loop)
            last_metrics.update(train_loss=float(loss.detach()), train_dist=float(diag["dist"]),
                                train_list=float(diag["list"]), train_hit=float(diag["hit"]),
                                grad_norm=grad_norm, lr=lr, samples_per_s=rate)
            if step % 100 == 0:
                print(f"[train] step={step} loss={last_metrics['train_loss']:.4f} "
                      f"dist={last_metrics['train_dist']:.4f} ceb={float(diag['ce_b']):.4f} "
                      f"cet={float(diag['ce_t']):.4f} list={last_metrics['train_list']:.4f} "
                      f"hit={last_metrics['train_hit']:.3f} gn={grad_norm:.2f} lr={lr:.2e} "
                      f"rows/s={rate:.0f}", flush=True)
        if val_due or milestone:
            last_metrics.update(run_validation(step))
            if step >= args.guard_after and baseline.get("val_pct") is not None:
                if last_metrics["val_pct"] < baseline["val_pct"] - args.guard_drop:
                    guard_hits += 1
                else:
                    guard_hits = 0
                if guard_hits >= 2:
                    stop_reason = "guard"
        if ckpt_due or stop_reason:
            last_metrics["step"] = step
            save_checkpoint(step, last_metrics)
        if milestone:
            official_eval(step)
        if stop_reason:
            break

    # ---- shutdown: checkpoint first, then status, then a best-effort official eval
    if stop_reason in ("session_end", "max_steps"):
        last_metrics["step"] = step
        if not (outdir / f"checkpoint-{step}").exists():
            save_checkpoint(step, last_metrics)
    if stop_reason == "max_steps":
        write_status(hf, args.hf_repo, args.hf_run, f"DONE step={step} max_steps reached")
    elif stop_reason == "session_end":
        write_status(hf, args.hf_repo, args.hf_run,
                     f"IN PROGRESS checkpoint-{step}; session time reached")
    elif stop_reason == "guard":
        write_status(hf, args.hf_repo, args.hf_run,
                     f"HALTED step={step}: validation {last_metrics['val_pct']:.2f}% is more than "
                     f"{args.guard_drop} points below the {baseline['val_pct']:.2f}% baseline twice")
    if stop_reason in ("session_end", "max_steps") and (
            not args.milestone_every or step % args.milestone_every):
        try:
            official_eval(step)
        except Exception as exc:  # the checkpoint is safe; a failed eval must not fail the session
            print(f"[official] eval at session end failed: {type(exc).__name__}: {exc}", flush=True)
    print(f"[train] stop={stop_reason} step={step} after {(time.time()-started)/3600:.2f}h", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        try:
            import traceback

            def flag(name, default):
                key = f"--{name}"
                return sys.argv[sys.argv.index(key) + 1] if key in sys.argv else default
            write_status(make_hf_api(ROOT), flag("hf-repo", "vedangfake/chess-slm-benchmark"),
                         flag("hf-run", "ccgavn-cont"),
                         type(exc).__name__ + ": " + str(exc) + "\n" + traceback.format_exc()[-4000:])
        except Exception as status_exc:
            print(f"[hf] could not upload failure status: {status_exc}", flush=True)
        raise
