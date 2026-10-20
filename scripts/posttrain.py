"""Train/evaluate the compact refutation model with mandatory HF persistence.

Use train --mode planner, ranking, or distill. Distillation uses a frozen
planner online on the same eligible parent positions, with exact legal moves.
Each account resumes the same immutable recipe and RNG/data sampling stream.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import chess
import numpy as np
import torch

from scripts.eval_gavn import tokenize_fen, _sha256_file
from scripts.kaggle_checkpoint import api, download_latest, latest_remote_checkpoint, write_status
from scripts.refutation import ARCHITECTURE, load_network, objective, planned_logits, outcome


class Groups:
    """Offset index avoids loading a multi-million-position corpus into RAM."""
    def __init__(self, path):
        self.path = Path(path)
        self.offsets = {"train": [], "dev": []}
        with self.path.open("rb") as stream:
            while True:
                offset = stream.tell(); line = stream.readline()
                if not line:
                    break
                row = json.loads(line)
                if "branches" in row:
                    self.offsets[row["split"]].append(offset)
        if not all(self.offsets.values()):
            raise ValueError("dataset requires nonempty train and development splits")

    def read(self, offsets):
        with self.path.open("rb") as stream:
            rows = []
            for offset in offsets:
                stream.seek(offset); rows.append(json.loads(stream.readline()))
            return rows


def pack(groups, utils, horizon, width, device):
    roots, actions, values, utilities, paths, slices = [], [], [], [], [], []
    for group in groups:
        start = len(roots)
        for branch in group["branches"]:
            roots.append(tokenize_fen(group["fen"]))
            actions.append(utils.MOVE_TO_ACTION[branch["action"]])
            values.append(branch["target"]["q"])
            utilities.append(branch["target"]["utility"])
            if len(branch["paths"]) < width:
                raise ValueError("data has fewer refutation paths than model")
            paths.append(branch["paths"][:width])
        slices.append(slice(start, len(roots)))
    n = len(roots)
    future = np.empty((n, width, horizon, 77), dtype=np.int64)
    active = np.zeros((n, width, horizon), bool)
    terminal = np.full((n, width, horizon), np.nan, dtype=np.float32)
    q = np.zeros((n, width, horizon), np.float32)
    legal = np.zeros((n, width, horizon, utils.NUM_ACTIONS), bool)
    policy = np.full((n, width, horizon), -1, np.int64)
    mate = np.zeros_like(policy)
    for i, branches in enumerate(paths):
        for k, path in enumerate(branches):
            for t in range(horizon):
                point = path[min(t, len(path)-1)]
                board = chess.Board(point["fen"])
                if t >= len(path) and not board.is_game_over():
                    raise ValueError("nonterminal truncated trajectory")
                future[i, k, t] = tokenize_fen(point["fen"])
                active[i, k, t] = t < len(path)
                # Child side is opposite root, and side alternates per step.
                root_turn = not chess.Board(path[0]["fen"]).turn
                terminal[i, k, t] = outcome(board, root_turn)
                q[i, k, t] = point["q"]
                mate[i, k, t] = point["mate"]
                if t < len(path) and point["move"] is not None:
                    for move in board.legal_moves:
                        legal[i, k, t, utils.MOVE_TO_ACTION[move.uci()]] = True
                    policy[i, k, t] = utils.MOVE_TO_ACTION[point["move"]]
                    if not legal[i, k, t, policy[i, k, t]]:
                        raise ValueError("illegal supervised policy target")
    result = dict(root=np.stack(roots), actions=np.asarray(actions), q=np.asarray(values, np.float32),
                  utility=np.asarray(utilities, np.float32), future=future, active=active,
                  terminal=terminal, state_q=q, legal=legal, policy_target=policy, mate_target=mate)
    return {k: torch.as_tensor(v, device=device) for k, v in result.items()} | {"groups": slices}


def source_digest():
    digest = hashlib.sha256()
    for name in ("posttrain.py", "refutation.py", "refutation_data.py", "train_ccgavn.py",
                 "train_gavn.py", "data_hygiene.py", "eval_gavn.py", "kaggle_checkpoint.py"):
        digest.update(name.encode()); digest.update((ROOT/"scripts"/name).read_bytes())
    return digest.hexdigest()


@torch.no_grad()
def teacher_targets(teacher, groups, utils):
    boards, moves, tokens, actions = [], [], [], []
    device = next(teacher.parameters()).device
    for group in groups:
        for branch in group["branches"]:
            board = chess.Board(group["fen"]); move = chess.Move.from_uci(branch["action"])
            boards.append(board); moves.append(move); tokens.append(tokenize_fen(board.fen()))
            actions.append(utils.MOVE_TO_ACTION[move.uci()])
    return planned_logits(teacher, boards, moves, utils).float().softmax(-1)


@torch.no_grad()
def development(network, data, utils, count, mode, device):
    """Fixed held-out groups; planner metrics use predicted, not oracle futures."""
    from scripts.refutation import score_moves
    network.eval()
    correct = total = 0; regret = 0.
    for row in data.read(data.offsets["dev"][:count]):
        board = chess.Board(row["fen"])
        branches = row["branches"]
        scores = score_moves(network, board, [chess.Move.from_uci(b["action"]) for b in branches],
                             utils, mode="planner" if mode == "planner" else "base")
        truths = np.array([b["target"]["utility"] for b in branches])
        best, picked = truths.max(), truths[int(scores.argmax())]
        correct += int(best-picked < 1e-6); total += 1; regret += best-picked
    return {"groups": total, "oracle_group_top1": correct/total, "utility_regret": regret/total}


def train(args):
    if not args.hf_run.startswith("ccgavn-refutation-"):
        raise ValueError("use a fresh ccgavn-refutation-* prefix")
    client = api(ROOT)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    if args.mode == "distill" and not args.teacher:
        raise ValueError("distillation requires --teacher with a trained planner")
    if args.steps < 1 or args.batch_groups < 1 or args.accumulate < 1 or args.warmup_steps < 0:
        raise ValueError("invalid optimization budget")
    if min(args.hf_upload_every, args.session_hours, args.eval_every, args.dev_groups) <= 0:
        raise ValueError("persistence/evaluation budgets must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("T4 training requires CUDA; no silent CPU fallback")
    device = torch.device("cuda")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    network, base_cfg, utils = load_network(args.init_checkpoint, args.sl_repo, device)
    if args.horizon is not None:
        if args.horizon < 1:
            raise ValueError("horizon must be positive")
        network.horizon = args.horizon
    if args.width is not None:
        network.width = args.width
    # All new heads are initialized deterministically after the run seed.
    manifest_path = Path(args.data).with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if network.horizon > manifest["horizon"] or network.width > manifest["width"]:
        raise ValueError("dataset horizon/width is insufficient")
    recipe = {k: getattr(args, k) for k in ("mode", "seed", "steps", "lr", "batch_groups", "accumulate",
                                           "warmup_steps", "freeze_steps", "eval_every", "dev_groups")}
    recipe.update(data_sha256=_sha256_file(Path(args.data)), manifest_sha256=_sha256_file(manifest_path),
                  init_sha256=_sha256_file(Path(args.init_checkpoint)/"state.pt"),
                  teacher_sha256=_sha256_file(Path(args.teacher)/"state.pt") if args.teacher else None,
                  source_sha256=source_digest(), torch_version=torch.__version__,
                  chess_version=chess.__version__, gpu_count=torch.cuda.device_count())
    recipe["replay_sha256"] = [_sha256_file(Path(p)) for p in args.replay_data]
    recipe["replay_fraction"] = args.replay_fraction
    recipe["action_map_sha256"] = hashlib.sha256(json.dumps(
        [utils.ACTION_TO_MOVE[i] for i in range(utils.NUM_ACTIONS)]).encode()).hexdigest()
    config = {k: base_cfg[k] for k in ("dim", "layers", "heads", "model_version") if k in base_cfg}
    config.update(architecture=ARCHITECTURE, recipe=recipe, horizon=network.horizon, width=network.width,
                  inference_mode="planner" if args.mode == "planner" else "base",
                  num_parameters=network.parameter_count, source_commit=base_cfg.get("source_commit"))
    data = Groups(args.data)
    replay = [Groups(p) for p in args.replay_data]
    if not 0 <= args.replay_fraction < 1:
        raise ValueError("replay fraction must be in [0, 1)")
    # Prior journals must have participated in the builder's cross-round split
    # ownership check. This is part of the immutable dataset manifest.
    prior = manifest.get("prior_data_sha256", [])
    if any(h not in prior for h in recipe["replay_sha256"]):
        raise ValueError("replay data was not included in builder --prior-data")
    anchor = copy.deepcopy(network.base).eval().requires_grad_(False)
    teacher = None
    if args.teacher:
        teacher, tcfg, _ = load_network(args.teacher, args.sl_repo, device)
        if tcfg.get("architecture") != ARCHITECTURE or tcfg.get("inference_mode") != "planner":
            raise ValueError("distillation teacher must be a planner checkpoint")
        teacher.eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda")
    step = 0; metrics = {}
    if args.resume_from_hf:
        cp = download_latest(client, args.hf_repo, args.hf_run, out/"resume")
        if cp is None:
            raise ValueError("resume requested but no complete remote checkpoint")
        saved_cfg = json.loads((cp/"config.json").read_text())
        if saved_cfg != config:
            raise ValueError("resume recipe, data, software or checkpoint identity changed")
        saved = torch.load(cp/"state.pt", map_location="cpu", weights_only=False)
        network.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"]); scaler.load_state_dict(saved["scaler"])
        step = saved["step"]; rng.bit_generator.state = saved["numpy_rng"]
        random.setstate(saved["python_rng"]); torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"]); metrics = saved["metrics"]
    elif latest_remote_checkpoint(client, args.hf_repo, args.hf_run) is not None:
        raise ValueError("remote run already exists: resume it or use a fresh prefix")
    # DataParallel uses both T4s when provided; the raw state has no module prefix.
    parallel = torch.nn.DataParallel(network) if torch.cuda.device_count() > 1 else network
    started = last_upload = time.monotonic()
    total_branches = 0
    last_saved_step = step if args.resume_from_hf else None
    session_start_step = step
    def save():
        nonlocal last_saved_step
        if step == last_saved_step:
            return
        cp = out/f"checkpoint-{step}"
        cp.mkdir(exist_ok=True)
        torch.save({"model": network.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "step": step, "numpy_rng": rng.bit_generator.state,
                    "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(), "metrics": metrics}, cp/"state.pt")
        (cp/"config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
        # One commit makes config and state visible together; never overwrite a
        # remote checkpoint unless it is the identical resumed step.
        client.upload_folder(repo_id=args.hf_repo, repo_type="dataset", folder_path=str(cp),
                             path_in_repo=f"{args.hf_run}/{cp.name}")
        progress = dict(metrics, step=step, total_steps=args.steps,
                        session_start_step=session_start_step,
                        session_seconds=round(time.monotonic()-started, 3),
                        updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        (out/"progress.json").write_text(json.dumps(progress, indent=2, sort_keys=True))
        client.upload_file(path_or_fileobj=str(out/"progress.json"),
                           path_in_repo=f"{args.hf_run}/progress.json",
                           repo_id=args.hf_repo, repo_type="dataset")
        last_saved_step = step
    try:
        write_status(client, args.hf_repo, args.hf_run, "IN PROGRESS: post-training")
        save()  # initial resumable state before any expensive updates
        while step < args.steps:
            network.train()
            network.base.requires_grad_(step >= args.freeze_steps or args.mode != "planner")
            optimizer.zero_grad(set_to_none=True)
            warmup = min(1., (step+1)/max(1, args.warmup_steps))
            decay = .5*(1+math.cos(math.pi*max(0, step-args.warmup_steps)/max(1, args.steps-args.warmup_steps)))
            lr = args.lr*warmup*decay
            for group in optimizer.param_groups:
                group["lr"] = lr
            loss_total = 0.
            for _ in range(args.accumulate):
                rows = []
                for _ in range(args.batch_groups):
                    corpus = replay[int(rng.integers(len(replay)))] if replay and rng.random() < args.replay_fraction else data
                    idx = int(rng.integers(len(corpus.offsets["train"])))
                    rows.extend(corpus.read([corpus.offsets["train"][idx]]))
                batch = pack(rows, utils, network.horizon, network.width, device)
                if args.mode == "distill":
                    batch["teacher"] = teacher_targets(teacher, rows, utils)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    reference = anchor(batch["root"], batch["actions"])
                with torch.autocast("cuda", dtype=torch.float16):
                    paths = (batch["future"], batch["active"], batch["terminal"]) if args.mode == "planner" else ()
                    outputs = parallel(batch["root"], batch["actions"], *paths)
                    loss = objective(outputs, batch, network.buckets, reference, args.mode)/args.accumulate
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite loss; recipe unchanged")
                scaler.scale(loss).backward()
                loss_total += float(loss.detach()); total_branches += len(batch["actions"])
            scaler.unscale_(optimizer)
            grad = torch.nn.utils.clip_grad_norm_(network.parameters(), 1.)
            if not torch.isfinite(grad):
                raise FloatingPointError("non-finite gradient; recipe unchanged")
            scaler.step(optimizer); scaler.update()
            step += 1
            metrics = {"step": step, "loss": loss_total, "lr": lr,
                       "branch_rows_per_second": total_branches/(time.monotonic()-started),
                       "session_seconds": time.monotonic()-started,
                       "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                       "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
            if step % args.eval_every == 0:
                metrics["dev"] = development(network, data, utils, args.dev_groups, args.mode, device)
            if step % 50 == 0 or "dev" in metrics:
                print(json.dumps(metrics), flush=True)
            if time.monotonic()-last_upload >= args.hf_upload_every or step == args.steps:
                save(); last_upload = time.monotonic()
            if time.monotonic()-started >= args.session_hours*3600:
                break
        save()
        write_status(client, args.hf_repo, args.hf_run,
                     "DONE: post-training complete" if step == args.steps else f"PAUSED: checkpoint-{step}; resume required")
    except BaseException as exc:
        # Only committed optimizer steps are checkpoints. Never persist partial
        # accumulated gradients as though they were a completed update.
        write_status(client, args.hf_repo, args.hf_run, f"FAILED at step {step}: {type(exc).__name__}: {exc}")
        raise


def evaluate(args):
    client = api(ROOT)
    cp = Path(args.checkpoint)
    digest = hashlib.sha256()
    for path in [cp/"state.pt", cp/"config.json", Path(args.puzzles),
                 ROOT/"scripts/eval_gavn.py", ROOT/"scripts/refutation.py",
                 *[Path(p) for p in args.mate_files.split(",")]]:
        digest.update(_sha256_file(path).encode())
    key = digest.hexdigest()[:20]
    prefix = f"{args.hf_run}/eval-{key}"
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    if args.resume_from_hf:
        from huggingface_hub import snapshot_download
        snap = snapshot_download(args.hf_repo, repo_type="dataset", token=client.token,
                                 allow_patterns=[f"{prefix}/*"])
        remote = Path(snap)/prefix
        if not remote.exists():
            raise ValueError("requested eval resume has no remote rows")
        for path in remote.rglob("*"):
            if path.is_file():
                dest = out/path.relative_to(remote)
                if dest.exists() and dest.read_bytes() != path.read_bytes():
                    raise ValueError("local and remote eval conflict; use a fresh outdir")
                dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dest)
    elif any(out.iterdir()):
        raise ValueError("nonempty evaluation directory requires explicit resume")
    # Evaluator identity pins state, config, datasets, and scoring mode.
    command = [sys.executable, str(ROOT/"scripts/eval_gavn.py"), "--checkpoint", str(cp),
               "--sl-repo", args.sl_repo, "--eval", args.mate_files, "--puzzles", args.puzzles,
               "--examples-out", str(out/"examples"), "--summary-out", str(out/"summary.json")]
    process = None
    def upload():
        # Copy only complete JSONL lines so a concurrent append cannot publish
        # a truncated row. The whole snapshot is uploaded in one HF commit.
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)
            for path in out.rglob("*"):
                if path.is_file():
                    raw = path.read_bytes()
                    if path.suffix == ".jsonl":
                        raw = raw[:raw.rfind(b"\n")+1]
                    dest = target/path.relative_to(out); dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(raw)
            client.upload_folder(repo_id=args.hf_repo, repo_type="dataset", folder_path=temp,
                                 path_in_repo=prefix)
    try:
        write_status(client, args.hf_repo, prefix, "IN PROGRESS: full 4000/10000 evaluation")
        process = subprocess.Popen(command, cwd=ROOT)
        last, start = time.monotonic(), time.monotonic()
        while process.poll() is None:
            time.sleep(5)
            if time.monotonic()-last >= args.hf_upload_every:
                upload(); last = time.monotonic()
            if time.monotonic()-start >= args.session_hours*3600:
                process.terminate(); process.wait(timeout=30); upload()
                write_status(client, args.hf_repo, prefix, "PAUSED: evaluation resume required")
                return
        upload()
        if process.returncode != 0:
            raise RuntimeError(f"evaluation exited {process.returncode}")
        summary = json.loads((out/"summary.json").read_text())
        if not summary["complete"] or summary["mate"]["total"] != 4000 or summary["puzzles"]["total"] != 10000:
            raise ValueError("incomplete evaluation cannot be marked DONE")
        write_status(client, args.hf_repo, prefix, "DONE: full 4000/10000 evaluation")
    except BaseException as exc:
        if process is not None and process.poll() is None:
            process.terminate(); process.wait(timeout=30)
        write_status(client, args.hf_repo, prefix, f"FAILED: {type(exc).__name__}: {exc}")
        raise


def export(args):
    network, cfg, _ = load_network(args.checkpoint, args.sl_repo)
    if cfg.get("inference_mode") != "base":
        raise ValueError("export requires a ranking or distilled one-pass checkpoint")
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=False)
    cfg.update(architecture="cc-gavn-v1", num_parameters=sum(p.numel() for p in network.base.parameters()),
               export_kind="inference-only", parent_state_sha256=_sha256_file(Path(args.checkpoint)/"state.pt"))
    for key in ("horizon", "width", "inference_mode"):
        cfg.pop(key, None)
    torch.save({"model": network.base.state_dict()}, out/"state.pt")
    (out/"config.json").write_text(json.dumps(cfg, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    tr = commands.add_parser("train")
    tr.add_argument("--init-checkpoint", required=True)
    tr.add_argument("--data", required=True)
    tr.add_argument("--mode", choices=["planner", "ranking", "distill"], default="planner")
    tr.add_argument("--teacher")
    tr.add_argument("--horizon", type=int)
    tr.add_argument("--width", type=int, choices=[1, 2])
    tr.add_argument("--replay-data", nargs="*", default=[])
    tr.add_argument("--replay-fraction", type=float, default=.3)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--steps", type=int, default=20000)
    tr.add_argument("--lr", type=float, default=3e-5)
    tr.add_argument("--batch-groups", type=int, default=2)
    tr.add_argument("--accumulate", type=int, default=16)
    tr.add_argument("--warmup-steps", type=int, default=500)
    tr.add_argument("--freeze-steps", type=int, default=1000)
    tr.add_argument("--eval-every", type=int, default=500)
    tr.add_argument("--dev-groups", type=int, default=256)
    ev = commands.add_parser("eval")
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--puzzles", required=True)
    ev.add_argument("--mate-files", default=",".join(str(ROOT/"data/positions"/n) for n in (
        "mate-selection-test.json", "mate-selection-test-noexplain.json",
        "mate-selection-test-tactic.json", "mate-selection-test-both.json")))
    ex = commands.add_parser("export")
    ex.add_argument("--checkpoint", required=True)
    ex.add_argument("--outdir", required=True)
    ex.add_argument("--sl-repo", default="/kaggle/working/searchless_chess")
    for cmd in (tr, ev):
        cmd.add_argument("--sl-repo", default="/kaggle/working/searchless_chess")
        cmd.add_argument("--outdir", required=True)
        cmd.add_argument("--hf-repo", default="vedangfake/chess-slm-benchmark")
        cmd.add_argument("--hf-run", required=True)
        cmd.add_argument("--hf-upload-every", type=float, default=900)
        cmd.add_argument("--resume-from-hf", action="store_true")
        cmd.add_argument("--session-hours", type=float, default=10.5)
    args = parser.parse_args()
    if args.command == "export":
        export(args)
        return
    if args.hf_upload_every <= 0 or args.session_hours <= 0:
        raise ValueError("persistence and session budgets must be positive")
    try:
        (train if args.command == "train" else evaluate)(args)
    except BaseException as exc:
        # Includes preflight failures before the inner training/eval loops.
        write_status(api(ROOT), args.hf_repo, args.hf_run,
                     f"FAILED: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
