"""All-legal-action root-policy RL for repaired CC-GAVN on one Kaggle T4.

The policy is a softmax over the model's expected return for every legal move.
Groups sampled from that policy receive exact-depth Stockfish rewards; their
relative advantages drive a clipped policy update, anchored to the frozen
repaired checkpoint by policy KL and verifier value calibration. Inference is
unchanged: one model pass per legal candidate, no engine or tree search.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download

from scripts.data_hygiene import load_exclusions, position_hashes
from scripts.eval_gavn import tokenize_fen
from scripts.kaggle_checkpoint import (UploadTimer, api as make_api,
                                       download_latest, upload_checkpoint,
                                       write_status)
from scripts.repair_checkpoint import make_model, repair

ROOT = Path(__file__).resolve().parent.parent
AMP_INIT_SCALE = 128.0
AMP_MAX_SCALE_RETRIES = 12
# This exact predecessor has the identical RL recipe/state layout; only its
# per-file Hub upload path differed. Permit resuming its complete optimizer
# checkpoint after switching checkpoint persistence to one commit per folder.
RESUME_COMPATIBLE_COMMITS = {"b9d1fd7b9923f6c83efc7b73b60c1fc25161ee0f"}


def gradient_issues(model):
    named = list(model.named_parameters())
    missing = [name for name, parameter in named if parameter.grad is None]
    with_grad = [(name, parameter) for name, parameter in named
                 if parameter.grad is not None]
    flags = torch.stack([~torch.isfinite(parameter.grad).all()
                         for _, parameter in with_grad]).detach().cpu().tolist()
    invalid = [name for (name, _), bad in zip(with_grad, flags) if bad]
    return missing, invalid


def validate_optimizer_coverage(model, optimizer):
    expected = {id(parameter): name for name, parameter in model.named_parameters()
                if parameter.requires_grad}
    grouped = [parameter for group in optimizer.param_groups
               for parameter in group["params"]]
    actual = {id(parameter) for parameter in grouped}
    duplicates = len(grouped) - len(actual)
    missing = [name for identity, name in expected.items() if identity not in actual]
    extra = sum(id(parameter) not in expected for parameter in grouped)
    if duplicates or missing or extra:
        raise RuntimeError(
            f"optimizer coverage mismatch; duplicates={duplicates}; "
            f"missing={missing}; extra_tensors={extra}")


def development_fold(position_hash: int) -> int:
    digest = hashlib.sha256(str(int(position_hash)).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") % 10


def source_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                   text=True).strip()


def load_labels(api, repo: str, manifest: dict, excluded: np.ndarray, utils):
    if (manifest.get("reward") != "searchless-centipawns-to-win-probability-v1"
            or manifest.get("searchless_commit") is None):
        raise ValueError("manifest lacks the corrected verifier reward identity")
    records, seen = [], set()
    for account, source in sorted(manifest["sources"].items()):
        remote = f"{manifest['label_prefix']}/{account}/labels.jsonl"
        cached = hf_hub_download(repo, remote, repo_type="dataset", token=api.token)
        count = int(source["count"])
        with open(cached, "rb") as fh:
            lines = [fh.readline() for _ in range(count)]
        if any(not line for line in lines):
            raise ValueError(f"{account}: fewer labels than frozen manifest")
        raw = b"".join(lines)
        if hashlib.sha256(raw).hexdigest() != source["sha256"]:
            raise ValueError(f"{account}: label prefix changed after manifest freeze")
        for line in lines:
            record = json.loads(line)
            if record["depth"] != manifest["depth"] or len(record["actions"]) < 2:
                raise ValueError(f"{account}: wrong depth or too few legal actions")
            fen = record["fen"]
            board = chess.Board(fen)
            moves = [move.uci() for move in board.legal_moves]
            if moves != record["moves"]:
                raise ValueError(f"{account}: verifier labels do not cover all legal moves")
            expected_actions = [int(utils.MOVE_TO_ACTION[move]) for move in moves]
            if expected_actions != record["actions"]:
                raise ValueError(f"{account}: verifier action IDs do not match move order")
            if len(record["actions"]) != len(moves) or len(record["rewards"]) != len(moves):
                raise ValueError(f"{account}: action/reward length mismatch")
            tokens = tokenize_fen(fen)
            if not (0 <= int(fen.split()[4]) < 1000 and 1 <= int(fen.split()[5]) < 1000):
                raise ValueError(f"{account}: non-roundtrippable FEN move counters")
            position_hash = int(position_hashes(tokens[None])[0])
            if position_hash != int(record["position_hash"]) or position_hash in excluded:
                raise ValueError(f"{account}: label identity or frozen exclusion failure")
            if position_hash in seen:
                raise ValueError("duplicate position across frozen label sources")
            seen.add(position_hash)
            rewards = np.asarray(record["rewards"], dtype=np.float32)
            if not np.all(np.isfinite(rewards)) or np.any(rewards < 0) or np.any(rewards > 1):
                raise ValueError("invalid verifier rewards")
            if record.get("engine") is None or len(record["scores_cp"]) != len(moves):
                raise ValueError(f"{account}: missing exact verifier score identity")
            expected_rewards = np.asarray([
                utils.centipawns_to_win_probability(int(score))
                for score in record["scores_cp"]], dtype=np.float32)
            if not np.allclose(rewards, expected_rewards, rtol=0.0, atol=1e-6):
                raise ValueError(f"{account}: verifier rewards disagree with frozen utility")
            records.append((tokens, np.asarray(record["actions"], dtype=np.int64), rewards,
                            position_hash))
    if len(records) < 1000:
        raise ValueError(f"only {len(records)} verified roots; require at least 1000")
    train = [row for row in records if development_fold(row[3]) != 0]
    dev = [row for row in records if development_fold(row[3]) == 0]
    if len(dev) < 500:
        raise ValueError(f"only {len(dev)} development roots; require at least 500")
    return train, dev


def flatten_batch(batch, device):
    tokens = np.concatenate([np.repeat(row[0][None, :], len(row[1]), axis=0)
                             for row in batch])
    actions = np.concatenate([row[1] for row in batch])
    rewards = np.concatenate([row[2] for row in batch])
    offsets = np.cumsum([0] + [len(row[1]) for row in batch]).tolist()
    return (torch.as_tensor(tokens, dtype=torch.long, device=device),
            torch.as_tensor(actions, dtype=torch.long, device=device),
            torch.as_tensor(rewards, dtype=torch.float32, device=device), offsets)


def policy_logprobs(values: torch.Tensor, offsets: list[int], temperature: float):
    return [torch.log_softmax(values[a:b] / temperature, dim=0)
            for a, b in zip(offsets[:-1], offsets[1:])]


def values_for(model, tokens, actions, bucket_values, device):
    with torch.autocast(device_type=device.type, dtype=torch.float16,
                        enabled=device.type == "cuda"):
        logits = model(tokens, actions)
    return torch.softmax(logits.float(), dim=-1) @ bucket_values


def objective(values, ref_logs, rewards, offsets, args):
    logs = policy_logprobs(values, offsets, args.policy_temperature)
    policy_losses, kl_terms, value_terms, entropy_terms = [], [], [], []
    for group, (a, b) in enumerate(zip(offsets[:-1], offsets[1:])):
        probs = logs[group].exp()
        centered = rewards[a:b] - rewards[a:b].mean()
        centered = centered / centered.std(unbiased=False).clamp_min(1e-4)
        # Exact counterfactual expectation over every legal action; there is
        # no replacement sampling or sampled PPO ratio in this root bandit.
        policy_losses.append(-(probs * centered).sum())
        kl_terms.append((probs * (logs[group] - ref_logs[group])).sum())
        value_terms.append(torch.mean((values[a:b] - rewards[a:b]) ** 2))
        entropy_terms.append(-(probs * logs[group]).sum())
    pg = torch.stack(policy_losses).mean()
    kl = torch.stack(kl_terms).mean()
    value = torch.stack(value_terms).mean()
    entropy = torch.stack(entropy_terms).mean()
    return (pg + args.kl_coef * kl + args.value_coef * value
            - args.entropy_coef * entropy), (pg, kl, value, entropy)


def evaluate_dev(model, dev, bucket_values, device, args) -> dict:
    model.eval()
    correct, regret, count = 0, 0.0, 0
    with torch.inference_mode():
        for row in dev:
            tokens, actions, rewards, offsets = flatten_batch([row], device)
            values = values_for(model, tokens, actions, bucket_values, device)
            prediction = int(torch.argmax(values))
            correct += int(prediction == int(torch.argmax(rewards)))
            regret += float(rewards.max() - rewards[prediction])
            count += 1
    model.train()
    return {"roots": count, "top1": correct / count, "mean_regret": regret / count}


def save_checkpoint(api, args, model, reference, optimizer, scheduler, scaler, rng,
                    manifest, manifest_sha, step, metrics, repair_report):
    folder = args.outdir / f"checkpoint-{step}"
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "reference": reference.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step, "numpy_rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
               folder / "state.pt")
    config = {"architecture": "cc-gavn-v1", "model_version": "v2",
              "recipe_version": "rl-root-v4", "dim": 208, "layers": 8, "heads": 8,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "trainable_parameter_count": sum(p.numel() for p in model.parameters()
                                               if p.requires_grad),
              "optimized_parameter_count": sum(p.numel() for group in optimizer.param_groups
                                                for p in group["params"]),
              "trainable_tensor_count": sum(1 for p in model.parameters() if p.requires_grad),
              "optimized_tensor_count": sum(len(group["params"])
                                             for group in optimizer.param_groups),
              "source_commit": source_commit(), "source_checkpoint": args.init_run + "/" + args.init_checkpoint,
              "searchless_commit": manifest["searchless_commit"],
              "repair": repair_report, "manifest_sha256": manifest_sha,
              "exclusion_digest": manifest["exclusion_digest"],
              "canonical_decision_head": "distribution_expectation",
              "algorithm": "exact-all-legal-counterfactual-policy-gradient",
              "label_source_commit": args.label_source_commit,
              "amp_init_scale": AMP_INIT_SCALE,
              "amp_max_scale_retries": AMP_MAX_SCALE_RETRIES,
              "steps": args.steps, "batch_positions": args.batch_positions,
              "warmup_steps": min(args.warmup_steps, args.steps - 1),
              "lr": args.lr,
              "kl_coef": args.kl_coef, "value_coef": args.value_coef,
              "entropy_coef": args.entropy_coef,
              "policy_temperature": args.policy_temperature,
              "seed": args.seed}
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (folder / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    upload_checkpoint(api, args.hf_repo, args.outdir, args.hf_run, folder.name)
    write_status(api, args.hf_repo, args.hf_run,
                 f"IN PROGRESS {step}/{args.steps} dev_top1={metrics.get('dev', {}).get('top1', 'n/a')}")
    for old in sorted(args.outdir.glob("checkpoint-*"),
                      key=lambda path: int(path.name.rsplit("-", 1)[-1]))[:-2]:
        import shutil
        shutil.rmtree(old)


def run(args):
    if args.policy_temperature <= 0:
        raise ValueError("policy temperature must be positive")
    if args.steps < 2 or args.batch_positions < 1:
        raise ValueError("invalid training budget")
    api = make_api(ROOT)
    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest_bytes = args.manifest.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    if manifest.get("source_commit") != args.label_source_commit:
        raise ValueError("frozen verifier source commit differs from label identity")
    if manifest.get("searchless_commit") != args.searchless_commit:
        raise ValueError("frozen verifier Searchless commit differs from learner")
    excluded, exclusion_digest = load_exclusions(args.mate_dir, args.puzzles)
    if manifest["exclusion_digest"] != exclusion_digest:
        raise ValueError("frozen-label and training exclusions differ")
    from searchless_chess.src import utils
    train, dev = load_labels(api, args.hf_repo, manifest, excluded, utils)
    print(f"[rl] verified {len(train)} train and {len(dev)} dev roots", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("RL learner requires a Kaggle T4 GPU")
    bucket_values = torch.as_tensor(np.asarray(
        utils.get_uniform_buckets_edges_values(128)[1], dtype=np.float32), device=device)
    rng = np.random.default_rng(args.seed)
    resume = download_latest(api, args.hf_repo, args.hf_run, args.outdir / "hf-resume")
    repair_report = None
    if resume is None:
        snapshot_download(args.hf_repo, repo_type="dataset", token=api.token,
                          local_dir=str(args.outdir / "source"), allow_patterns=[
                              f"{args.init_run}/{args.init_checkpoint}/*"])
        source = args.outdir / "source" / args.init_run / args.init_checkpoint
        repair_report = repair(source, args.outdir / "repaired", args.sl_repo)
        cfg = json.loads((source / "config.json").read_text(encoding="utf-8"))
        model, _ = make_model(cfg, "v2", args.sl_repo)
        weights = torch.load(args.outdir / "repaired/weights.pt", map_location="cpu",
                             weights_only=True)["model"]
        model.load_state_dict(weights)
        reference, _ = make_model(cfg, "v2", args.sl_repo)
        reference.load_state_dict(weights)
        model.to(device); reference.to(device)
        start_step = 0
    else:
        cfg = json.loads((resume / "config.json").read_text(encoding="utf-8"))
        if (cfg["recipe_version"] != "rl-root-v4" or cfg["manifest_sha256"] != manifest_sha
                or cfg["source_commit"] not in {source_commit(), *RESUME_COMPATIBLE_COMMITS}
                or cfg["label_source_commit"] != args.label_source_commit
                or cfg["searchless_commit"] != args.searchless_commit
                or cfg["steps"] != args.steps):
            raise ValueError("RL resume identity, source code, or budget changed")
        repair_report = cfg["repair"]
        model, _ = make_model(cfg, "v2", args.sl_repo)
        reference, _ = make_model(cfg, "v2", args.sl_repo)
        state = torch.load(resume / "state.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"]); reference.load_state_dict(state["reference"])
        model.to(device); reference.to(device)
        start_step = int(state["step"])
        rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(state["torch_rng"])
        if state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    if sum(p.numel() for p in model.parameters()) >= 5_000_000:
        raise ValueError("model exceeds 5M parameter gate")
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    no_decay, decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if parameter.ndim <= 1 or "rel_" in name or "embed" in name
         else decay).append(parameter)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01},
                                   {"params": no_decay, "weight_decay": 0.0}],
                                  lr=args.lr, betas=(0.9, 0.95))
    validate_optimizer_coverage(model, optimizer)
    warmup = min(args.warmup_steps, args.steps - 1)
    def lr_scale(step):
        if step < warmup:
            return max(1, step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, args.steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    scaler = torch.amp.GradScaler("cuda", init_scale=AMP_INIT_SCALE, enabled=True)
    if resume is not None:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
    else:
        save_checkpoint(api, args, model, reference, optimizer, scheduler, scaler, rng,
                        manifest, manifest_sha, 0, {"step": 0, "dev": evaluate_dev(
                            model, dev, bucket_values, device, args)}, repair_report)
    timer = UploadTimer(args.upload_every)
    timer.mark()
    model.train()
    for step in range(start_step, args.steps):
        indices = rng.integers(0, len(train), size=args.batch_positions)
        batch = [train[int(index)] for index in indices]
        tokens, actions, rewards, offsets = flatten_batch(batch, device)
        with torch.no_grad():
            ref_values = values_for(reference, tokens, actions, bucket_values, device)
            ref_logs = policy_logprobs(ref_values, offsets, args.policy_temperature)
        before = ({name: parameter.detach().clone() for name, parameter in model.named_parameters()}
                  if step == 0 else None)
        for retry in range(AMP_MAX_SCALE_RETRIES + 1):
            optimizer.zero_grad(set_to_none=True)
            values = values_for(model, tokens, actions, bucket_values, device)
            loss, parts = objective(values, ref_logs, rewards, offsets, args)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite RL loss at step {step + 1}")
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if step == 0:
                missing = [name for name, parameter in model.named_parameters()
                           if parameter.grad is None]
                if missing:
                    raise RuntimeError(f"incomplete RL gradients; missing={missing}")
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            if scale_after >= scale_before:
                break
            _, invalid = gradient_issues(model)
            if retry >= AMP_MAX_SCALE_RETRIES:
                raise RuntimeError(
                    f"nonfinite RL gradients after {retry + 1} attempts; "
                    f"scale={scale_after}; invalid={invalid}")
            print(f"[rl] nonfinite gradients at step={step+1}; retry={retry+1}/"
                  f"{AMP_MAX_SCALE_RETRIES} scale={scale_after:g}", flush=True)
        else:
            raise RuntimeError(f"AMP retry loop exhausted at step {step + 1}")
        scheduler.step()
        if step == 0:
            changed = [name for name, parameter in model.named_parameters()
                       if not torch.equal(before[name], parameter.detach())]
            if not changed:
                raise RuntimeError("RL optimizer completed step 1 without changing any parameters")
            print(f"[rl] first optimizer update: changed {len(changed)}/"
                  f"{len(before)} parameter tensors; all trainable tensors were in the "
                  "optimizer and had finite gradients", flush=True)
        if (step + 1) % 20 == 0:
            print(f"[rl] step={step+1}/{args.steps} loss={loss.item():.4f} "
                  f"pg={parts[0].item():.4f} kl={parts[1].item():.4f} "
                  f"value={parts[2].item():.4f} grad={grad_norm:.3f}", flush=True)
        if (step + 1) % args.ckpt_every == 0 or step + 1 == args.steps or timer.due():
            metrics = {"step": step + 1, "loss": float(loss.detach()),
                       "pg": float(parts[0].detach()), "kl": float(parts[1].detach()),
                       "value": float(parts[2].detach()), "grad_norm": grad_norm,
                       "amp_scale": scaler.get_scale(),
                       "dev": evaluate_dev(model, dev, bucket_values, device, args)}
            print(f"[rl] checkpoint={step+1} dev={metrics['dev']}", flush=True)
            save_checkpoint(api, args, model, reference, optimizer, scheduler, scaler, rng,
                            manifest, manifest_sha, step + 1, metrics, repair_report)
            timer.mark()
    write_status(api, args.hf_repo, args.hf_run, f"DONE {args.steps} steps")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--sl-repo", type=Path, required=True)
    p.add_argument("--puzzles", type=Path, required=True)
    p.add_argument("--mate-dir", type=Path, default=ROOT / "data/positions")
    p.add_argument("--hf-repo", default="vedangfake/chess-slm-benchmark")
    p.add_argument("--hf-run", default="ccgavn-rl-root-v4-seed0")
    p.add_argument("--init-run", default="ccgavn-5m-seed0-v2")
    p.add_argument("--init-checkpoint", default="checkpoint-3896658")
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch-positions", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--kl-coef", type=float, default=0.2)
    p.add_argument("--value-coef", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=0.002)
    p.add_argument("--policy-temperature", type=float, default=0.2)
    p.add_argument("--searchless-commit", required=True)
    p.add_argument("--label-source-commit", required=True)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--upload-every", type=float, default=900)
    p.add_argument("--seed", type=int, default=20261025)
    args = p.parse_args()
    api = make_api(ROOT)
    try:
        run(args)
    except BaseException as exc:
        write_status(api, args.hf_repo, args.hf_run,
                     f"FAILED {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
