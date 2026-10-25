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


def source_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                   text=True).strip()


def load_labels(api, repo: str, manifest: dict, excluded: np.ndarray):
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
            if set(moves) != set(record["moves"]) or len(moves) != len(record["moves"]):
                raise ValueError(f"{account}: verifier labels do not cover all legal moves")
            if len(record["actions"]) != len(moves) or len(record["rewards"]) != len(moves):
                raise ValueError(f"{account}: action/reward length mismatch")
            tokens = tokenize_fen(fen)
            position_hash = int(position_hashes(tokens[None])[0])
            if position_hash != int(record["position_hash"]) or position_hash in excluded:
                raise ValueError(f"{account}: label identity or frozen exclusion failure")
            if position_hash in seen:
                raise ValueError("duplicate position across frozen label sources")
            seen.add(position_hash)
            rewards = np.asarray(record["rewards"], dtype=np.float32)
            if not np.all(np.isfinite(rewards)) or np.any(rewards < 0) or np.any(rewards > 1):
                raise ValueError("invalid verifier rewards")
            records.append((tokens, np.asarray(record["actions"], dtype=np.int64), rewards,
                            position_hash))
    if len(records) < 1000:
        raise ValueError(f"only {len(records)} verified roots; require at least 1000")
    train = [row for row in records if row[3] % 100 != 0]
    dev = [row for row in records if row[3] % 100 == 0]
    if len(dev) < 10:
        raise ValueError(f"only {len(dev)} development roots")
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


def objective(values, old_logs, ref_logs, rewards, offsets, samples,
              advantages, args):
    logs = policy_logprobs(values, offsets, args.policy_temperature)
    policy_losses, kl_terms, value_terms, entropy_terms = [], [], [], []
    for group, (a, b) in enumerate(zip(offsets[:-1], offsets[1:])):
        indices = samples[group]
        ratio = torch.exp(logs[group][indices] - old_logs[group][indices])
        adv = advantages[group]
        clipped = ratio.clamp(1 - args.clip, 1 + args.clip)
        policy_losses.append(-torch.minimum(ratio * adv, clipped * adv).mean())
        probs = logs[group].exp()
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


def save_checkpoint(api, args, model, reference, optimizer, scaler, rng,
                    manifest, manifest_sha, step, metrics, repair_report):
    folder = args.outdir / f"checkpoint-{step}"
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "reference": reference.state_dict(),
                "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "step": step, "numpy_rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
               folder / "state.pt")
    config = {"architecture": "cc-gavn-v1", "model_version": "v2",
              "recipe_version": "rl-root-v1", "dim": 208, "layers": 8, "heads": 8,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "source_commit": source_commit(), "source_checkpoint": args.init_run + "/" + args.init_checkpoint,
              "repair": repair_report, "manifest_sha256": manifest_sha,
              "exclusion_digest": manifest["exclusion_digest"],
              "canonical_decision_head": "distribution_expectation",
              "algorithm": "group-relative-clipped-root-policy",
              "steps": args.steps, "batch_positions": args.batch_positions,
              "group_size": args.group_size, "ppo_epochs": args.ppo_epochs,
              "lr": args.lr, "clip": args.clip,
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
    if args.policy_temperature <= 0 or not 0 < args.clip < 1:
        raise ValueError("policy temperature and PPO clip are invalid")
    if args.steps < 1 or args.batch_positions < 1 or args.group_size < 2:
        raise ValueError("invalid training budget")
    api = make_api(ROOT)
    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest_bytes = args.manifest.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    if manifest.get("source_commit") != source_commit():
        raise ValueError("frozen verifier source commit differs from learner code")
    excluded, exclusion_digest = load_exclusions(args.mate_dir, args.puzzles)
    if manifest["exclusion_digest"] != exclusion_digest:
        raise ValueError("frozen-label and training exclusions differ")
    train, dev = load_labels(api, args.hf_repo, manifest, excluded)
    print(f"[rl] verified {len(train)} train and {len(dev)} dev roots", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("RL learner requires a Kaggle T4 GPU")
    from searchless_chess.src import utils
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
        if (cfg["recipe_version"] != "rl-root-v1" or cfg["manifest_sha256"] != manifest_sha
                or cfg["source_commit"] != source_commit() or cfg["steps"] != args.steps):
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
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    if resume is not None:
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
    else:
        save_checkpoint(api, args, model, reference, optimizer, scaler, rng,
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
            old_values = values_for(model, tokens, actions, bucket_values, device)
            ref_values = values_for(reference, tokens, actions, bucket_values, device)
            old_logs = policy_logprobs(old_values, offsets, args.policy_temperature)
            ref_logs = policy_logprobs(ref_values, offsets, args.policy_temperature)
            samples, advantages = [], []
            for group, (a, b) in enumerate(zip(offsets[:-1], offsets[1:])):
                indices_g = torch.multinomial(old_logs[group].exp(), args.group_size,
                                               replacement=True)
                sampled_rewards = rewards[a:b][indices_g]
                adv = (sampled_rewards - sampled_rewards.mean()) / \
                      (sampled_rewards.std(unbiased=False) + 1e-6)
                samples.append(indices_g)
                advantages.append(adv)
        for epoch in range(args.ppo_epochs):
            optimizer.zero_grad(set_to_none=True)
            values = values_for(model, tokens, actions, bucket_values, device)
            loss, parts = objective(values, old_logs, ref_logs, rewards, offsets,
                                    samples, advantages, args)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if step == 0 and epoch == 0:
                missing = [name for name, parameter in model.named_parameters()
                           if parameter.grad is None]
                if missing:
                    raise RuntimeError(f"trainable parameters outside RL graph: {missing}")
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            scaler.step(optimizer); scaler.update()
        if (step + 1) % 20 == 0:
            print(f"[rl] step={step+1}/{args.steps} loss={loss.item():.4f} "
                  f"pg={parts[0].item():.4f} kl={parts[1].item():.4f} "
                  f"value={parts[2].item():.4f} grad={grad_norm:.3f}", flush=True)
        if (step + 1) % args.ckpt_every == 0 or step + 1 == args.steps or timer.due():
            metrics = {"step": step + 1, "loss": float(loss.detach()),
                       "pg": float(parts[0].detach()), "kl": float(parts[1].detach()),
                       "value": float(parts[2].detach()), "grad_norm": grad_norm,
                       "dev": evaluate_dev(model, dev, bucket_values, device, args)}
            print(f"[rl] checkpoint={step+1} dev={metrics['dev']}", flush=True)
            save_checkpoint(api, args, model, reference, optimizer, scaler, rng,
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
    p.add_argument("--hf-run", default="ccgavn-rl-root-v1-seed0")
    p.add_argument("--init-run", default="ccgavn-5m-seed0-v2")
    p.add_argument("--init-checkpoint", default="checkpoint-3896658")
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch-positions", type=int, default=8)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--ppo-epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--kl-coef", type=float, default=0.04)
    p.add_argument("--value-coef", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=0.002)
    p.add_argument("--policy-temperature", type=float, default=0.05)
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
