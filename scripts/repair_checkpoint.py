"""Convert the trained v1 CC-GAVN function to the effective v2 architecture.

The v1 query-side dynamic attention term is constant across each softmax row.
Only its key-side half is retained.  Field gates start at one, so they can
learn field identity without changing the initial function.  This script
refuses to produce a checkpoint unless actual legal-move logits agree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import chess
import numpy as np
import torch

from scripts.eval_gavn import tokenize_fen
from scripts.train_ccgavn import CCGAVN, candidate_relation_types
from scripts.train_gavn import action_tables


def convert_weights(old: dict[str, torch.Tensor], layers: int, heads: int,
                    dim: int) -> dict[str, torch.Tensor]:
    """An exact functional transplant, excluding the softmax-null parameters."""
    result = {name: value.clone() for name, value in old.items()}
    result["global_field"] = torch.ones(13, dim, dtype=old["global_pos"].dtype)
    for index in range(layers):
        for suffix in ("weight", "bias"):
            key = f"blocks.{index}.dynamic.{suffix}"
            value = old[key]
            if value.shape[0] != heads * 2 * 65:
                raise ValueError(f"unexpected v1 dynamic shape: {key} {value.shape}")
            result[key] = value.reshape(heads, 2, 65, *value.shape[1:])[:, 1].reshape(
                heads * 65, *value.shape[1:]).contiguous()
    return result


def make_model(cfg: dict, version: str, sl_repo: Path):
    src, dst, promo, utils = action_tables(sl_repo)
    model = CCGAVN(torch, int(cfg["dim"]), int(cfg["layers"]), int(cfg["heads"]),
                   src, dst, promo, candidate_relation_types(), model_version=version)
    return model, utils


def assert_equivalent(old_model, new_model, utils, tolerance: float = 1e-4) -> dict:
    fens = (
        chess.STARTING_FEN,
        "r3k2r/pppq1ppp/2n1bn2/3pp3/3PP3/2N1BN2/PPPQ1PPP/R3K2R w KQkq - 4 10",
        "4k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
        "8/8/8/8/3pP3/8/4K3/7k b - e3 0 20",
    )
    all_tokens, all_actions = [], []
    for fen in fens:
        board = chess.Board(fen)
        for move in board.legal_moves:
            all_tokens.append(tokenize_fen(board.fen()))
            all_actions.append(utils.MOVE_TO_ACTION[move.uci()])
    tokens = torch.as_tensor(np.stack(all_tokens), dtype=torch.long)
    actions = torch.as_tensor(all_actions, dtype=torch.long)
    old_model.eval(); new_model.eval()
    with torch.inference_mode():
        differences = []
        for start in range(0, len(actions), 32):
            before = old_model(tokens[start:start + 32], actions[start:start + 32])
            after = new_model(tokens[start:start + 32], actions[start:start + 32])
            differences.append(float((before - after).abs().max()))
    maximum = max(differences)
    if not np.isfinite(maximum) or maximum > tolerance:
        raise ValueError(f"functional transplant failed: max logit delta {maximum} > {tolerance}")
    return {"legal_candidates": len(actions), "max_logit_delta": maximum,
            "tolerance": tolerance}


def repair(source: Path, output: Path, sl_repo: Path) -> dict:
    cfg = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if cfg.get("architecture") != "cc-gavn-v1" or cfg.get("model_version", "v1") != "v1":
        raise ValueError("source must be the trained v1 CC-GAVN checkpoint")
    old_state = torch.load(source / "state.pt", map_location="cpu", weights_only=False)
    old_model, utils = make_model(cfg, "v1", sl_repo)
    new_model, _ = make_model(cfg, "v2", sl_repo)
    old_model.load_state_dict(old_state["model"], strict=True)
    new_model.load_state_dict(convert_weights(old_state["model"], int(cfg["layers"]),
                                              int(cfg["heads"]), int(cfg["dim"])), strict=True)
    old_params = sum(p.numel() for p in old_model.parameters())
    new_params = sum(p.numel() for p in new_model.parameters())
    if old_params != int(cfg["parameter_count"]) or new_params >= 5_000_000:
        raise ValueError(f"parameter identity/budget failure: {old_params} -> {new_params}")
    parity = assert_equivalent(old_model, new_model, utils)
    digest = hashlib.sha256((source / "state.pt").read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": new_model.state_dict()}, output / "weights.pt")
    report = {"source_checkpoint": str(source), "source_state_sha256": digest,
              "source_commit": cfg.get("source_commit"),
              "source_model_version": "v1", "model_version": "v2",
              "old_parameters": old_params, "new_parameters": new_params,
              "removed_null_parameters": old_params - new_params + 13 * int(cfg["dim"]),
              **parity}
    (output / "repair-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sl-repo", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(repair(args.source, args.output, args.sl_repo), indent=2))


if __name__ == "__main__":
    main()
