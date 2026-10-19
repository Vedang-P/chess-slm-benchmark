"""Shared-weight, exact-transition refutation network. No engine at inference.

The root scorer keeps the original checkpoint contract. All future values are
side-to-move values, converted to the root player's perspective before fusion.
Two opponent replies are considered; each continuation thereafter is greedy.
This is bounded internal planning, not exhaustive minimax or a one-pass model.
"""
from __future__ import annotations

import json
from pathlib import Path

import chess
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.eval_gavn import tokenize_fen
from scripts.train_ccgavn import CCGAVN, candidate_relation_types
from scripts.train_gavn import action_tables

ARCHITECTURE = "cc-gavn-refutation-v1"


class RefutationNetwork(nn.Module):
    def __init__(self, base, horizon=4, width=2):
        super().__init__()
        if horizon < 1 or width not in (1, 2):
            raise ValueError("horizon >= 1; refutation width must be 1 or 2")
        self.base, self.horizon, self.width = base, horizon, width
        d = base.dim
        self.null = nn.Parameter(torch.zeros(d))
        self.policy_src = nn.Linear(d, 32)
        self.policy_dst = nn.Linear(d, 32)
        self.policy_bias = nn.Linear(d, 1)
        self.policy_promo = nn.Linear(d, 5)
        self.value = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 128))
        # 0=no known mate; 1..16 winning; 17..32 losing, distances clipped.
        self.mate = nn.Linear(d, 33)
        self.fusion = nn.Sequential(nn.Linear(2*d + 4, 64), nn.GELU(),
                                    nn.Linear(64, 128))
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)
        self.register_buffer("buckets", (torch.arange(128) + .5) / 128)
        self.parameter_count = sum(p.numel() for p in self.parameters())
        if self.parameter_count >= 5_000_000:
            raise ValueError(f"strict 5M cap exceeded: {self.parameter_count}")

    def state(self, tokens):
        _, h = self.base(tokens, null_token=self.null, return_hidden=True)
        squares = h[:, :64]
        src = self.policy_src(squares)[:, self.base.action_src]
        dst = self.policy_dst(squares)[:, self.base.action_dst]
        policy = (src * dst).sum(-1) / 32**.5
        policy = policy + self.policy_bias(squares)[:, self.base.action_src].squeeze(-1)
        policy = policy + self.policy_promo(squares)[:, self.base.action_dst].gather(
            2, self.base.action_promo[None, :, None].expand(len(h), -1, 1)).squeeze(-1)
        return self.value(h[:, 64]), policy, self.mate(h[:, 64]), h[:, 64]

    def forward(self, root, actions, future=None, active=None, terminal=None):
        logits, hidden = self.base(root, actions, return_hidden=True)
        result = {"base": logits, "logits": logits}
        if future is None:
            return result
        n, width, depth, _ = future.shape
        values, policies, mates, h = self.state(future.reshape(-1, 77))
        return self.fuse(logits, hidden[:, 64], values, policies, mates, h, active, terminal)

    def fuse(self, logits, root_h, values, policies, mates, h, active, terminal):
        n, width, depth = active.shape
        result = {"base": logits}
        h = h.reshape(n, width, depth, -1)
        q = (values.float().softmax(-1) @ self.buckets).reshape(n, width, depth)
        # After root move (index 0), opponent is to move. Alternate thereafter.
        sign = torch.arange(depth, device=q.device) % 2 == 0
        root_q = torch.where(sign[None, None], 1-q, q)
        root_q = torch.where(torch.isfinite(terminal), terminal, root_q)
        last = active.long().sum(-1).clamp_min(1) - 1
        leaf_q = root_q.gather(2, last[..., None]).squeeze(-1)
        leaf_h = h.gather(2, last[..., None, None].expand(-1, -1, 1, h.size(-1))).squeeze(2)
        worst = leaf_q.argmin(1)
        batch = torch.arange(n, device=q.device)
        q0 = logits.float().softmax(-1) @ self.buckets
        qmin, qmax = leaf_q.min(1).values, leaf_q.max(1).values
        features = torch.cat((root_h, leaf_h[batch, worst],
                              torch.stack((q0, qmin, qmax-qmin,
                                           last.float().mean(1)/depth), -1)), -1)
        result.update(logits=logits + self.fusion(features), state_values=values,
                      policy=policies, mate=mates, leaf_q=leaf_q)
        return result


def load_network(checkpoint, sl_repo, device="cpu"):
    checkpoint = Path(checkpoint)
    cfg = json.loads((checkpoint / "config.json").read_text())
    src, dst, promo, utils = action_tables(Path(sl_repo))
    base = CCGAVN(torch, cfg["dim"], cfg["layers"], cfg["heads"],
                  src, dst, promo, candidate_relation_types(),
                  model_version=cfg.get("model_version", "v1"))
    network = RefutationNetwork(base, cfg.get("horizon", 4), cfg.get("width", 2))
    if not np.allclose(utils.get_uniform_buckets_edges_values(128)[1],
                       network.buckets.numpy(), rtol=0, atol=1e-7):
        raise ValueError("upstream return-bin convention changed")
    state = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=False)
    if cfg.get("architecture") == ARCHITECTURE:
        network.load_state_dict(state["model"], strict=True)
    elif cfg.get("architecture") == "cc-gavn-v1":
        network.base.load_state_dict(state["model"], strict=True)
    else:
        raise ValueError("expected a CC-GAVN or refutation checkpoint")
    return network.to(device), cfg, utils


def outcome(board, root_color):
    # Includes terminal game rules, but does not declare a claimable draw
    # automatically. The official evaluator applies its repetition convention.
    end = board.outcome(claim_draw=False)
    return np.nan if end is None else .5 if end.winner is None else float(end.winner == root_color)


@torch.no_grad()
def rollout(network, boards, moves, utils, return_cache=False):
    """Batched learned continuations. Exact boards retain repetition history."""
    device = next(network.parameters()).device
    n, w, depth = len(boards), network.width, network.horizon
    future = np.empty((n, w, depth, 77), dtype=np.int64)
    active = np.zeros((n, w, depth), dtype=bool)
    terminal = np.full((n, w, depth), np.nan, dtype=np.float32)
    states = []
    for board, move in zip(boards, moves):
        child = board.copy(stack=True)
        child.push(move)
        states.extend(child.copy(stack=True) for _ in range(w))
    caches = []
    for t in range(depth):
        eligible, legal = [], []
        for j, board in enumerate(states):
            i, k = divmod(j, w)
            future[i, k, t] = tokenize_fen(board.fen())
            value = outcome(board, boards[i].turn)
            # Store the first terminal state, then only padded copies.
            active[i, k, t] = t == 0 or not np.isfinite(terminal[i, k, t-1])
            terminal[i, k, t] = value
            if not np.isfinite(value) and t < depth-1:
                eligible.append(j)
                legal.append(sorted(board.legal_moves, key=lambda m: m.uci()))
        if return_cache:
            outputs = network.state(torch.tensor(np.stack([tokenize_fen(s.fen()) for s in states]), device=device))
            caches.append(outputs)
            logits = outputs[1][eligible]
        elif eligible:
            tokens = torch.tensor(np.stack([tokenize_fen(states[j].fen()) for j in eligible]),
                                  device=device)
            _, logits, _, _ = network.state(tokens)
        if eligible:
            for row, (j, choices) in enumerate(zip(eligible, legal)):
                ids = [utils.MOVE_TO_ACTION[m.uci()] for m in choices]
                # At the first opponent reply use two distinct hypotheses.
                order = torch.argsort(logits[row, ids], descending=True, stable=True).tolist()
                rank = min(j % w, len(order)-1) if t == 0 else 0
                states[j].push(choices[order[rank]])
    paths = tuple(torch.as_tensor(x, device=device) for x in (future, active, terminal))
    if return_cache:
        cache = tuple(torch.stack([c[i] for c in caches], 1).flatten(0, 1) for i in range(4))
        return paths, cache
    return paths


@torch.no_grad()
def planned_logits(network, boards, moves, utils):
    device = next(network.parameters()).device
    tokens = torch.tensor(np.stack([tokenize_fen(b.fen()) for b in boards]), device=device)
    actions = torch.tensor([utils.MOVE_TO_ACTION[m.uci()] for m in moves], device=device)
    logits, h = network.base(tokens, actions, return_hidden=True)
    paths, cache = rollout(network, boards, moves, utils, return_cache=True)
    return network.fuse(logits, h[:, 64], *cache, paths[1], paths[2])["logits"]


@torch.inference_mode()
def score_moves(network, board, moves, utils, chunk=16, mode="planner"):
    if mode not in ("base", "planner"):
        raise ValueError(f"unknown inference mode: {mode}")
    device = next(network.parameters()).device
    outputs = []
    for start in range(0, len(moves), chunk):
        subset = moves[start:start+chunk]
        tokens = torch.tensor(np.repeat(tokenize_fen(board.fen())[None], len(subset), 0), device=device)
        actions = torch.tensor([utils.MOVE_TO_ACTION[m.uci()] for m in subset], device=device)
        logits = (planned_logits(network, [board]*len(subset), subset, utils)
                  if mode == "planner" else network(tokens, actions)["logits"])
        outputs.extend((logits.float().softmax(-1) @ network.buckets).cpu().tolist())
    return np.asarray(outputs)


def value_target(q, buckets):
    """Two-bin interpolation preserves expectation without sharp bucket jumps."""
    x = (q.clamp(0, 1)*128 - .5).clamp(0, 127)
    low = x.long(); high = (low+1).clamp_max(127)
    out = torch.zeros((*q.shape, 128), device=q.device)
    out.scatter_add_(-1, low[..., None], (1-(x-low))[..., None])
    out.scatter_add_(-1, high[..., None], (x-low)[..., None])
    return out


def objective(output, batch, buckets, anchor, mode, temperature=.05):
    logits = output["logits"].float()
    targets = value_target(batch["q"], buckets)
    loss = -(targets*F.log_softmax(logits, -1)).sum(-1).mean()
    q = (logits.softmax(-1) @ buckets).clamp(1e-5, 1-1e-5)
    rank = logits.new_zeros(())
    for group in batch["groups"]:
        # Centipawn/mate utility retains ordering inside saturated WDL bins.
        truth = F.softmax(batch["utility"][group]/temperature, 0)
        rank = rank - (truth * F.log_softmax(torch.logit(q[group])/1., 0)).sum()
    loss = loss + rank/len(batch["groups"])
    loss = loss + .2*F.kl_div(F.log_softmax(output["base"].float(), -1),
                              anchor.float().softmax(-1), reduction="batchmean")
    if mode == "distill":
        loss = loss + F.kl_div(F.log_softmax(logits, -1), batch["teacher"], reduction="batchmean")
    if mode == "planner":
        mask = batch["active"].flatten()
        state_q = batch["state_q"].flatten()[mask]
        loss = loss - .5*(value_target(state_q, buckets)*F.log_softmax(
            output["state_values"][mask].float(), -1)).sum(-1).mean()
        legal = batch["legal"].reshape(-1, batch["legal"].size(-1))
        policy_mask = batch["policy_target"].flatten() >= 0
        if policy_mask.any():
            policy = output["policy"][policy_mask].float().masked_fill(~legal[policy_mask], -1e9)
            loss = loss + F.cross_entropy(policy, batch["policy_target"].flatten()[policy_mask])
        # Unknown/non-proven mates are not treated as proof of no mate.
        mate_mask = mask & (batch["mate_target"].flatten() > 0)
        if mate_mask.any():
            loss = loss + .1*F.cross_entropy(output["mate"][mate_mask], batch["mate_target"].flatten()[mate_mask])
    return loss
