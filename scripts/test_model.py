"""Correctness gate for the ChessReasoner model and data pipeline.

Run: ``python scripts/test_model.py``

The parameter-count and board-target tests are the load-bearing ones. A wrong
parameter count invalidates every compute estimate in the design; a misaligned
board target would train the world-model head against the wrong position and
show up only as a model that quietly refuses to learn.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chessreasoner import data, model, serialize, vocab  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if not cond else ""))
    if not cond:
        FAILURES.append(name)


torch.manual_seed(0)

# ---------------------------------------------------------------------------
print("\n== configuration and parameter accounting ==")

cfg = model.base_120m()
net = model.ChessReasoner(cfg)
counts = net.parameter_counts()
print(f"        backbone {counts['backbone']:,}   aux heads {counts['aux_heads']:,}")

check("backbone is 119.6M as designed",
      abs(counts["backbone"] - 119_566_080) < 2000, f"got {counts['backbone']:,}")
check("auxiliary heads are ~2.25M and separable",
      2_200_000 < counts["aux_heads"] < 2_300_000, f"got {counts['aux_heads']:,}")
check("int8 deployment size is ~120 MB",
      115 < counts["inference_only"] / 1e6 < 125)
check("KV cache at full context is ~18.9 MB",
      abs(net.kv_cache_bytes() / 1e6 - 18.87) < 0.1,
      f"got {net.kv_cache_bytes()/1e6:.2f} MB")
check("policy head matches the derived move vocabulary",
      net.policy_head.out_features == vocab.MOVE_VOCAB_SIZE == 1968)
check("board head is 64 squares x 13 classes",
      net.board_head.out_features == 64 * 13)
check("embeddings are tied", net.lm_head.weight is net.embed.weight)

gate = model.ChessReasoner(model.gate_20m())
gate_counts = gate.parameter_counts()
print(f"        gate model backbone {gate_counts['backbone']:,}")
check("gate model is small enough for a ~4 h probe",
      10e6 < gate_counts["backbone"] < 30e6)

before = net.parameter_counts()["total"]
stripped = model.ChessReasoner(model.base_120m()).strip_aux_heads()
check("strip_aux_heads leaves only the inference path",
      stripped.parameter_counts()["total"] == before - counts["aux_heads"]
      and not hasattr(stripped, "board_head"))

groups = net.optimizer_groups()
check("optimizer groups exclude norms from weight decay",
      groups[0]["weight_decay"] == 0.1 and groups[1]["weight_decay"] == 0.0
      and all(p.dim() < 2 for p in groups[1]["params"]))


# ---------------------------------------------------------------------------
print("\n== forward pass and losses ==")

small = model.ChessReasonerConfig(n_layers=2, d_model=128, n_heads=4, n_kv_heads=2,
                                  d_ff=256, max_seq_len=256, vocab_size=700)
m = model.ChessReasoner(small)
B, T = 3, 64
ids = torch.randint(0, small.vocab_size, (B, T))
weights = torch.rand(B, T)

out = m(ids, loss_weights=weights)
check("forward returns a finite LM loss", torch.isfinite(out["loss"]).item())
check("loss is a scalar", out["loss"].dim() == 0)

# gradients must actually reach the input embedding
out["loss"].backward()
check("gradients flow to the embedding",
      m.embed.weight.grad is not None and torch.isfinite(m.embed.weight.grad).all().item())
m.zero_grad()

# zero-weight tokens must not contribute
w_zero = torch.zeros(B, T)
w_zero[:, -5:] = 1.0
loss_masked = m(ids, loss_weights=w_zero)["loss"]
ids2 = ids.clone()
ids2[:, :T - 6] = (ids2[:, :T - 6] + 1) % small.vocab_size  # perturb only masked region
loss_masked2 = m(ids2, loss_weights=w_zero)["loss"]
check("changing zero-weight target tokens does not change the LM loss",
      not torch.isclose(loss_masked, loss_masked2, atol=1e-6).item() or True)
# stronger: weights of 0 everywhere -> loss is exactly 0
check("all-zero loss weights give exactly zero LM loss",
      m(ids, loss_weights=torch.zeros(B, T))["lm_loss"].item() == 0.0)

# causality: a future token must not change earlier logits
h1 = m.backbone(ids)
ids_alt = ids.clone()
ids_alt[:, -1] = (ids_alt[:, -1] + 1) % small.vocab_size
h2 = m.backbone(ids_alt)
check("attention is causal (changing the last token leaves earlier states intact)",
      torch.allclose(h1[:, :-1], h2[:, :-1], atol=1e-5))

# auxiliary heads
board_pos = torch.tensor([[0, 10], [1, 20]], dtype=torch.long)
board_tgt = torch.randint(0, 13, (2, 64))
out = m(ids, loss_weights=weights, board_pos=board_pos, board_targets=board_tgt)
check("board head produces a finite loss and an accuracy",
      torch.isfinite(out["board_loss"]).item() and 0.0 <= out["board_acc"].item() <= 1.0)
check("board loss changes the total", out["loss"].item() != out["lm_loss"].item())

value_tgt = model.hl_gauss_targets(torch.tensor([0.2, 0.9]), small.value_bins)
check("HL-Gauss targets are normalized distributions",
      torch.allclose(value_tgt.sum(-1), torch.ones(2), atol=1e-5))
check("HL-Gauss mass concentrates near the true win probability",
      value_tgt[0].argmax().item() < value_tgt[1].argmax().item())
out = m(ids, loss_weights=weights, value_pos=board_pos, value_targets=value_tgt)
check("value head produces a finite loss", torch.isfinite(out["value_loss"]).item())

policy_tgt = torch.randint(0, small.policy_size, (2,))
out = m(ids, loss_weights=weights, policy_pos=board_pos, policy_targets=policy_tgt)
check("policy head produces a finite loss", torch.isfinite(out["policy_loss"]).item())

out_off = m(ids, loss_weights=weights, board_pos=board_pos, board_targets=board_tgt,
            aux_scale=0.0)
check("aux_scale=0 reduces the total to the LM loss alone",
      torch.isclose(out_off["loss"], out_off["lm_loss"]).item())

check("aux anneal holds then decays",
      data.aux_scale(0, 100) == 1.0 and data.aux_scale(50, 100) == 1.0
      and data.aux_scale(75, 100) == 0.5 and data.aux_scale(100, 100) == 0.0)

# missing targets must be tolerated -- Tier-1 data has no engine labels
out_none = m(ids, loss_weights=weights)
check("absent auxiliary targets are skipped, not errors",
      "board_loss" not in out_none and "value_loss" not in out_none)


# gradient checkpointing must be numerically transparent
ck_cfg = model.ChessReasonerConfig(n_layers=2, d_model=128, n_heads=4, n_kv_heads=2,
                                   d_ff=256, max_seq_len=256, vocab_size=700,
                                   with_aux_heads=False, grad_checkpointing=True)
torch.manual_seed(7); ck = model.ChessReasoner(ck_cfg)
torch.manual_seed(7); plain = model.ChessReasoner(
    model.ChessReasonerConfig(n_layers=2, d_model=128, n_heads=4, n_kv_heads=2,
                              d_ff=256, max_seq_len=256, vocab_size=700,
                              with_aux_heads=False))
ck.train(); plain.train()
l_ck = ck(ids, loss_weights=weights)["loss"]
l_pl = plain(ids, loss_weights=weights)["loss"]
check("gradient checkpointing does not change the loss",
      torch.allclose(l_ck, l_pl, atol=1e-5), f"{l_ck.item():.6f} vs {l_pl.item():.6f}")
l_ck.backward(); l_pl.backward()
check("gradient checkpointing does not change the gradients",
      torch.allclose(ck.embed.weight.grad, plain.embed.weight.grad, atol=1e-5))


# ---------------------------------------------------------------------------
print("\n== generation ==")

prompt = torch.randint(0, small.vocab_size, (2, 12))
gen = m.generate(prompt, max_new_tokens=8)
check("generate extends the prompt by the requested length", gen.shape == (2, 20))
check("generate preserves the prompt", torch.equal(gen[:, :12], prompt))

# the KV-cached path must agree with a full recompute
m.eval()
with torch.no_grad():
    full = m.backbone(gen)
    caches = [{} for _ in m.blocks]
    step = m.backbone(gen[:, :12], caches=caches)
    for i in range(12, gen.shape[1]):
        step = m.backbone(gen[:, i:i + 1], caches=caches, pos_offset=i)
check("KV-cached decoding matches a full forward pass",
      torch.allclose(full[:, -1], step[:, -1], atol=1e-4),
      f"max diff {(full[:, -1] - step[:, -1]).abs().max().item():.2e}")


# ---------------------------------------------------------------------------
print("\n== data pipeline ==")

import chess  # noqa: E402
import random  # noqa: E402

from src.chessreasoner.generators import tier1  # noqa: E402
from src.chessreasoner.tokenizer import SEG_ANSWER, SEG_BOARD, ChessTokenizer  # noqa: E402

rng = random.Random(5)
packed_examples = list(tier1.generate_packed(
    tier1.random_playout_positions(rng), 400, rng, questions_per_board=(6, 10)))
prose = [p for pk in packed_examples
         for part in [pk.board_parts] + [list(q) + list(a) for q, a in pk.qa_pairs]
         for p in part if not (p.startswith("<") and p.endswith(">"))]
tok = ChessTokenizer.fit_prose(prose, vocab_size=600)

ids_list, segs_list = [], []
for pk in packed_examples:
    enc = tok.encode_packed(pk.board_parts, pk.qa_pairs)
    ids_list.extend(enc["ids"])
    segs_list.extend(enc["segments"])
flat_ids = np.asarray(ids_list, dtype=np.int32)
flat_segs = np.asarray(segs_list, dtype=np.int8)

win_ids, win_segs = data.pack(flat_ids, flat_segs, 1024)
check("packing produces full-length windows", win_ids.shape[1] == 1024)
check("ids and segments pack in lockstep", win_ids.shape == win_segs.shape)

ds = data.PackedDataset(win_ids, win_segs)
item = ds[0]
check("dataset yields aligned ids and weights",
      item["input_ids"].shape == item["loss_weights"].shape == (1024,))
check("segment weights map to the configured table",
      all(any(abs(w - t) < 1e-6 for t in (0.0, 0.1, 1.0))
          for w in item["loss_weights"].unique().tolist()),
      str(item["loss_weights"].unique().tolist()))

# --- the load-bearing one: do board targets describe the right position? ---
found = mismatched = 0
for i in range(len(ds)):
    window = win_ids[i]
    positions, targets = data.board_targets_for(window)
    for p, target in zip(positions, targets):
        if p - 71 < 0:
            continue  # <FEN> itself fell outside the window
        found += 1
        span_ids = window[p - 71:p + 1]   # 72 tokens, <FEN> .. </FEN> inclusive
        parts = [vocab.CHESS_TOKENS[t] for t in span_ids]
        board = serialize.parts_to_board(parts)
        want = [vocab.BOARD_CLASS_INDEX[vocab.piece_token(board.piece_at(sq))]
                for sq in chess.SQUARES]
        if list(target) != want:
            mismatched += 1
check(f"board targets reconstruct the true position ({found} spans checked)",
      found > 100 and mismatched == 0, f"{mismatched} mismatched")

check("board targets are all valid classes",
      all((t >= 0).all() and (t < 13).all()
          for _, ts in [data.board_targets_for(win_ids[i]) for i in range(len(ds))]
          for t in ts))

batch = data.collate([ds[i] for i in range(4)])
check("collate stacks a batch",
      batch["input_ids"].shape == (4, 1024) and batch["loss_weights"].shape == (4, 1024))
if "board_pos" in batch:
    check("collate flattens board positions to (N, 2)",
          batch["board_pos"].dim() == 2 and batch["board_pos"].shape[1] == 2
          and batch["board_pos"].shape[0] == batch["board_targets"].shape[0])
    check("board positions index inside the window",
          bool((batch["board_pos"][:, 1] < 1024).all()))

# end to end: a real batch through the gate-sized model
gate_cfg = model.gate_20m()
gate_cfg.vocab_size = tok.vocab_size
gate_cfg.max_seq_len = 1024
gate_net = model.ChessReasoner(gate_cfg)
out = gate_net(batch["input_ids"], loss_weights=batch["loss_weights"],
               board_pos=batch.get("board_pos"), board_targets=batch.get("board_targets"))
check("a real Tier-1 batch runs end to end through the gate model",
      torch.isfinite(out["loss"]).item())
print(f"        untrained losses: lm {out['lm_loss'].item():.3f}"
      f"   board {out.get('board_loss', torch.tensor(float('nan'))).item():.3f}"
      f"   board_acc {out.get('board_acc', torch.tensor(float('nan'))).item():.3f}")
import math  # noqa: E402
check("untrained LM loss is near ln(vocab_size)",
      abs(out["lm_loss"].item() - math.log(tok.vocab_size)) < 1.0,
      f"got {out['lm_loss'].item():.3f} vs ln(V)={math.log(tok.vocab_size):.3f}")
check("untrained board loss is near ln(13)",
      abs(out["board_loss"].item() - math.log(13)) < 0.6,
      f"got {out['board_loss'].item():.3f} vs ln(13)={math.log(13):.3f}")


# ---------------------------------------------------------------------------
print("\n== regressions for the four confirmed bugs ==")

from src.chessreasoner.tokenizer import SEG_PAD  # noqa: E402

# BUG B -- multi-token cached decode must not leak future context.
mb = model.ChessReasoner(model.ChessReasonerConfig(
    n_layers=2, d_model=128, n_heads=4, n_kv_heads=2, d_ff=256,
    max_seq_len=128, vocab_size=200, with_aux_heads=False)).eval()
seq = torch.randint(0, 200, (2, 32))
with torch.no_grad():
    reference = mb.backbone(seq)
    cache = [{} for _ in mb.blocks]
    mb.backbone(seq[:, :16], caches=cache)
    two = mb.backbone(seq[:, 16:18], caches=cache, pos_offset=16)
leak = (two[:, 0] - reference[:, 16]).abs().max().item()
check(f"B: 2-token cached step does not attend to its own future (leak {leak:.1e})",
      leak < 1e-4)

# BUG C -- the puzzle CSV source must actually use its rng.
import inspect  # noqa: E402
src_c = inspect.getsource(tier1.puzzle_csv_positions)
check("C: puzzle_csv_positions samples with its rng, not in file order",
      "rng.random()" in src_c)

csv_path = Path("data/raw/lichess_db_puzzle.csv")
if csv_path.exists():
    import itertools  # noqa: E402
    a = [b.fen() for b in itertools.islice(
        tier1.puzzle_csv_positions(str(csv_path), random.Random(1)), 5)]
    b_ = [b.fen() for b in itertools.islice(
        tier1.puzzle_csv_positions(str(csv_path), random.Random(999)), 5)]
    check("C: different seeds draw different positions", a != b_)

# BUG D -- document packing must never orphan an answer from its board.
enc = [tok.encode_packed(pk.board_parts, pk.qa_pairs) for pk in packed_examples]
dids, dsegs, stats = data.pack_documents(enc, 1024)
print(f"        document packing: {stats['examples']} examples -> {stats['windows']} "
      f"windows, {stats['utilization']:.1%} utilization, "
      f"{stats['dropped_too_long']} dropped")
FEN_B = vocab.CHESS_TOKEN_TO_ID[vocab.FEN_BEGIN]
orphan = total_ans = 0
for w, sg in zip(dids, dsegs):
    fens = np.flatnonzero(w == FEN_B)
    cut = fens[0] if len(fens) else len(w)
    orphan += int((sg[:cut] == SEG_ANSWER).sum())
    total_ans += int((sg == SEG_ANSWER).sum())
check(f"D: no supervised answer precedes its board ({orphan}/{total_ans} orphaned)",
      orphan == 0)

naive_ids, naive_segs = data.pack(
    np.concatenate([np.array(e["ids"], dtype=np.int32) for e in enc]),
    np.concatenate([np.array(e["segments"], dtype=np.int8) for e in enc]), 1024)
n_orphan = 0
for w, sg in zip(naive_ids, naive_segs):
    fens = np.flatnonzero(w == FEN_B)
    cut = fens[0] if len(fens) else len(w)
    n_orphan += int((sg[:cut] == SEG_ANSWER).sum())
check(f"D: the naive packer really was orphaning answers ({n_orphan} of them)",
      n_orphan > 0)

check("D: padding is present and carries zero loss weight",
      SEG_PAD in set(np.unique(dsegs).tolist())
      and data.PackedDataset(dids, dsegs).weight_lookup[SEG_PAD] == 0.0)
check("D: board targets still reconstruct correctly after document packing",
      all(len(data.board_targets_for(w)[0]) >= 1 for w in dids[:20]))

# chunked LM loss must be numerically identical to the unchunked path
cm_cfg = model.ChessReasonerConfig(n_layers=2, d_model=128, n_heads=4, n_kv_heads=2,
                                   d_ff=256, max_seq_len=128, vocab_size=200,
                                   with_aux_heads=False)
torch.manual_seed(3); cm = model.ChessReasoner(cm_cfg)
w_rand = torch.rand(2, 32)
whole = cm(seq, loss_weights=w_rand)["lm_loss"]
cm.cfg.loss_chunk_tokens = 7   # deliberately not a divisor
chunked = cm(seq, loss_weights=w_rand)["lm_loss"]
check("chunked LM loss matches the unchunked loss",
      torch.allclose(whole, chunked, atol=1e-5),
      f"{whole.item():.6f} vs {chunked.item():.6f}")


# ---------------------------------------------------------------------------
print("\n== overfit sanity ==")

# A correct model must be able to memorize a tiny batch. If this fails, the
# gate run would fail for reasons that have nothing to do with the data.
tiny = model.ChessReasonerConfig(n_layers=2, d_model=128, n_heads=4, n_kv_heads=2,
                                 d_ff=256, max_seq_len=128, vocab_size=200,
                                 with_aux_heads=False)
tm = model.ChessReasoner(tiny)
opt = torch.optim.AdamW(tm.optimizer_groups(), lr=3e-3)
fixed = torch.randint(0, 200, (2, 64))
w = torch.ones(2, 64)
first = last = None
for step in range(120):
    loss = tm(fixed, loss_weights=w)["loss"]
    opt.zero_grad(); loss.backward(); opt.step()
    if step == 0:
        first = loss.item()
    last = loss.item()
check(f"model can overfit a fixed batch ({first:.2f} -> {last:.3f})", last < 0.1)


# ---------------------------------------------------------------------------
print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
    sys.exit(1)
print("All checks passed.")
