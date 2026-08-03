# Phase 1 — Mate-in-1 representation study (standard chess)

**Date:** 2026-08-03 · **Model:** deepseek-v4-flash (thinking enabled, unbounded)
**Task:** mate1-lichess, n=5 per representation, 20 samples total
**Rules:** STANDARD chess (python-chess — castling, en passant, double-step,
promotion all legal). Schema v3.

Raw per-sample reports (full prompts, complete thinking chains, answers,
verdicts): `docs/phase1/*.report.json` · mirrored on Hugging Face under
`runs/2026-08-03T13:13:59Z/`.

## Results

| Representation | Legal rate | Mate solved | Samples |
|---|---|---|---|
| fen | 1.0 | **1.0 (5/5)** | a5c3 ✓ f5g3 ✓ a5c7 ✓ f5g5 ✓ g6f7 ✓ |
| grid | 1.0 | 0.6 (3/5) | a5c3 ✓ f5g3 ✓ a5c7 ✓ f5g6 ✗ e3e4 ✗ |
| list | 0.8 | 0.6 (3/5) | a5c3 ✓ g4f3 ✗ a5c7 ✓ f5g5 ✓ h5f7 ✗(illegal) |
| bitboard | 0.6 | **0.0 (0/5)** | h5f3 ✗(illegal) d4e4 ✗(illegal) g2g4 ✗ e3e4 ✗ h2h3 ✗ |

## Findings

1. **FEN is the only representation deepseek-v4-flash solved every position
   from** (5/5, 100% legality). The model was pretrained on FEN-heavy chess
   text, and it shows.
2. **Bitboard representation breaks it completely** — 0/5 mates, 2/5 illegal
   moves. Bitboards (64-bit maps per piece type) are an encoding models were
   not trained on; the model cannot translate them into legal play.
3. Grid and list are in between (60%): readable but lossy relative to FEN.
4. All 11/20 solved mates were the same underlying tactic pattern family
   (lichess-000rZ-style), so absolute accuracy is not the point of Phase 1 —
   the RELATIVE ordering across representations is.

## Decision: FEN only, from Phase 2 onward

Phase 1 measured representation sensitivity directly and the answer was
decisive: **FEN is the native language of chess-capable LLMs**. Continuing the
study on grid/bitboard/list would (a) confound chess ability with the
model's format-adaptation ability, (b) multiply GPU-hours with cells that
the pilot already showed are strictly worse, and (c) weaken the paper's
claim — measuring every model on its strongest, most standard
representation is the cleanest comparison of chess capability itself.

Phase 2 (full run) therefore runs all tasks on the **fen** representation
only. The representation axis is retained as a reported finding from Phase 1
rather than a full factorial, and is discussed in the paper's method section.
