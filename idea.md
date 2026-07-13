# Research Idea (v4 — GRPO fine-tuning for Gemma 4 spatial reasoning)

## Core Research Question

Does GRPO fine-tuning (AlphaMaze's recipe) improve Gemma 4 E2B's spatial/maze navigation
reasoning, and does that improvement generalize across structurally different benchmark formats
(GridRoute's obstacle grids, Lost in Aggregation's multi-scale mazes) — or does it only work on
whatever single format it was trained on, the way every existing GRPO-for-spatial-reasoning result
has only ever been tested on one format?

## Method

SFT then GRPO, applied to Gemma 4 E2B via Unsloth (official Gemma 4 GRPO support confirmed,
~9GB VRAM for the RL stage — comfortably within the A5000's 24GB). Evaluate the base model and
the fine-tuned model across both benchmarks; the generalization gap (or lack of one) between
train-format and transfer-format performance is the central result.

## Why This, Specifically

- AlphaMaze (arXiv:2502.14669) proved the recipe works, but only for DeepSeek-R1-Distill-Qwen-1.5B,
  only on one 5x5 maze format, only in-distribution.
- Ji et al. (arXiv:2507.13362) show GRPO beats SFT for spatial-reasoning OOD generalization, but
  on PaLI-Gemma2-3B (vision-language, not text-only) and only for rephrased-query OOD (not
  cross-benchmark structural transfer).
- Xi et al. (arXiv:2603.12011) find RL fine-tuning generalizes well within an environment but
  transfers weakly to genuinely unseen environments, for LLM agents generally — not tested for
  spatial/maze navigation specifically.
- Nobody has combined: text-only on-device SLM + Gemma 4 specifically + true cross-benchmark
  (not just cross-phrasing) structural transfer for spatial navigation.
- Either outcome is a real result: if generalization holds, that's a useful, actionable finding
  for on-device agent deployment; if it doesn't, that directly extends Xi et al.'s general finding
  into a concrete, previously-untested domain.

## Status

- Multilingual angle dropped (was the v2/v3 direction) — see `archive/multilingual-direction/`
  for that work, preserved but no longer pursued.
- Feasibility confirmed in principle: Unsloth documents Gemma 4 E2B GRPO training on ~9GB VRAM.
  Our own from-scratch (transformers+peft) feasibility check hit real dependency issues
  (transformers version, Pillow version) — recommend using Unsloth's documented path directly
  rather than continuing to debug a from-scratch implementation.
- Evaluation harness (`train.py`) reworked to run baseline (and later, fine-tuned) models across
  GridRoute and Lost in Aggregation. MazeEval integration not yet built.
- Actual SFT/GRPO training pipeline not yet built — next step, following Unsloth's Gemma 4 guide
  (https://unsloth.ai/docs/models/gemma-4/train), adapted from their Sudoku GRPO example to maze
  navigation using AlphaMaze's public training data (`homebrewltd/Maze-Reasoning-GRPO-v0.1`,
  Apache 2.0) as a starting point.

## Novelty

6/10 — see `novelty-assessment.md`. Real and correctly scoped: the general pattern (GRPO
generalizes better than SFT for spatial reasoning) is already shown in adjacent settings; this
specific combination (text-only, on-device, Gemma 4, true cross-benchmark structural transfer)
is not.

## Target Venue

Efficient and On-Device AI Agents Workshop @ NeurIPS 2026. Deadline August 29, 2026. See
`docs/workshop_info.md`.
