# Experiment Plan — Gemma 4 E2B GRPO Fine-Tuning for Cross-Benchmark Spatial Reasoning

## Objective

Measure whether GRPO fine-tuning improves Gemma 4 E2B's spatial navigation accuracy, and whether
that improvement transfers from the training benchmark (GridRoute) to a structurally different
one (Lost in Aggregation), or is format-specific like every prior GRPO-for-maze result.

## Dataset

- **Training:** AlphaMaze's public GRPO dataset (`homebrewltd/Maze-Reasoning-GRPO-v0.1`, Apache
  2.0), adapted to our task format, or regenerated GridRoute-style tasks via
  `src/grid_generator.py` if the format doesn't translate cleanly.
- **In-distribution eval:** GridRoute (held-out tasks, same format as training).
- **Out-of-distribution eval:** Lost in Aggregation (`data/lost_in_aggregation/`, already in repo)
  — tree-structured mazes with topology annotations, structurally different from GridRoute's
  rectangular-obstacle grids.
- **Stretch:** MazeEval (partial observability) — no loader built yet.

## Model

Gemma 4 E2B, fine-tuned via Unsloth (SFT then GRPO), following Unsloth's official Gemma 4 guide.
Not building the training loop from scratch — adapting their documented recipe.

## Training Protocol

- Stage 1 (SFT): tokenized maze/path representations, teach step-by-step movement prediction.
- Stage 2 (GRPO): reward = valid path + optimal path length, following AlphaMaze's reward design.
- Compute: Unsloth documents ~9GB VRAM for Gemma 4 E2B RL training — fits the A5000 (24GB) with
  large margin.
- Seed 42 throughout.

## Baselines

- Untrained Gemma 4 E2B on both benchmarks (already runnable via `train.py`).
- AlphaMaze's own reported numbers (86% SFT / 93% SFT+GRPO on their single format) as a sanity
  check that the recipe is implemented correctly, not as a benchmark we're directly compared
  against (different model, different task format).

## Evaluation Metrics

- Valid-path rate and optimal-path rate per benchmark (via the fixed `src/evaluation.py`).
- Primary quantity: the **gap** between in-distribution (GridRoute) and out-of-distribution (Lost
  in Aggregation) performance, for the fine-tuned model vs. the baseline model. A shrinking gap
  after fine-tuning suggests generalization; a stable or widening gap suggests overfitting to the
  training format (matching Xi et al.'s general finding).

## Ablations

- SFT-only vs. SFT+GRPO (matches AlphaMaze's own ablation, confirms the RL stage's specific
  contribution to any generalization effect, not just the SFT stage).
- Cross-model check with Qwen2.5-1.5B/3B and DeepSeek-R1-Distill-Qwen-1.5B (AlphaMaze's own base
  model, doubling as a replication sanity check) to see whether any generalization (or lack
  thereof) is Gemma-specific or general to the recipe. All models loaded/fine-tuned via Unsloth.

## Not Yet Built

The actual SFT/GRPO training script (next step, via Unsloth). MazeEval loader. Everything above
this line describes the plan; `train.py` currently only implements the baseline evaluation half.
