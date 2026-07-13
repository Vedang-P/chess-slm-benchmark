# Refined Research Idea (v3 — GRPO fine-tuning for Gemma 4 spatial reasoning)

Novelty score: 6/10 (see `novelty-assessment.md`).

## Problem Statement

On-device SLMs deployed as navigation agents need reliable spatial reasoning, but every existing
demonstration that RL fine-tuning improves this (AlphaMaze) or generalizes it (Ji et al.) has
only ever been tested on one model, on one benchmark format, in one modality. Whether the gains
hold when a *different* on-device model (Gemma 4 E2B) is trained and then tested on a
*structurally different* benchmark than it was trained on is untested, and directly matters for
whether this technique is a real deployment strategy or an overfit result.

## Proposed Approach

**Step 1 — Baseline.** Evaluate untrained Gemma 4 E2B on GridRoute and Lost in Aggregation
(valid-path rate, optimal-path rate). This is what `train.py` currently does.

**Step 2 — Train.** SFT then GRPO on Gemma 4 E2B via Unsloth, using AlphaMaze's public GRPO
training data (`homebrewltd/Maze-Reasoning-GRPO-v0.1`) as the starting point, adapted/regenerated
if needed to match our task format. Train on GridRoute-style tasks only (the "in-distribution"
condition).

**Step 3 — Evaluate transfer.** Run the fine-tuned model on both GridRoute (in-distribution) and
Lost in Aggregation (out-of-distribution, structurally different — tree-mazes with topology
annotations vs. GridRoute's rectangular-obstacle grids). The gap between these two numbers,
compared against the baseline model's gap, is the central result.

**Step 4 (stretch) — MazeEval.** Add partial-observability navigation as a third, even more
structurally different transfer target, if time allows. No loader built yet.

## What Is Novel

- First GRPO fine-tuning applied to Gemma 4 (any Gemma 4 variant) for spatial reasoning
  specifically — Gemma+GRPO is documented for math/general reasoning (Unsloth guides, Google's
  own Gemma community blog post) but not spatial/navigation tasks.
- First test of true cross-benchmark (structurally different, not just reworded) generalization
  for GRPO-trained spatial reasoning in a text-only (not vision-language) on-device model.
- Directly extends Xi et al.'s general "RL generalizes poorly across genuinely different
  environments" finding into a concrete, previously untested, practically relevant domain.

## Key Assumptions

1. AlphaMaze's GRPO recipe transfers to Gemma 4's architecture without major modification —
   plausible (Unsloth documents Gemma 4 GRPO support generally) but not yet confirmed for this
   specific maze/reward setup.
2. GridRoute and Lost in Aggregation are different enough in structure to count as a genuine
   cross-benchmark generalization test, not just a size/difficulty variation — true by
   construction (rectangular obstacle grids vs. tree-structured mazes with topology labels) but
   worth stating explicitly since Xi et al.'s "difficulty within an environment" vs. "genuinely
   unseen environment" distinction is exactly what this hinges on.

## Evaluation Plan

- **Models:** Gemma 4 E2B (primary). DeepSeek-R1-Distill-Qwen-1.5B (AlphaMaze's own base model —
  replication sanity check, confirms the pipeline reproduces their reported numbers before
  trusting results on untested models). Qwen2.5-1.5B/3B (generalization check — is any effect
  Gemma-specific or general to the recipe). All loaded/fine-tuned via Unsloth uniformly.
- **Benchmarks:** GridRoute (train + in-distribution eval), Lost in Aggregation (out-of-distribution
  eval), MazeEval (stretch).
- **Metrics:** valid-path rate, optimal-path rate (via `src/evaluation.py`, already fixed for the
  compliance/feasibility conflation and VMR bugs), reported separately per benchmark, before and
  after fine-tuning — the *change in the OOD-vs-ID gap* is the primary quantity, not raw accuracy
  numbers alone.
- **Baseline:** untrained Gemma 4 E2B on both benchmarks (Step 1, already runnable).

## Risks

- **Training may not converge or may not improve over baseline at all** — a real possible outcome,
  not just a caveat. Still publishable as a negative result if reported honestly, but changes the
  pitch significantly.
- **Generalization may simply fail** (per Xi et al.'s general finding) — this is actually a
  *reasonably likely* outcome given prior literature, not a low-probability edge case. Plan the
  paper so this is a legitimate, interesting finding, not a failed experiment.
- **Unsloth/Gemma 4 compatibility** — documented to work, but not yet verified end-to-end on our
  specific hardware for a full training run (only inference-level loading has been tested so far,
  and that hit real dependency version issues before being resolved).

## Next Actions

1. Follow Unsloth's Gemma 4 GRPO guide directly (adapt their Sudoku example) rather than
   continuing the from-scratch transformers+peft path.
2. Get/adapt AlphaMaze's public GRPO training data for our task format.
3. Run Step 1 (baseline eval) now — `train.py` supports this already.
4. Train, then run Step 3 (transfer eval) and compare.
