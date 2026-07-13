# Refined Research Idea (v4 — diagnose, then fix cross-benchmark generalization)

Novelty score: 6/10 (see `novelty-assessment.md` — Step 6 checked against Elhady et al.,
arXiv:2606.01464, which already does consistency-reward RL for cross-*lingual* math; our
contribution there is testing whether it transfers to cross-*format* spatial reasoning, not
inventing the mechanism).

## Problem Statement

AlphaMaze proved SFT+GRPO can take a small model's maze-solving from ~0% to 93%, and separate
work shows GRPO-trained models generalize better than SFT-trained ones for spatial reasoning —
but only within the same task, reworded. Whether that generalization holds across genuinely
different benchmark structures is untested, and if it doesn't hold, nobody has proposed a fix
targeted at *why* it fails rather than just training on more data and hoping. This project answers
both: does it generalize, why (or why not), and can a technique designed around the specific
failure mode close the gap better than naive mixed-format training alone.

## Proposed Approach (7 steps)

1. **Baseline** (built, runnable): untrained Gemma 4 E2B on GridRoute + Lost in Aggregation.
2. **Train single-format**: SFT + GRPO on GridRoute only, via Unsloth, following AlphaMaze's
   recipe (adapt their public `Maze-Reasoning-GRPO-v0.1` data/reward design).
3. **Transfer eval**: single-format model on Lost in Aggregation (OOD).
4. **Critical failure analysis**: categorize *how* the model fails on the OOD benchmark, from
   multiple angles (representation confusion, planning vs. execution errors, error localization
   within the path, cross-model comparison against Qwen2.5/DeepSeek-distill). This is the
   diagnostic centerpiece and must be thorough — it directly determines what step 6 targets.
5. **Train mixed-format**: SFT + GRPO on GridRoute + Lost in Aggregation combined. Naive-fix
   baseline. Evaluate whether this alone closes the gap found in step 3/4.
6. **Cross-format-consistency reward**: propose a GRPO reward modification, designed around
   step 4's specific findings, that explicitly targets cross-format consistency rather than
   per-format correctness alone (e.g., present the same underlying problem in both formats during
   training, reward consistent correct answers across representations).
7. **Final comparison**: single-format vs. mixed-format vs. consistency-reward model, across both
   benchmarks. Central result table.

## What Is Novel

- First GRPO fine-tuning applied to Gemma 4 for spatial reasoning.
- First test of true cross-benchmark (not reworded-query) structural generalization for
  GRPO-trained spatial reasoning in a text-only on-device model.
- First test of whether consistency-reward RL (Elhady et al.'s technique, validated only for
  cross-lingual math so far) transfers to cross-*format* spatial reasoning — an application/
  transfer contribution, not a new method. Framed honestly, motivated by original failure
  analysis (Step 4) rather than applied speculatively.
- Extends "Beyond Specialization"'s classical-RL finding (diverse training fixes cross-environment
  transfer) into the LLM/GRPO setting, and goes further by testing whether a *targeted* technique
  beats naive data mixing.

## Key Assumptions

1. GridRoute and Lost in Aggregation are structurally different enough to be a genuine
   cross-benchmark test (true by construction — rectangular obstacle grids vs. tree-structured
   mazes with topology labels).
2. The failure analysis (step 4) will surface a specific, addressable failure mode, not a diffuse
   mix of unrelated errors — if failures are too heterogeneous, step 6's targeted technique has
   nothing clear to target. Worth checking early, not assuming.
3. AlphaMaze's GRPO recipe transfers to Gemma 4's architecture without major modification beyond
   the Unsloth loading fix already needed.

## Evaluation Plan

- **Models:** Gemma 4 E2B (primary). DeepSeek-R1-Distill-Qwen-1.5B (AlphaMaze's own base —
  replication sanity check). Qwen2.5-1.5B/3B (recipe-generalization check). All via Unsloth.
- **Benchmarks:** GridRoute (train + in-distribution eval), Lost in Aggregation (OOD eval).
  MazeEval as a stretch third benchmark (no loader built yet).
- **Metrics:** valid-path rate, optimal-path rate (via `src/evaluation.py`) per benchmark per
  training condition (baseline / single-format / mixed-format / consistency-reward). Primary
  quantity: the ID-vs-OOD gap, and how each training condition changes it. Secondary: failure-mode
  category breakdown (step 4), not just aggregate rates.
- **Baseline:** untrained model (already runnable); AlphaMaze's own reported numbers as a
  recipe-correctness sanity check, not a direct comparison (different model/format).

## Risks

- **Training may not converge or may not beat baseline** — real possible outcome, plan for it.
- **Generalization may already hold without any fix** (contradicting Xi et al.) — in which case
  steps 5/6 become "we tested whether a fix was needed and found it wasn't," still a valid,
  interesting result, but changes the pitch from "we fixed X" to "we found X doesn't need fixing
  here, contrary to the general pattern."
- **Failure modes may be too heterogeneous** for step 6's targeted reward to have a clean target —
  check this as soon as step 4's data exists, before investing time designing step 6.
- **Unsloth/Gemma 4 compatibility** for the actual training loop (not just loading) is not yet
  verified end-to-end on our hardware.

## Next Actions

1. ~~Novelty spot-check on Step 6~~ — done, see `novelty-assessment.md`.
2. **Empirical timing test, Gemma 4 E2B only:** short GRPO run (~50-100 steps) to get a real
   wall-clock/GPU-hour number on the A5000 before committing to the full plan. This determines
   whether all 4 models x 3 conditions is realistic or needs cutting down.
3. Based on that number, decide final model/condition scope, then follow Unsloth's Gemma 4 GRPO
   guide for the real training runs (steps 2 and 5 of the pipeline).
4. Run step 1 (baseline eval, already built) to get a fresh baseline number.
5. Design the failure-analysis categorization scheme (step 4) before results come in, so
   categorization is principled and decided in advance, not post-hoc.
