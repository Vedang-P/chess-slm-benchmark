# Experiment Plan — Diagnose and Fix Cross-Benchmark Generalization in Gemma 4 GRPO Fine-Tuning

## Phasing: Phase 1 (reproduce + prepare) before Phase 2 (our actual experiments)

**Phase 1 — reproduce known baselines using only their own public code/data, and prepare every
dataset we'll need. Do not trust our own pipeline's numbers until it can reproduce a published
result on someone else's public artifacts first.**

Baselines with confirmed public code (reproducible):
- **AlphaMaze** (`github.com/menloresearch/visual-thinker`, Apache 2.0): (a) run their public
  checkpoint (`homebrewltd/AlphaMaze-v0.2-1.5B`) through *our* eval harness on their public
  `Maze-Bench-v0.2` — validates our eval code; (b) retrain their SFT+GRPO recipe from scratch on
  their public data (`Maze-Reasoning-v0.1`, `Maze-Reasoning-GRPO-v0.1`) and base model
  (DeepSeek-R1-Distill-Qwen-1.5B) — validates our training pipeline. Do (a) first, it's cheap
  (no training) and resolves inconsistent numbers found secondhand in earlier lit review passes.
- **GridRoute** (`github.com/LinChance/GridRoute`, code confirmed public): run their actual
  AoP-Dijkstra prompting code on Qwen2.5-7B (public, fits our hardware, also one of their tested
  models) and compare against their published number. Doubles as a real "algorithm-guided
  prompting, no training" baseline for our own results table later.

Datasets prepared but NOT reproducible as baselines (their eval code isn't public):
- **Lost in Aggregation**: maze data already in repo; their own site states evaluation/baseline
  code is "coming soon," not released as of 2026-07-13. We use the data with our own harness, but
  don't claim to reproduce their published model numbers.
- **MazeEval**: no code repository found at all. Not a Phase 1 target. If wanted later, it means
  building a compatible version from the paper's description — Phase 2/stretch scope.

**Phase 2 — our actual novel experiments** (the 7-step pipeline: baseline eval, single-format
training, transfer eval, failure analysis, mixed-format training, consistency-reward training,
final comparison — detailed below). Does not start until Phase 1 confirms the pipeline reproduces
known results.

## Immediate Next Step Within Phase 1

Cheapest, highest-value first check: AlphaMaze checkpoint + our eval harness (script:
`reproduce_alphamaze_eval.py`). No training needed, resolves a real open question (their reported
SFT-only vs SFT+GRPO numbers were inconsistent across secondary sources found earlier). Do this
before `train_grpo.py`'s timing test — if our eval harness can't correctly score a known-good
public checkpoint, a training-time GRPO number would be premature anyway.

## Objective

Measure whether GRPO fine-tuning improves Gemma 4 E2B's spatial navigation accuracy, whether that
improvement transfers to a structurally different benchmark than it was trained on, diagnose
*why* it does or doesn't via multi-angle failure analysis, and test whether a technique targeted
at the diagnosed failure mode (cross-format-consistency reward) closes the gap better than naive
mixed-format training.

## Dataset

- **Training (single-format):** AlphaMaze's `homebrewltd/Maze-Reasoning-GRPO-v0.1` (Apache 2.0),
  adapted to our task format, or regenerated GridRoute-style tasks via `src/grid_generator.py`.
- **Training (mixed-format):** the above, combined with an equivalent-scale sample of Lost in
  Aggregation-style tasks.
- **In-distribution eval:** GridRoute held-out tasks.
- **Out-of-distribution eval:** Lost in Aggregation (`data/lost_in_aggregation/`, already in repo).
- **Stretch:** MazeEval (partial observability) as a third, even more different transfer target.

## Model

Gemma 4 E2B (primary), DeepSeek-R1-Distill-Qwen-1.5B, Qwen2.5-1.5B/3B — all loaded and fine-tuned
via Unsloth (plain transformers+peft cannot attach LoRA to Gemma 4 at all — confirmed on hardware).

## Training Protocol

- Stage 1 (SFT): tokenized maze/path representations, step-by-step movement prediction.
- Stage 2 (GRPO), three reward variants tested in sequence:
  - **Single-format condition:** AlphaMaze's original reward (valid path + optimal length),
    trained on GridRoute only.
  - **Mixed-format condition:** same reward, trained on GridRoute + Lost in Aggregation combined.
  - **Consistency-reward condition:** reward augmented with a cross-format consistency term,
    design finalized after the failure analysis (see below) — trained on GridRoute only (the
    interesting test is whether a *targeted* reward, not more diverse data, closes the gap).
- Compute: ~9GB VRAM for GRPO per Unsloth's Gemma 4 documentation, comfortably within the A5000's
  24GB.
- Seed 42 throughout.

## Failure Analysis Protocol (Step 4 — do this thoroughly, it drives Step 6's design)

For every failed transfer-eval case (single-format model on Lost in Aggregation), categorize
along at least these independent axes:
- **Representation confusion:** does the model apply GridRoute-style assumptions to Lost in
  Aggregation's different wall/path encoding or topology structure?
- **Stage of failure:** does it fail at parsing the problem (wrong start/goal/obstacles extracted)
  vs. planning (valid parse, invalid or suboptimal path) vs. partial execution (starts correctly,
  loses track partway through)?
- **Cross-model consistency:** do Qwen2.5/DeepSeek-distill fail the same way, or differently —
  tells us whether the failure mode is Gemma-specific or general to the recipe.
- **Severity gradient:** does failure rate/type change smoothly with maze size/complexity, or is
  it a cliff?

Do not skip straight to aggregate pass/fail rates — the categorized breakdown is itself a result,
and it's what step 6's reward design should be built from, not designed speculatively in advance.

## Baselines

- Untrained Gemma 4 E2B on both benchmarks (already runnable via `train.py`).
- Single-format-trained model (this is itself a baseline for the mixed-format and
  consistency-reward conditions).
- AlphaMaze's own reported numbers as a recipe-correctness sanity check (different model/format,
  not a direct comparison).

## Evaluation Metrics

- Valid-path rate, optimal-path rate per benchmark per training condition (fixed `src/evaluation.py`).
- Primary: the ID-vs-OOD performance gap, compared across all three trained conditions
  (single-format / mixed-format / consistency-reward) plus the untrained baseline.
- Secondary: failure-mode category distribution (from Step 4), compared before/after each fix.

## Ablations

- SFT-only vs. SFT+GRPO, for each training condition (confirms the RL stage's specific
  contribution, not just SFT).
- Cross-model check (Qwen2.5-1.5B/3B, DeepSeek-R1-Distill-Qwen-1.5B) — is any effect (positive or
  negative) Gemma-specific or general to the recipe.

## Not Yet Built

- The actual SFT/GRPO training scripts for all three conditions (next, via Unsloth).
- The failure-analysis categorization tooling (design before results come in, not after).
- The consistency-reward's exact mechanism (finalize after Step 4's findings; needs a novelty
  spot-check before committing full training time to it).
- MazeEval loader (stretch).
