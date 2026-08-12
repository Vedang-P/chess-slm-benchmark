# Refined Research Idea

## Problem Statement
Small language models (SLMs) that reason about chess in natural language need
high-quality, sample-efficient training data. Existing chess datasets for LLM
training are either whole-game text (unverifiable per-move quality, no
reasoning signal) or position-level but phase-unlabeled (MATE: 1M expert
positions; C1: 39k engine-verified teacher traces sampled by difficulty).
Neither the training-data community nor the benchmark community treats the
opening / middlegame / endgame axis as a first-class dimension — yet these
phases demand fundamentally different reasoning (memorized theory, tactical
calculation, endgame technique), and LLM endgame reasoning is essentially
unmeasured. We propose a phase-segregated, engine-verified trace dataset for
SLM distillation, and a phase-stratified benchmark released from the same
pipeline.

## Proposed Approach
1. **Position source.** MATE train zips (208k rows, already local under
   `data/raw/mate-train/`), deduped by FEN, sampled per phase.
2. **Phase classifier (published artifact).** Deterministic, documented
   thresholds: opening = ply ≤ 12 (from game start) or first N book plies;
   endgame = non-king material ≤ threshold (e.g., ≤ 13 points or
   piece-configuration rules); else middlegame. Publish the classifier with
   the dataset so boundaries are reproducible.
3. **Teacher traces.** deepseek-v4-flash, unbounded thinking, per position,
   same prompt family as our MATE campaign; **phase-aware filtering**: the
   Stockfish ±100cp eval-stability filter is calibrated per phase (different
   thresholds for open middlegames vs K+P endgames), and the teacher's
   phase-specific weaknesses are audited.
4. **Training.** gemma 4 E2B QLoRA SFT on the segregated traces. Three arms:
   (a) phase-uniform mixture, (b) phase-curriculum (opening → middlegame →
   endgame), (c) C1-style difficulty-balanced baseline. Compare accuracy,
   tokens-per-correct, and per-phase breakdown vs base gemma and deepseek.
5. **Benchmark.** Hold out a stratified slice (e.g., 300 positions/phase,
   FEN-deduped against all training traces and MATE test sets, Stockfish
   ground truth + per-position metadata). This is the primary contribution:
   a phase-stratified instrument for rating any chess model (base gemma,
   MATE-LoRA, trace-LoRA, deepseek, frontier baselines) with per-phase
   accuracy.

## What is Novel
- **Phase axis as training-data design dimension** — no prior work
  segregates trace-distillation data by opening/middlegame/endgame.
- **Phase-stratified benchmark** — no LLM chess benchmark stratifies by game
  phase; endgame tier is nearly absent from LLM chess evals.
- **Phase-aware trace filtering** — filtering thresholds calibrated per phase,
  acknowledging teacher weaknesses differ by phase (deepseek: openings from
  memory, endgames weakest).
- **Published phase classifier** — reproducible boundaries for chess ML
  (currently no consensus definition exists).

## Key Assumptions
- Phase-stratified supervision transfers to stronger full-game play than
  uniform sampling (testable vs C1's 39k).
- deepseek's traces are informative enough per phase that filtering by eval
  stability retains sufficient volume (endgame is the risk: weakest teacher
  play + tightest filter).
- MATE train positions are a sufficient, non-contaminated source (no overlap
  with MATE test sets used in our 4×1000 eval).
- A 2B student can absorb phase-specific reasoning from ~40k verified traces.

## Evaluation Plan
- **Training:** 3 arms above, same hyperparameters; report per-phase accuracy,
  overall accuracy, tokens-per-correct, legality rate.
- **Benchmark:** release the phase-stratified holdout (300/phase); evaluate
  base gemma, MATE-LoRA, trace-LoRA (each arm), deepseek-v4-flash, and MATE
  fine-tune anchors (63.5/89.7/94.6/95.2) + GPT-4/Claude zero-shot anchors.
- **Game strength:** Elo vs Stockfish `UCI_Elo` anchors (1400/1800/2200/2600)
  for the trained models, to counter the whole-game objection
  (Complete Chess Games, 2501.17186) with game-level evidence.
- **Hygiene:** FEN-dedup vs training traces and MATE test sets; report
  position-source provenance per phase.

## Risks
- Training arm may not beat C1-style uniform sampling → benchmark + classifier
  still stand as the contribution; the training comparison becomes a
  "uniform vs phase-stratified" ablation either way.
- Endgame trace volume may be too thin after filtering (teacher weakness +
  tight filter) → relax endgame filter or supplement endgame positions from
  tablebase/lichms sources.
- Phase classification boundary disputes → publish thresholds and show
  robustness (accuracy stable under ±2-ply / ±1-point threshold shifts).
- MATE train positions overlap with the MATE test sets → dedup is mandatory
  and already in the pipeline design.

## Next Actions
1. Build the phase classifier + position sampler (MATE train → 3 phase pools,
   dedup, metadata schema). Verify class balance.
2. Build phase-aware trace generation (deepseek per position, phase-specific
   filters), reusing run_mate_eval.py machinery with a train-position task
   file.
3. Build the benchmark holdout (stratified, FEN-deduped, SF ground truth).
4. Train 3 arms on RunPod; evaluate on benchmark + Elo battery.
5. Release dataset + classifier + benchmark (HF), write up.
