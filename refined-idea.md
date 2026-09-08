# Refined Research Idea — CC-GAVN

## Problem Statement

Can a strictly sub-5M searchless chess action-value model improve the
accuracy–latency Pareto frontier set by the released 9M ChessBench transformer?
The earlier project result is not an answer: its best no-Q model was evaluated
through an untrained head. We first establish the corrected baseline, then test
whether action-conditioned geometric computation improves compact
distributional action-value distillation.

## Proposed Approach

CC-GAVN v1 uses 64 FEN square tokens and one token representing the candidate
UCI move (source square, destination square, promotion). All eight attention
layers jointly update board and candidate tokens with live chess-relation bias
and a state-conditioned dynamic bias. The candidate token's final state emits
the canonical 128-bin return distribution. At 208 width, eight layers, and
eight heads, it contains 4,762,088 trainable parameters.

Training samples only ChessBench train shards. It optimizes a temperature-
scaled teacher-distribution cross-entropy plus a smooth target around the raw
Stockfish return bucket. A 1% deterministic position-level development split
is excluded from updates. An exact horizontal reflection maps board squares,
castling rights, en-passant fields, and action IDs together, preserving the
target distribution without synthetic labels.

## What is Novel

- Candidate-action token participates in every geometry-aware board layer,
  rather than only a final action-scoring MLP.
- Rule-complete horizontal chess symmetry augmentation for a searchless
  action-value model.
- Strict ≤5M parameter accounting with distributional decision scoring,
  checkpoint-resumable training, and held-out development selection.
- A protocol audit that distinguishes trained distribution output from an
  untrained auxiliary scalar head.

## Key Assumptions

- Horizontal reflection preserves ChessBench action-return labels after exact
  FEN and UCI remapping.
- Candidate-conditioned interaction is more sample-efficient than a
  board-only encoder at this parameter budget.
- Development distribution loss correlates sufficiently with final action
  quality to select a checkpoint without repeated frozen-test probing.

## Evaluation Plan

Compare legacy GAVN, corrected-relations GAVN (4.999M), and CC-GAVN (4.762M)
with matched data and training budget. Select by development loss, then run the
official distribution-expectation engine once on all four frozen MATE subsets
and the full 10K puzzle suite. Report seeds, bootstrap confidence intervals,
latency, parameter count, calibration, and error overlap with the 9M teacher.

## Risks

The architecture may not beat the 9M teacher. If it does not, this work must
report the measured frontier honestly rather than force a SOTA claim. A result
on only a 1,000-puzzle slice is not sufficient for the workshop paper.

## Next Actions

1. Replace invalid historical q-head metrics with corrected distribution-head
   evaluations.
2. Run the CC-GAVN smoke gate, then one full seed after GPU availability.
3. Run matched controls and promote only development-selected checkpoints to
   frozen evaluation.
4. Expand to three seeds only when a development improvement is observed.
