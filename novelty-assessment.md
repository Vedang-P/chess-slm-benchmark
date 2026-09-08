# Novelty Assessment — CC-GAVN

## Assessment: 6/10 before experiments

CC-GAVN is an incremental-but-plausibly publishable architecture contribution,
not a new learning paradigm. Its novelty comes from a specific combination not
found in the closest ChessBench work: candidate-action conditioning at every
geometric board-encoder layer, exact chess rule-state symmetry augmentation,
and a leakage-safe distributional action-value evaluation protocol below a
strict 5M-parameter budget.

## Direct overlap

- Ruoss et al. already establish searchless action-value prediction on
  ChessBench. CC-GAVN uses the same task and must beat their 9M model rather
  than claim a different benchmark.
- Chessformer already establishes square tokens, geometric attention, and a
  structured move head. CC-GAVN must not claim those components as novel.
- Finite-group game networks establish the general benefit of equivariance;
  CC-GAVN's contribution is careful use of a valid chess reflection including
  FEN rule state, not the general symmetry principle.
- Distillation objectives and teacher-assistant ideas are established. Direct
  KL plus a raw-return target is a control, not a new loss theory.

## What is distinct

The prior repository GAVN uses board-only attention followed by a final
source/destination MLP. CC-GAVN adds a candidate move token to every layer, so
attention can represent action-specific tactical interactions before the final
return distribution. The model retains the official action-value interface
(one candidate per forward pass), which lets it plug into the frozen Searchless
Chess puzzle harness without runtime search.

The paper-worthy insight would be empirical: at the 5M scale, candidate
conditioning and exact rule-state augmentation recover enough conditional
structure to alter the accuracy–latency frontier. That requires matched
ablations: legacy GAVN versus corrected-relations GAVN versus CC-GAVN, each
with the same data, development split, budget, and at least three seeds for the
winner.

## Risks

The new token may add computation without useful capacity, and an 85.0% MATE
correction for the legacy checkpoint does not predict CC-GAVN performance.
The official puzzle result has not yet been corrected at the time of this
assessment. A model that improves pairwise MATE but not full puzzle sequences
would be an incomplete frontier claim. The target of reliably beating the 9M
teacher is therefore a hypothesis, not a promised outcome.
