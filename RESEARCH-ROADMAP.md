# Sub-9M Searchless Chess Research Roadmap

Status: design locked for implementation, 2026-08-30

## Research question

Can a chess-specific model with fewer than 9M parameters retain or exceed the
released Ruoss et al. searchless action-value model on both of the project’s
primary measurements?

1. MATE-style two-choice selection on the exact four 1,000-position sets.
2. The official 10K Lichess puzzle protocol, where the complete solution
   sequence must be selected correctly.

The headline claim will only be made if the final checkpoint wins on a frozen
evaluation protocol and its confidence interval overlaps a genuine improvement,
not because it wins on a tuned subset.

## Repository audit

The current repository contains four materially different lines:

- The released Ruoss 9M/136M/270M action-value checkpoints and evaluators.
- A 5M width/depth student (`scripts/train_student.py`) trained from offline
  teacher bucket log-probabilities.
- Structured and unstructured pruning experiments on the 9M checkpoint.
- A separate natural-language MATE/GRPO line, which is not the right compute
  path for this objective.

The measured baselines currently available in `results/` are:

| model | MATE rows | accuracy | puzzle rows | accuracy |
|---|---:|---:|---:|---:|
| Ruoss 9M | 4,000 | 3,949/4,000 = 98.725% | 10,000 | 8,613/10,000 = 86.13% |
| Ruoss 136M | 4,000 | 3,976/4,000 = 99.400% | — | — |
| Ruoss 270M | 4,000 | 3,977/4,000 = 99.425% | — | — |

The 136M and 270M results are already present in the JSON artifacts; the
older `README.md` and `PROJECT-STATUS.md` incorrectly say that those runs are
pending. The 9M puzzle log has both a normal and EMA run (86.13% and 86.38%),
so the paper should state which checkpoint variant is used.

The current sub-9M evidence is not yet a fair failure result:

- The main 5M distillation run stopped around step 8,000, while its log targets
  22,000–30,000 steps.
- It trains on the manageable `data/test/action_value_data.bag` derivative,
  not the full ChessBench train distribution. This is appropriate for a smoke
  test, not a final claim.
- Several pruning/puzzle runs failed before evaluation due to checkpoint
  schema, dimension, or mode mismatches.
- Existing student training saves only the EMA parameter tree and has no
  optimizer/data-state/Hugging Face resume path. It cannot be the final trainer
  under this project’s persistence requirement.

### Hostile audit of the existing 5M student

The architecture is a useful compatibility baseline, but it has several
problems that make its previous score uninterpretable:

- The trainer applied `log_softmax` to a tensor that the official predictor
  already returned as log-probabilities. This changed the student distribution
  before both KL and CE, and is now fixed in `scripts/train_student.py`.
- The model is still a 77-token FEN-like decoder with a 1,968-action embedding
  table. It spends capacity learning board geometry and action structure that
  are already known exactly.
- Rank-pair construction materializes 16,358,344 Python tuples for the current
  1.84M-row set. That is a large CPU/RAM startup cost and silently makes the
  experiment dependent on a particular pair-generation policy.
- The run loads the complete student data and teacher matrix eagerly, has no
  validation stream, and saves only EMA weights. A killed job therefore loses
  optimizer state, RNG/data position, and the ability to reproduce a trajectory.
- The prior data is a derivative of the official test bag. It is valid for
  smoke testing but is not a defensible substitute for the ChessBench training
  distribution.
- The `--init` path restores weights but not optimizer moments, EMA history, or
  the random sampler state, so “resume” is actually a new optimization run.

## Proposed model: Geometric Action-Value Network (GAVN)

The primary model will be a small encoder-only network built around the actual
structure of chess rather than around the FEN character sequence.

### Input

- 64 square tokens, one per board square.
- 12 piece/empty channels per square, oriented to the side to move.
- A compact global token for side-to-move, castling rights, en-passant square,
  halfmove clock, and fullmove bucket.
- Optional horizontal reflection augmentation with an exact action transform.

### Trunk

- 8 pre/post-normalized residual blocks.
- 192 or 224 hidden width, 8 heads, and a 2x gated MLP; the candidate models
  are expected to land in the 3–6M range.
- A chess relation bias containing rank/file/diagonal/knight/king and ray
  relations, plus a learned low-rank state-conditioned bias. The fixed part
  supplies cheap geometry; the dynamic part lets the model turn relations on
  and off according to occupancy and game phase.

### Action head

- Factor every move into source square, destination square, and promotion.
- Produce a source-destination score instead of relying on a 1,968-row action
  embedding table.
- Retain a 128-bin return distribution for compatibility and calibration, but
  train an explicit scalar Q head and a pairwise ranking head because the
  benchmark only needs the ordering of two legal moves.

This is a controlled use of the ideas in Chessformer (square tokens, geometric
attention, and source-destination actions), adapted to the Ruoss action-value
target rather than human move prediction. The paper contribution will be the
combination of action-factorized value distillation, decision-aware losses, and
symmetry/hardness sampling; we will not call any individual ingredient novel.

## Training curriculum

### Gate 0: data and evaluator correctness

Before a long run, validate on a tiny hand-checkable batch:

- teacher outputs are normalized log-probabilities over 128 bins;
- action IDs, source/destination IDs, and promotions round-trip exactly;
- the model agrees with the official 9M evaluator on a deliberately copied
  checkpoint or on a synthetic toy model;
- MATE candidate ordering and puzzle sequence scoring are unchanged;
- no MATE or official puzzle FEN is in the training stream.

### Gate 1: controlled 5M baseline

Repair and rerun the existing architecture with the same data, but report
training examples, unique FENs, optimizer state, all checkpoints, and validation
curves. This tells us whether the previous result was simply under-trained.

Use the full legal-action distribution for the training stream rather than the
small test bag. The test bag is only for debugging and must not be used to
select the final model.

### Gate 2: GAVN ablation matrix

Use a held-out development split made from training positions, never the frozen
MATE/puzzle tests. Run the following matched comparisons:

1. FEN decoder baseline vs square-token baseline.
2. Fixed chess relation bias vs dynamic bias vs no bias.
3. action-ID head vs source-destination head.
4. return-distribution loss vs scalar-Q + pairwise loss vs the combined loss.
5. no augmentation vs side-to-move canonicalization + horizontal reflection.

Keep the parameter budget within 3M, 5M, and 7M bands. Select the best model
only by the development protocol; use the frozen test sets once for the final
table.

### Gate 3: decision-aware distillation

The loss is planned as:

`L = L_dist + λq L_Q + λrank L_rank + λbest L_best + λstate L_state`

- `L_dist`: temperature-scaled teacher return-distribution distillation.
- `L_Q`: Huber loss on the teacher expected return.
- `L_rank`: pairwise logistic/hinge loss, weighted by teacher margin.
- `L_best`: within-position softmax over legal moves, with teacher-Q targets.
- `L_state`: auxiliary state-value target derived from the teacher’s action set.

We will compare forward KL, reverse KL, and Jensen–Shannon distillation in a
small controlled sweep. The loss choice is not assumed from language-model
distillation results because this is a 128-bin decision distribution, not token
generation.

Sampling will have two phases: natural ChessBench sampling for broad coverage,
then a bounded hard-example mixture based on teacher uncertainty and small
action margins. The hard mixture will be capped and reported so it cannot turn
the benchmark into a hidden cherry-pick.

### Gate 4: teacher assistant and self-distillation

If the 3M/5M student cannot learn the 270M distribution, use the released 9M
model as a closer teacher for a warm-start, then distill from the 136M/270M
teacher. This tests the teacher-assistant hypothesis explicitly; it is not a
silent change to the main experiment.

Only after a competent checkpoint exists will we try a small on-policy set of
student disagreements, scored offline by the same oracle. This is an optional
follow-up, not a replacement for the clean supervised comparison.

## Compute plan

The three Kaggle accounts provide approximately 90 T4/P100 GPU-hours per
week. We will use them as independent, resumable jobs:

- Account A: repaired FEN baseline and optimizer/loss sweep.
- Account B: GAVN architecture/geometry sweep.
- Account C: best-configuration longer run plus held-out error analysis.

Each job must upload a complete checkpoint directory containing model weights,
optimizer state, scheduler state, RNG/data cursor, config, metrics, and failure
status to `vedangfake/chess-slm-benchmark` (or the explicitly configured HF
repo). Every trainer will support `--resume-from-hf`. Local Kaggle output is
treated as a cache, never as the only copy.

## Frozen success criteria

The first target is a model under 9M that meets or beats the 9M model on both
MATE and puzzles. The stronger target is a model under 6M that reaches at
least 98.2% on the no-explain MATE subset and at least 86.1% on the official
10K puzzles, with the exact checkpoint and all seeds reported.

For a paper-quality claim, we will additionally report:

- bootstrap confidence intervals and per-subset results;
- unique-FEN overlap and all leakage exclusions;
- full-sequence puzzle accuracy, first-move accuracy, and rating buckets;
- action-value calibration and Kendall rank correlation on held-out data;
- parameter count, serialized size, latency, and FLOPs per decision;
- error overlap between 9M, GAVN, 136M, and 270M;
- at least three seeds for the selected configuration if the initial win is
  smaller than the confidence interval.

## Literature anchors

- Ruoss et al., *Amortized Planning with Large-Scale Transformers: A Case
  Study on Chess*, arXiv:2402.04494 / NeurIPS 2024.
- Monroe et al., *Mastering Chess with a Transformer Model*, arXiv:2409.12272.
- Monroe et al., *Chessformer: A Unified Architecture for Chess Modeling*,
  arXiv:2605.19091 / ICLR 2026.
- Gu et al., *MiniLLM: Knowledge Distillation of Large Language Models*,
  ICLR 2024.
- Busbridge et al., *Distillation Scaling Laws*, arXiv:2502.08606.
- Schultz et al., *Mastering Board Games by External and Internal Planning
  with Language Models*, arXiv:2412.12119.
