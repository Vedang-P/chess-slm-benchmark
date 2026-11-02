# Project Status

Updated 2026-11-02.

## Best verified model

`ccgavn-5m-seed0-v2/checkpoint-3896658` is a 4,762,088-parameter, one-pass
CC-GAVN model (3,895,352 parameters after the verified function-preserving
transplant). Its frozen evaluation is archived at
`eval-results/ccgavn-5m-seed0-v2-3896k-final/eval-summary.json` in the
[Hugging Face dataset](https://huggingface.co/datasets/vedangfake/chess-slm-benchmark).

| Evaluation | CC-GAVN | 9M reference |
| --- | ---: | ---: |
| MATE, 4,000 decisions | 3,710 (92.75%) | 3,949 (98.72%) |
| Official puzzles, 10,000 full sequences | 7,620 (76.20%) | 8,613 (86.13%) |

The benchmarks were inspected during development; they are fixed comparison
sets, not untouched holdouts. MATE rows repeat positions. Model selection in
the current experiment uses a separate held-out validation set (below), and a
sealed test set is scored once at the end.

## What did not work

Two post-training attempts from this checkpoint lowered puzzle accuracy:
root-policy RL (v1/v4: 56.87% / 51.07%) and the tactical distillation
continuation (nine evaluations between 55.6% and 64.1%, last 55.89% at step
4,346,658). That continuation restarted at 26% of the original peak learning
rate, trained only on first-decision puzzle roots with no broad replay, and
overwrote general play. It was stopped on 2026-11-02; its checkpoints and the
labeled puzzle shards remain on Hugging Face.

## Current experiment: conservative continuation (cont1)

Same 3.9M architecture, restarted from checkpoint 3,896,658 with:

* constant 2e-5 learning rate after warmup, and an EMA of the weights (the EMA
  weights are what is evaluated);
* every batch mixes the original teacher-distilled corpus (replay) with whole
  tactical roots from the labeled puzzle shards;
* a listwise loss over each tactical root's legal moves;
* held-out validation: 4,000 Lichess puzzles disjoint from the official 10,000
  and rating-matched to them, scored with the exact harness rules during
  training, plus a regression guard that halts the run if validation stays more
  than 3 points below the starting model;
* sequential 11.5-hour learner sessions across the five Kaggle accounts; each
  session saves a complete checkpoint to Hugging Face and the next resumes it.

## Sprint results (30,000 steps each, one Kaggle session per arm)

All arms start from checkpoint 3,896,658 with the same trainer, seed and data
stream; only the listed change differs. Validation is the 4,000 held-out puzzles
(start: 77.98%); the official 10,000 puzzles and 4,000 MATE rows are logged once at
the end and were not used to choose anything.

| Arm | Change | Validation | Official puzzles | MATE | Broad dev loss |
| --- | --- | ---: | ---: | ---: | ---: |
| start | verified checkpoint | 77.98% | 7,620 (76.20%) | 3,710 | 2.878 |
| a0 | broad replay only (control) | 77.70% (-0.27) | 7,631 (76.31%) | 3,703 | 2.879 |
| a1 | 30% tactical roots, pointwise losses | 79.72% (+1.75) | 7,910 (79.10%) | 3,671 | 2.978 |
| a3 | 50% tactical roots + listwise loss | 87.15% (+9.18) | 8,619 (86.19%) | 3,804 | 3.078 |
| a4 | 30% tactical + listwise, lr 5e-5 | 86.75% (+8.78) | 8,625 (86.25%) | 3,764 | 2.996 |

What it shows:

* The low constant LR with EMA is safe: the control moved by -0.27 points, so the
  earlier regression came from the data and loss design, not from continuing.
* The listwise loss over each root's legal moves is what matters. The same tactical
  roots with a pointwise loss gain 1.75 points; with the listwise loss they gain 9.
* Both listwise arms are above the 86.13% reference (8,613 puzzles) at step 30,000,
  by 6 and 12 puzzles. That is inside the roughly 0.35-point sampling error of a
  10,000-puzzle score, so it is a tie, not yet a win. The sealed test puzzles and
  longer training decide it.
* The gain is paid for on the broad corpus (dev loss +4% for a4, +7% for a3) and MATE
  stays below the 9M reference (94.1% and 95.1% vs 98.72%). This is a tactics
  specialist trained on puzzle-distribution positions (benchmark puzzles and
  positions excluded), not a uniformly stronger model.
* 31% of the starting model's validation failures were at later plies, not the first
  decision; the labeled roots only cover the first decision.

The main run is arm a4 continued without a step limit (resumed from its own
checkpoint, optimizer and EMA). Stage 2 compares exact-expectation policy
optimisation (`--rl-weight 1.0`, KL anchor 0.05) against a control with the same
branch point and data (runs r1 and c1, both from the a4 EMA weights at step 30,000).

The [monitor](https://chess-slm-monitor.vedangpandeyy.workers.dev/) shows
validation curves per run, official evaluations, pool quota and supervisor
state.
