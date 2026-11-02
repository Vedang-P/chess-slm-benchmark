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

Sprint arms vary one change at a time (control, pointwise mix, listwise mix,
50% tactical, higher LR); the main run uses the recipe that validates best.
Stage 2 is exact-expectation policy optimisation (see `scripts/` once added)
against a plain supervised-continuation control.

Success gate: strictly above 86.13% on all 10,000 official puzzles.
Pre-registered checkpoints: about 80% by 2026-11-23 and 83% by 2026-12-14.

The [monitor](https://chess-slm-monitor.vedangpandeyy.workers.dev/) shows
validation curves per run, official evaluations, pool quota and supervisor
state.
