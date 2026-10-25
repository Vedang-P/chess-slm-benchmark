# Project Status

Updated 2026-10-25. The best verified result remains the baseline below. The
root-policy RL successor is deployed as a new experiment; its score is pending.

## Best model

Corrected-v2 CC-GAVN has 4,762,088 parameters and uses one-pass searchless
inference. Its final checkpoint is `ccgavn-5m-seed0-v2/checkpoint-3896658`.
A local copy of its weights and config is preserved outside the repository at
`../something-new-baseline-artifacts/ccgavn-5m-seed0-v2/checkpoint-3896658/`.

| Evaluation | Correct | Score |
| --- | ---: | ---: |
| MATE, 4,000 decisions | 3,710 | 92.75% |
| Official puzzles, 10,000 sequences | 7,620 | 76.20% |
| Searchless Chess 9M reference, MATE | 3,949 / 4,000 | 98.72% |
| Searchless Chess 9M reference, puzzles | 8,613 / 10,000 | 86.13% |

The baseline trails the 9M reference by 5.97 percentage points on MATE and
9.93 points on puzzles. Evaluation artifacts are archived at
`eval-results/ccgavn-5m-seed0-v2-3896k-final/eval-summary.json` in the Hugging
Face dataset `vedangfake/chess-slm-benchmark`.

## Reproduction and integrity

- The corrected-v2 run reached 3,896,658 steps, warm-starting from the earlier
  CC-GAVN checkpoint and using the frozen 102-shard schedule.
- MATE scores exactly 4,000 rows. Puzzle accuracy requires the entire solution
  sequence to be correct across all 10,000 puzzles.
- These sets have been inspected repeatedly during model development; they are
  fixed comparison sets, not untouched holdouts. MATE positions repeat across
  rows, so uncertainty should be clustered by position.
- Long runs must upload complete resumable checkpoints and status to Hugging
  Face. The shared checkpoint helpers are in `scripts/kaggle_checkpoint.py`.

## Root-policy RL experiment

- Starting checkpoint: `ccgavn-5m-seed0-v2/checkpoint-3896658`.
- Functional repair removes 869,440 softmax-null attention parameters and
  initializes field gates to one. Local comparison over 71 legal candidates
  found a maximum logit change of 1.36e-5, below the 1e-4 conversion gate.
- Five CPU verifier jobs label 2,000 roots each, scoring every legal move at
  Stockfish depth 14 and excluding frozen benchmark positions.
- One T4 learner runs 20,000 group-relative clipped policy updates, resuming
  complete weights, optimizer, RNG, and config from Hugging Face. The final
  checkpoint receives the existing checkpoint-scoped MATE/puzzle evaluation.
- The primary decision gate is more than 86.13% on all 10,000 puzzles. This
  has not been achieved or measured for RL yet.

The [monitor](https://chess-slm-monitor.vedangpandeyy.workers.dev/) shows only
this run's live phases and archived baseline/teacher comparison. GitHub Actions
drives the pool; Hugging Face holds restart state and final artifacts.
