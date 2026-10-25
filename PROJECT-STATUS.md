# Project Status

Updated 2026-10-25. The project is reset to its best verified baseline; no
successor experiment is active.

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

## Working tree

The maintained code covers the CC-GAVN baseline, its frozen evaluation, the
training corpus, and the existing project monitor. The next research direction
is undecided; use this result as the starting point for that decision.
