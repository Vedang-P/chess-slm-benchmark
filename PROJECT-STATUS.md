# Project Status

Updated 2026-10-20. Detailed historical notes remain in Git history.

## Research target

Build a chess action-value model strictly below 5M parameters that surpasses
the gap to the released Searchless Chess 9M model on MATE decisions and full
Lichess puzzle sequences. MATE and puzzle sets have been inspected repeatedly;
they are development diagnostics, not untouched holdouts.

## Model and latest recorded results

The active model is corrected-v2 CC-GAVN: 4,762,088 parameters, eight layers,
width 208, and eight heads. It conditions every geometric layer on a candidate
move. The final training and evaluation runs have been reported complete, but
their final summary is not present in this checkout. The latest local
corrected-v2 result is a preview, so do not report it as the final score.

| Run | MATE / 4,000 | Puzzles / 10,000 | Status |
|---|---:|---:|---|
| CC-GAVN @1.62M | 3,668 (91.70%) | 7,431 (74.31%) | complete |
| corrected-v2 @3.4M | 3,697 (92.425%) | 7,577 (75.77%) | historical preview |
| Searchless Chess 9M reference | 3,949 (98.72%) | 8,613 (86.13%) | measured reference |

Use the checkpoint-scoped Hugging Face evaluation summary and per-example
rows to record the completed final run. Keep the preview and final results
distinct.

## Integrity requirements

- Score exactly 4,000 MATE rows and 10,000 official puzzles. A puzzle counts
  only when its full solution sequence is correct.
- Preserve checkpoint identity, configuration, weights, optimizer/scheduler
  state, RNG/data state, metrics, and failure status on Hugging Face; support
  resume. See `AGENTS.md`.
- The corrected-v2 recipe uses a stage-relative shard schedule, benchmark and
  development exclusions across all shards, castling-safe reflection, and
  learning-rate updates across optimizer groups.
- Evaluation rows are checkpoint-scoped and resumable. Cleanup fails closed if
  saved rows do not match the checkpoint and dataset identity.
- MATE positions overlap across subsets. Report position-clustered
  uncertainty; row-level binomial intervals overstate precision.

## Active code

- `scripts/train_ccgavn.py`: CC-GAVN model and trainer.
- `scripts/eval_gavn.py`, `scripts/analyze_mate.py`,
  `scripts/eval_recovery.py`: exact evaluation, analysis, and recovery.
- `kernels/ccgavn-2b/train_2b.py`: Kaggle training entry point.
- `scripts/watch_2b.py`, `scripts/ensure_eval_preview.py`: run orchestration.
- `configs/` and `data/positions/`: frozen shard tags and benchmark inputs.
- `tests/test_correctness.py`: model, data, and recovery safeguards.

## Next research step

CC-GAVN-R now has source for the shared-weight refutation model, oracle and
on-policy labeling, ranking/planner/distillation training, and persistent full
evaluation. See `scripts/refutation.py`, `scripts/refutation_data.py`,
`scripts/posttrain.py`, and `POSTTRAINING-PLANNER-PLAN.md`.

The analytic parameter total is 4,846,255; a runtime guard enforces <5M.
No planner checkpoint or score exists from this work. Python syntax checking
was rejected by workspace execution policy, so runtime correctness and T4
resource usage are unverified. The latest remote results page also failed to
load; the completed base run's final score still needs artifact confirmation.
The 23-paper review supports a controlled counterfactual-refutation study,
not a guaranteed score improvement. The user permits learned internal
planning while retaining the strict 5M cap and no external engine at inference.

The project monitor source now lists this study as planned and blocked at
preflight; no score gain is claimed. Its deployment dry run could not resolve
the declared `chess.js` dependency from the local checkout, so the website
update is not published. Under the project P0 rule, deployment stopped at this
failed check rather than changing the build setup without review.
