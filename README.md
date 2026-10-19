# Chess SLM Benchmark

Research code for compact, searchless chess action-value models and the
CC-GAVN-R post-training study. Training runs use Kaggle GPUs; complete model
and optimizer checkpoints are uploaded to Hugging Face so runs can resume.

## Current state

- Base: corrected-v2 CC-GAVN, 4,762,088 parameters.
- Latest recorded preview: 3,697/4,000 MATE and 7,577/10,000 puzzles at 3.4M
  steps. This is not the final run result.
- Post-training: CC-GAVN-R code is written, but no new training run or score
  has been produced. It adds a learned two-line refutation planner and keeps
  the planned model under 5M parameters. See
  [the plan](POSTTRAINING-PLANNER-PLAN.md).
- The repeated MATE and puzzle sets are development diagnostics, not untouched
  holdouts. See [project status](PROJECT-STATUS.md) for details and caveats.

## Run the core tools

Install dependencies with `make setup`. Use a GPU environment for training.
The main entry points are:

| Purpose | Entry point |
| --- | --- |
| Train the base model | `scripts/train_ccgavn.py` |
| Score MATE and puzzle benchmarks | `scripts/eval_gavn.py` |
| Build oracle-labeled refutation data | `scripts/refutation_data.py` |
| Train, evaluate, or export CC-GAVN-R | `scripts/posttrain.py` |
| Monitor active Kaggle runs | `scripts/watch_1b.py`, `scripts/watch_2b.py` |

Training and evaluation commands, experiment gates, and the five-account
compute schedule are documented in the post-training plan. The Cloudflare
project monitor lives in `monitor/`; deploy it from that directory with
`npx wrangler deploy` after installing its locked dependencies.

## Repository map

- `scripts/`: model, dataset, checkpoint, evaluation, and orchestration code.
- `kernels/`: Kaggle entry points and the shard-build scripts they run.
- `configs/`: frozen data schedules and the existing run-arm marker.
- `data/positions/`: compact benchmark inputs; large training data stays
  remote or in local ignored storage.
- `monitor/`: Cloudflare Worker dashboard.
- `figures/`: source diagram for the base architecture.
- `tests/`: correctness and recovery checks.

## Checkpoint requirement

Every long training or evaluation run must persist weights, optimizer and
scheduler state, RNG state, configuration, and metrics to Hugging Face; support
resume; upload a final artifact; and publish failure status. The shared helpers
are in `scripts/kaggle_checkpoint.py`; `scripts/posttrain.py` shows their use.
See [AGENTS.md](AGENTS.md) for the full rule.

## Research references

- [Literature review](lit-review.md)
- [Measured project status](PROJECT-STATUS.md)
- [CC-GAVN-R implementation and compute plan](POSTTRAINING-PLANNER-PLAN.md)
- [Searchless Chess paper](https://arxiv.org/abs/2402.04494) and
  [official code](https://github.com/google-deepmind/searchless_chess)
