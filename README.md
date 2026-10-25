# Chess SLM Benchmark

Research code for compact, searchless chess action-value models. The current
baseline is corrected-v2 CC-GAVN, trained on Kaggle GPUs and evaluated against
fixed MATE and Lichess puzzle sets.

## Current best result

- Model: CC-GAVN, 4,762,088 parameters, one-pass searchless inference.
- Checkpoint: `ccgavn-5m-seed0-v2/checkpoint-3896658`.
- A local copy of its weights and config is kept outside this repository at
  `../something-new-baseline-artifacts/ccgavn-5m-seed0-v2/checkpoint-3896658/`.
- MATE: 3,710 / 4,000 (92.75%).
- Puzzles: 7,620 / 10,000 (76.20%).
- 9M Searchless Chess reference: 98.72% MATE / 86.13% puzzles.

These benchmark sets were inspected during development; they are fixed
comparison sets, not untouched holdouts. See [project status](PROJECT-STATUS.md)
for the run identity and evaluation caveats.

## Automated RL run

The successor run starts from the verified checkpoint above. It removes the
attention parameters that cancel in softmax, preserving the baseline function,
then trains a 3.90M-parameter root policy with exact-depth Stockfish rewards.
Five Kaggle CPU jobs collect 10,000 all-legal-move labels from disjoint corpus
shards. A supervisor rotates the learner across available T4 accounts and
launches the frozen 4,000-row MATE / 10,000-puzzle evaluation after 20,000 RL
updates. This is an experiment; no RL benchmark result exists yet.

GitHub Actions runs the supervisor and publishes the [live monitor](https://chess-slm-monitor.vedangpandeyy.workers.dev/)
every 10–30 minutes. The Kaggle jobs upload resumable labels, model weights,
optimizer state, and status to [Hugging Face](https://huggingface.co/datasets/vedangfake/chess-slm-benchmark).
The primary success gate is puzzle accuracy above the 9M reference's 86.13%.

## Core files

| Purpose | Path |
| --- | --- |
| Train CC-GAVN | `scripts/train_ccgavn.py` |
| Evaluate MATE and puzzles | `scripts/eval_gavn.py` |
| Recover interrupted evaluation | `scripts/eval_recovery.py` |
| Build and audit the training corpus | `scripts/build_full_dataset.py`, `scripts/audit_train_contamination.py` |
| Kaggle training entry point | `kernels/ccgavn-2b/train_2b.py` |
| Frozen benchmark positions | `data/positions/` |
| Project monitor | `monitor/` |
| RL verifier, trainer, and supervisor | `scripts/rl_labels.py`, `scripts/train_rl.py`, `scripts/watch_rl.py` |

The baseline training checkpoint and optimizer state are persisted to the
Hugging Face dataset `vedangfake/chess-slm-benchmark`. Long runs must support
resume and publish progress and failure status; see [AGENTS.md](AGENTS.md).

## Setup

Install dependencies with `make setup`. Use a GPU environment for training.
The Searchless Chess source tree is an external dependency at `searchless_chess/`.
