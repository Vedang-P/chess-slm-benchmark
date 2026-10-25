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

The baseline training checkpoint and optimizer state are persisted to the
Hugging Face dataset `vedangfake/chess-slm-benchmark`. Long runs must support
resume and publish progress and failure status; see [AGENTS.md](AGENTS.md).

## Setup

Install dependencies with `make setup`. Use a GPU environment for training.
The Searchless Chess source tree is an external dependency at `searchless_chess/`.
