# Chess SLM Benchmark

Compact, one-pass chess action-value models trained on Kaggle T4 GPUs. The
best verified CC-GAVN checkpoint has **4,762,088 parameters**, **3,710/4,000
MATE (92.75%)**, and **7,620/10,000 official puzzles (76.20%)**. The 9M
Searchless Chess reference scores 98.72% MATE and 86.13% puzzles. The fixed
benchmarks have been seen during development and are comparison sets, not
untouched holdouts.

## Tactical continuation

The next run starts at `ccgavn-5m-seed0-v2/checkpoint-3896658` and performs
2,000,000 additional updates with the existing model, trainer loss, batch 2048,
and cosine learning-rate schedule extended to step 5,896,658. It first labels
6,147,346 frozen Lichess puzzle roots, excluding the 9,995 exact matches in
the official 10,000-puzzle set. Every legal root move gets a Stockfish 16
50 ms evaluation and a 9M teacher distribution. Five Kaggle accounts prepare
the corpus and rotate T4 learner sessions. Full weights, optimizer, and RNG
state upload to Hugging Face before each session ends. Exact 4,000 MATE and
10,000-puzzle evaluations run at every 50,000 continuation steps.

The pipeline must pass a 500-root label, teacher, two-update learner, and
10 MATE / 20 puzzle smoke test before the full corpus stage starts. The
[live monitor](https://chess-slm-monitor.vedangpandeyy.workers.dev/) and
[Hugging Face dataset](https://huggingface.co/datasets/vedangfake/chess-slm-benchmark)
show the current phase and artifacts. The success gate is **more than 86.13%**
on all 10,000 official puzzles. A result exists only after its full frozen
evaluation completes.

## Main code

| Task | File |
| --- | --- |
| Frozen source and exclusions | `scripts/freeze_puzzle_corpus.py` |
| Stockfish action-value labels | `scripts/label_puzzle_shard.py` |
| 9M teacher distributions | `scripts/label_puzzle_teacher.py` |
| CC-GAVN learner and HF resume | `scripts/train_ccgavn.py`, `scripts/kaggle_checkpoint.py` |
| Exact MATE and puzzle evaluation | `scripts/eval_gavn.py` |
| Five-account supervisor | `scripts/watch_tactical.py`, `scripts/pipeline_tick.py` |
| Kaggle entry point | `kernels/tactical/worker.py` |
| Live website | `monitor/` |

The source manifest and run identity are pinned in `scripts/tactical_config.py`.
Every run that can lose progress must persist complete checkpoints and failure
status to Hugging Face, as specified in [AGENTS.md](AGENTS.md).
