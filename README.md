# Chess SLM Benchmark

Compact, one-pass chess action-value models trained on Kaggle T4 GPUs. The
best verified CC-GAVN checkpoint has **4,762,088 parameters**, **3,710/4,000
MATE (92.75%)**, and **7,620/10,000 official puzzles (76.20%)**. The 9M
Searchless Chess reference scores 98.72% MATE and 86.13% puzzles. The fixed
benchmarks have been seen during development and are comparison sets, not
untouched holdouts. See [PROJECT-STATUS.md](PROJECT-STATUS.md) for the current
experiment and what has failed.

## Conservative continuation (cont1)

The next run restarts from `ccgavn-5m-seed0-v2/checkpoint-3896658` (via the
verified 3,895,352-parameter transplant) with a low constant learning rate, an
EMA of the weights, a broad-replay plus tactical-root batch mix, a listwise loss
over each root's legal moves, and validation on 4,000 held-out puzzles that are
disjoint from the official 10,000. Five Kaggle accounts run back-to-back
11.5-hour learner sessions; every session uploads a complete checkpoint to
Hugging Face before its cutoff and the next account resumes it. The success
gate is **more than 86.13%** on all 10,000 official puzzles; a result exists
only after its full evaluation completes.

## Main code

| Task | File |
| --- | --- |
| Held-out splits, tactical pack, stateless samplers | `scripts/cont_data.py`, `scripts/build_cont_data.py` |
| Continuation trainer (EMA, mix, listwise loss, guard, resume) | `scripts/train_cont.py` |
| Batched exact-rules puzzle and MATE evaluation | `scripts/cont_eval.py` |
| Sequential five-account supervisor | `scripts/watch_cont.py`, `scripts/cont_tick.py`, `.github/workflows/cont-tick.yml` |
| Kaggle entry point and helpers | `kernels/cont/worker.py`, `scripts/kaggle_pool.py`, `scripts/launch_cont.py` |
| Later-ply puzzle roots: shard labeler, work plan, CPU labeling fleet, training pack, run comparison | `scripts/label_ply_shard.py`, `scripts/ply_data.py`, `scripts/build_ply_plan.py`, `scripts/watch_ply_labels.py`, `kernels/cont/label_worker.py`, `scripts/build_ply_pack.py`, `scripts/compare_ply_runs.py` |
| CC-GAVN model and original trainer | `scripts/train_ccgavn.py`, `scripts/kaggle_checkpoint.py` |
| Original exact MATE and puzzle evaluation | `scripts/eval_gavn.py` |
| Live website | `monitor/` |

Run identity and the recipe are pinned in `scripts/cont_config.py`. The earlier
tactical-continuation code (`scripts/watch_tactical.py`, `kernels/tactical/`)
is kept for its labeled data and history; its workflow is disabled. Every run
that can lose progress must persist complete checkpoints and failure status to
Hugging Face, as specified in [AGENTS.md](AGENTS.md).
