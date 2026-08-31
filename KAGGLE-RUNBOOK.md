# Kaggle Runbook: Full-Data Sub-9M Chess Experiments

This document describes the complete workflow added in commit `7b008cf`.
Kaggle kernels should pull the repository from GitHub and run the scripts and
notebooks from that checkout.

## What was added

### New research model

`scripts/train_gavn.py` implements GAVN, the Geometric Action-Value Network.
It is a small PyTorch model with:

- 64 board-square tokens instead of a character-level FEN sequence;
- square embeddings and chess relation-aware attention;
- a state-conditioned geometric attention bias;
- source-square, destination-square, and promotion action features;
- 128-bin teacher distribution distillation;
- explicit scalar action-value regression.

The two main configurations are approximately:

| configuration | width | layers | parameters |
|---|---:|---:|---:|
| GAVN-3M | 192 | 8 | 4.13M |
| GAVN-5M | 224 | 8 | 5.30M |

### Repaired 5M control

`scripts/train_student.py` is the existing FEN/action-ID student, repaired for
controlled comparison:

- removed the accidental second `log_softmax`;
- added a `--sl-repo` argument for Kaggle paths;
- added complete checkpoint state: parameters, Adam moments, EMA, RNG, and
  configuration;
- added periodic HF uploads, HF resume, and failure-status uploads;
- fixed `--max-records` so smoke tests slice teacher labels correctly.

This model is a control, not the proposed final architecture.

### Evaluation and persistence

- `scripts/eval_gavn.py` evaluates GAVN on MATE and the official puzzle CSV.
- `scripts/kaggle_checkpoint.py` handles complete HF checkpoint upload/download.
- `scripts/build_student_train_set.py` can convert an official action-value bag
  into the compact NumPy training format and now accepts a mounted source repo.
- Every long training run must use a unique HF prefix and periodic uploads.

## Data requirements

The final experiment must use the full ChessBench training distribution. Do
not use the old `data/test/action_value_data.bag` derivative for final claims.

Decision record (2026-08-31): teacher = released **9M** checkpoint
(`9M/6400000/params_ema`, dim 256 / layers 8 / heads 8), dataset = **8 train
shards** (~134M rows, ~45GB processed). The 270M was the original runbook
choice; the user switched to the 9M teacher (cheaper labeling, directly
targets the "beat 9M" criterion). 8 shards are processed across multiple
resumable Kaggle sessions (Kaggle kernels die at ~12h; progress is saved to HF
after every shard).

Every training kernel should have a Kaggle Dataset mounted at:

```text
/kaggle/input/chessbench-full/
```

It must contain:

```text
train_set.npz       # full training positions/actions/win probabilities
teacher_logp.npy    # 9M teacher labels, shape [N, 128], fp16
```

`train_set.npz` must contain:

```text
tokens    uint8/int array, shape [N, 77]
actions   integer array, shape [N]
winprob   float array, shape [N]
```

`teacher_logp.npy` must contain normalized log-probabilities over 128 return
buckets, with the same number of rows as `tokens`.

### Creating the full-data Dataset (multi-session)

The full ChessBench train bag is ~2.9TB (2,148 shards); the dataset target is 8
shards (~45GB), built with a resumable pipeline because one Kaggle session
cannot finish it:

1. Push `notebooks/05_kaggle_build_full_data.ipynb` to a Kaggle kernel
   (account 1). It installs the era JAX stack, downloads the 9M teacher, and
   runs `scripts/build_full_dataset.py`.
2. `build_full_dataset.py` processes shards one at a time: download raw shard
   from GCS -> parse (tokens/actions/winprob) -> 9M teacher label (fp16) ->
   upload shard artifacts to HF (`vedangfake/chess-slm-benchmark` under
   `chessbench-full-build/shard-XXXXX/`) -> update the manifest (uploaded after
   every shard). A killed kernel resumes from the manifest and loses at most
   one shard.
3. Rerun the same notebook (or just the production cell) until all 8 shards
   are done. Progress check: the manifest at
   `chessbench-full-build/manifest.json` on HF.
4. Run `notebooks/06_kaggle_assemble_publish.ipynb` once: it downloads all 8
   shard pieces, merges them via memmap into `train_set.npz` +
   `teacher_logp.npy` (streamed, no full-RAM copies), validates, and publishes
   the Kaggle Dataset `chessbench-full` (public, so all three accounts can
   mount it).

The full bag and teacher labels are intentionally not committed to GitHub.
They are too large for a normal repository and should be stored as Kaggle/HF
Datasets. GitHub contains the reproducible code and experiment definitions.

## Kaggle setup

Create a Kaggle Secret named:

```text
HF_WRITE_TOKEN
```

The token needs write access to:

```text
vedangfake/chess-slm-benchmark
```

Note: there is no public API to create or attach Kaggle secrets (UI only:
Notebook Editor > Add-ons > Secrets). The pushed launch variants therefore
embed the token as an environment fallback constant (kernels are private).
Rotate the token if any kernel is ever published; never commit the token to
GitHub.

In a fresh Kaggle kernel, run:

```python
!git clone https://github.com/Vedang-P/chess-slm-benchmark.git \
    /kaggle/working/chess-slm-benchmark
%cd /kaggle/working/chess-slm-benchmark
!git pull --ff-only
```

Then run the cells in:

```text
notebooks/00_kaggle_setup_audit.ipynb
```

The setup notebook installs dependencies, clones the official
`searchless_chess` source tree, checks the GPU, validates data shapes, and
checks teacher log-probability normalization.

Run the setup notebook once in each kernel. Do not run two training processes
inside one kernel; use separate kernels so each process receives the GPU.

## Training notebooks

### Repaired 5M control

Run:

```text
notebooks/01_kaggle_baseline_5m.ipynb
```

This first runs a 20-step smoke test. If it passes, it launches the full 5M
control. The important variables are:

```python
RUN_ID = 'account1-baseline-5m-seed0'
STEPS = 30000
RESUME = True
```

Use a different `RUN_ID` for every independent run. With `RESUME = True`, the
trainer downloads the latest complete checkpoint under that HF prefix.

The final EMA checkpoint will be inside a local directory like:

```text
/kaggle/working/account1-baseline-5m-seed0/checkpoint-30000/
```

### GAVN model

Run:

```text
notebooks/02_kaggle_train_gavn.ipynb
```

The main configuration variables are:

```python
RUN_ID = 'account1-gavn-3m-seed0'
DIM = 192
LAYERS = 8
HEADS = 8
SEED = 0
STEPS = 30000
RESUME = True
```

Use `DIM = 192` for the approximately 4.13M model and `DIM = 224` for the
approximately 5.30M model. Do not change multiple design variables in the same
comparison without recording the change in the experiment table.

## Parallel schedule

There should be one training kernel per GPU. Two kernels may run concurrently
on one Kaggle account. With three accounts, six jobs can run concurrently:

| slot | `RUN_ID` | purpose |
|---|---|---|
| account 1 / GPU 1 | `account1-baseline-5m-seed0` | repaired FEN control |
| account 1 / GPU 2 | `account1-gavn-3m-seed0` | 3M geometry candidate |
| account 2 / GPU 1 | `account2-gavn-5m-seed0` | 5M geometry candidate |
| account 2 / GPU 2 | `account2-gavn-3m-seed1` | seed/variance check |
| account 3 / GPU 1 | `account3-gavn-5m-geometry` | geometry ablation |
| account 3 / GPU 2 | `account3-gavn-5m-loss` | loss ablation |

Each job must have a different `RUN_ID` and `--hf-run`. Never let two kernels
upload to the same HF prefix.

Recommended first wave:

1. baseline 5M, seed 0;
2. GAVN-3M, seed 0;
3. GAVN-5M, seed 0;
4. GAVN-3M, seed 1;
5. one fixed-geometry GAVN ablation;
6. one loss-weight ablation.

Do not select a final model from the frozen tests during this wave. Use a
development split from the training distribution for decisions.

## Resume and failure handling

Training checkpoints are saved locally every `--ckpt-every` steps and uploaded
periodically, normally every 1,800 seconds.

The HF layout is:

```text
vedangfake/chess-slm-benchmark/
  account1-gavn-3m-seed0/
    checkpoint-2000/
      state.pt or state.npz
      config.json
      metrics.json
      RNG/state files
    checkpoint-4000/
    run-status.txt
```

If a Kaggle kernel stops:

1. Start a fresh kernel.
2. Pull the current GitHub repository.
3. Attach the same `chessbench-full` Dataset.
4. Use the same `RUN_ID`.
5. Leave `RESUME = True`.
6. Rerun the production cell.

The trainer restores the latest complete checkpoint, including optimizer
state and RNG state. If HF upload fails, the local checkpoint is retained and
the log reports the failure; do not silently start a different experiment.

## Evaluation

After training, attach or copy the selected checkpoint and run:

```text
notebooks/03_kaggle_eval_frontier.ipynb
```

Set `CKPT` to the exact checkpoint directory containing `state.pt` and
`config.json`, for example:

```python
CKPT = Path('/kaggle/input/chess-checkpoints/account1-gavn-3m-seed0/checkpoint-30000')
```

The evaluator runs:

- all available MATE JSON files under `data/positions/`;
- the official `searchless_chess/data/puzzles.csv` protocol;
- full solution-sequence accuracy, where every required move must be correct.

Preserve the complete stdout, checkpoint path, Git commit, dataset version,
GPU type, seed, and command line. The frozen MATE and puzzle sets are for final
reporting, not hyperparameter selection.

## Current scientific guardrails

- The released Ruoss baselines are already measured: 9M is 98.725% combined
  MATE and 86.13% on the local official puzzle harness; 136M and 270M are both
  approximately 99.4% on the combined MATE benchmark.
- The `both` and `tactic` MATE subsets share the same underlying FENs, so the
  four subsets are not four independent test pools.
- The local 9M puzzle score differs from the published 9M number, so the exact
  evaluator/checkpoint convention must be recorded and investigated.
- A sub-9M model matching the 9M baseline is the first success criterion.
  Matching 136M/270M below 9M is a separate stretch target.
- No result is a research claim until it is reproduced from the full training
  distribution with frozen evaluation and resumable artifacts.

## Useful direct commands

From `/kaggle/working/chess-slm-benchmark`:

```bash
# GAVN smoke test is run by notebook 02; equivalent CLI shape:
python scripts/train_gavn.py \
  --data /kaggle/input/chessbench-full/train_set.npz \
  --teacher /kaggle/input/chessbench-full/teacher_logp.npy \
  --sl-repo /kaggle/working/searchless_chess \
  --outdir /kaggle/working/gavn-3m \
  --dim 192 --layers 8 --heads 8 \
  --hf-run account1-gavn-3m-seed0 \
  --resume-from-hf

# Evaluate a GAVN checkpoint:
python scripts/eval_gavn.py \
  --checkpoint /kaggle/working/gavn-3m/checkpoint-30000 \
  --sl-repo /kaggle/working/searchless_chess \
  --eval data/positions/mate-selection-test-noexplain.json \
  --puzzles /kaggle/working/searchless_chess/data/puzzles.csv
```

The notebooks are the preferred entry point because they expose the paths and
configuration cells visibly and preserve the experiment context in Kaggle.
