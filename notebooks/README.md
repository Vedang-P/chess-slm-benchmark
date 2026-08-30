# Kaggle runbook

From each Kaggle kernel, pull the repository first:

```bash
!git clone https://github.com/Vedang-P/chess-slm-benchmark.git /kaggle/working/chess-slm-benchmark
!git -C /kaggle/working/chess-slm-benchmark pull --ff-only
```

Then run `00_kaggle_setup_audit.ipynb` once per kernel, followed by exactly one
training notebook per GPU. Two kernels can run concurrently on each account;
with three accounts, launch six jobs in parallel:

| slot | run | purpose |
|---|---|---|
| A1 | `baseline-5m-seed0` | repaired compatibility control |
| A2 | `gavn-3m-seed0` | square-token 3M candidate |
| B1 | `gavn-5m-seed0` | square-token 5M candidate |
| B2 | `gavn-3m-seed1` | seed check / variance |
| C1 | `gavn-5m-geom-seed0` | geometry ablation |
| C2 | `gavn-5m-loss-seed0` | distillation-loss ablation |

Every run must have a unique `RUN_ID` and therefore a unique `--hf-run` prefix.
Do not upload multiple jobs into the same prefix. Set the `HF_WRITE_TOKEN`
Kaggle secret before running production cells.

Attach one Kaggle Dataset called `chessbench-full` to every training kernel. It
must contain the full training export:

```text
train_set.npz       # full ChessBench tokens [N,77], actions [N], winprob [N]
teacher_logp.npy    # 270M teacher log-probabilities [N,128]
```

Notebook 04 creates this export from the official training bag. Publish its
`/kaggle/working/chessbench-full` directory as a Kaggle Dataset, then attach it
as `chessbench-full`. Do not use the old test-bag derivative for final runs.

Suggested sequence:

1. Run setup/audit in all six kernels.
2. Run both smoke tests.
3. Launch the six production configurations above.
4. Select using a held-out development split and then run the frozen evaluator.
5. Keep all logs and HF checkpoint prefixes; do not delete failed runs.
