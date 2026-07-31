# Terminal-driven GPU options (2026-08)

How to run the chess sweep with a GPU you drive from this terminal — and what
happens to live monitoring on each.

## Modal (recommended — terminal + live logs + our monitor)

- Free tier: **$30 of credits/month** (T4 is ~$0.19/hr → ~150 GPU-hours/month,
  far beyond the ~33h this sweep needs).
- `pip install modal && modal token new`, then:
  ```bash
  modal run modal_app.py                     # full sweep on a T4
  modal run modal_app.py --check --smoke     # fast local-ish validation
  ```
- Live log streaming straight to this terminal; the `--monitor` flag pushes
  progress to the public live repo, so the dashboard works identically to
  Kaggle. Results live in a Modal volume (`chess-results`) and are backed up
  per-cell to the live repo.
- Optional: create a Modal secret `hf_token` for the gated Gemma models.
- `modal volume get chess-results chess/comparison_table.csv` to fetch results.

## Kaggle (what we use now)

- **No terminal execution.** The CLI (`kaggle kernels push/status`) can launch
  a notebook and poll status, but cannot stream output mid-run.
- Live monitoring is handled out-of-band: the notebook's `--monitor` pushes
  state to the public live repo; the dashboard renders it.
- Quota: ~30h/week of T4 GPU.

## Others

| Service | Free tier | Terminal? | Notes |
|---|---|---|---|
| Lightning AI | limited free GPU | `lightning run` | queues on free tier |
| RunPod | one-time credits | SSH pods | good, but one-shot credits |
| Google Colab | T4, heavy quota | no | browser only; session dies idle |
| Oracle Cloud Free | ARM CPU, no GPU | SSH | always-on CPU fallback for smoke tests |
| GitHub Actions | no free GPU runners | — | paid runners only |
