# Mission Control (Cloudflare Worker monitor)

Live dashboard for the whole project, phone-friendly:
**https://chess-slm-monitor.vedangpandeyy.workers.dev**

## What it shows

- **Stages** — 320k training → 320k archived eval → 1B corpus labeling →
  1B continuation (→1.62M steps) → archived eval → corrected-v2 2B
  continuation (disarmed until explicitly armed) → 3.90M final, each with a
  progress bar.
- **Loss curves** — train and dev loss per checkpoint across the legacy
  `ccgavn-5m-seed0` prefix and the corrected `ccgavn-5m-seed0-v2` prefix
  (steps continue across both; legacy points are never rewritten).
- **Monitoring + final evals** — bar chart + table of every
  `eval-summary.json` on HF (MATE % / puzzles %), with the Ruoss 9M teacher
  and legacy GAVN reference rows. Milestone evals are development
  diagnostics, not a holdout: the MATE/puzzle sets were scored repeatedly
  during training, and only the one-shot `*-final` run is the protocol result
  (with that exposure disclosed).
- **1B corpus** — labeled new rows vs the 920M target, shards done/planned.
- **Runs** — kernel statuses per account, GPU quota left per account, and each
  run's `run-status.txt` from HF.
- **Eval games** — archived CC-GAVN vs Stockfish PGNs, replayable move by
  move with autoplay (FENs are precomputed, no client-side chess lib).

## Architecture

- `monitor/src/index.js` — Cloudflare Worker (`chess-slm-monitor`):
  - serves the dashboard at `/`,
  - `GET /api/snapshot` — the KV snapshot (stages recomputed at serve time),
  - `POST /api/ingest` — merge endpoint used by the local supervisor
    (header `x-ingest-key`),
  - a `*/10` cron does a light refresh (HF file tree, corpus counts, eval
    summaries, a few run statuses) so the page stays fresh even when the Mac
    is off. Workers have a hard 50-subrequest limit, which is why the heavy
    data is pushed from outside (below).
- KV namespace `SNAPSHOT` (`16a8423fd12c40c8a3990bda9c911798`) holds one JSON
  snapshot.
- **Full cloud automation, no local machine involved:** the Worker's `*/10`
  cron also sends a `repository_dispatch` tick to GitHub (throttled to ~10 min
  via KV). The `pipeline-tick` GitHub Action then runs the keep-alive checks
  (`ensure_build_2b.py`, `watch_1b.py`, `watch_2b.py`, `ensure_eval_320k.py`,
  `ensure_eval_preview.py`) and `scripts/push_monitor.py`, which assembles the
  heavy sections — full
  training curve, eval summaries, corpus progress, PGN games, kernel
  statuses, quotas — and POSTs them to `/api/ingest`. A 30-minute cron in the
  same workflow acts as a fallback; if both stall, the page shows the last
  snapshot with its timestamp.
- Secrets in the Worker: `HF_TOKEN` (read HF), `INGEST_KEY`, `GH_TOKEN`
  (dispatch the Actions tick). GitHub repo secrets used by the workflow:
  `HF_WRITE_TOKEN`, `MONITOR_INGEST_KEY`, and the five accounts' Kaggle
  credentials. `MONITOR_URL` is set in the workflow env.

## Updating the Worker

```bash
cd monitor
npx wrangler deploy
```

## Notes

- Nothing in the training recipe is affected; the monitor is read-only on HF
  and Kaggle (plus quota/status reads from the local supervisor).
- If both the Mac is off and the cron stalls, the page shows the last
  snapshot with its `updated_at` timestamp.
