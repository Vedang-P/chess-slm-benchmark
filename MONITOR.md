# Mission Control (Cloudflare Worker monitor)

Live dashboard for the whole project, phone-friendly:
**https://chess-slm-monitor.vedangpandeyy.workers.dev**

## What it shows

- **Stages** — 320k training → 320k frozen eval → 1B corpus labeling → 1B
  continuation (→1.62M steps) → 1B frozen eval, each with a progress bar.
- **Loss curves** — train and dev loss per checkpoint of `ccgavn-5m-seed0`
  (all checkpoints, from the first 5k step to the latest upload).
- **Frozen evals** — bar chart + table of every `eval-summary.json` ever
  archived on HF (MATE % / puzzles %), with the Ruoss 9M teacher and legacy
  GAVN reference rows.
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
- `scripts/push_monitor.py` (run by `scripts/supervise_all.py` every 15 min)
  assembles the heavy sections — full training curve, eval summaries, corpus
  progress, PGN games, kernel statuses, quotas — and POSTs them to the Worker.
- Secrets in the Worker: `HF_TOKEN` (read HF), `INGEST_KEY` (in `.env` as
  `MONITOR_INGEST_KEY`, along with `MONITOR_URL`).

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
