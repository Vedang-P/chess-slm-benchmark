# Chess Benchmark — Live Monitor Dashboard

Static dashboard for the chess representation study (gemma + deepseek-v4-flash).
No build step — plain HTML/CSS/JS + Chart.js (CDN). Deploy to Cloudflare Pages
as-is.

## How it gets data

The runner pushes `monitor/state.json`, `monitor/history.jsonl` and
`monitor/live.json` to the **public** repo `Vedang-P/chess-bench-live` (branch
`main`) via the GitHub contents API:

- `scripts/run_mate_eval.py` — MATE move-selection eval (Kaggle workers)
- `scripts/run_mate_eval.py --live-push` — the MATE move-selection run

The dashboard reads them through the Cloudflare Worker proxy (3s cache) and
falls back to `raw.githubusercontent.com` (5 min CDN cache) if the worker is
unreachable.

## Two run kinds, two dashboards

`state.json` carries `run_kind`, and the page renders a different scoreboard,
chart set and table for each. This matters: a MATE selection run has no notion
of move legality and a sweep has no A/B split, so a single fixed layout always
showed cards that could never have a value.

| `run_kind` | scoreboard | charts |
|---|---|---|
| `mate-selection` | positions, accuracy vs expert, answer rate, choice bias, accuracy by expert label, throughput/ETA, tokens, api errors | accuracy over the run, choice distribution, outcome breakdown, B-preference drift |
| `sweep` | cells completed, legal rate, mate-in-1/2 strict, stockfish top-1 | parsing/legality, tactics pipeline, move strength, progress over time |

Snapshots without `run_kind` are treated as a sweep unless `mode == "mate"`.

**`api_error` is not a model failure.** Transport failures (gateway 5xx,
timeouts) are counted separately and excluded from every rate; the table shows
`scored / attempted` whenever the two differ.

> The dashboard repo is public on purpose: `raw.githubusercontent.com`
> refuses private repos, and a browser needs no auth. It contains ONLY
> `monitor/*.json` — never code or results.

## Files

- `index.html` — page skeleton
- `styles.css` — dark theme, cards, responsive grid
- `app.js` — polling, run-kind-aware scoreboard/charts/table, live board
- `config.js` — `STATE_URL`, `HISTORY_URL`, `REFRESH_S` (edit for your repo)
- `pieces.js` — inline cburnett SVG pieces for the live board
- `wrangler.toml` — Cloudflare Pages config

## Deploy

### Cloudflare Pages
```bash
npx wrangler pages deploy frontend/ --project-name chess-bench-live
```
Or: Cloudflare dashboard → Pages → Create → Direct upload → drag the `frontend/` folder.

## Local dev

```bash
cd frontend && python3 -m http.server 8080
# open http://localhost:8080
```

## Reusing it later

Everything data-related is decoupled: point `CONFIG.STATE_URL` at any JSON
that matches the `state.json` schema and the dashboard renders it. The two
producers are the contract:

- sweep — `scripts/run_mate_eval.py` → `Monitor._state`
- MATE — `scripts/run_mate_eval.py` → `_state_payload` / `_mate_metrics`

Keep `run_kind` set; without it a new run kind silently renders through the
wrong layout.
