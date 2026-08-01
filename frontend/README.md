# Chess Benchmark — Live Monitor Dashboard

Static dashboard for the anti-goal chess benchmark sweep running on Kaggle.
No build step — plain HTML/CSS/JS + Chart.js (CDN). Deploy to Cloudflare Pages
as-is.

## How it gets data

The Kaggle sweep pushes `monitor/state.json` + `monitor/history.jsonl` to the
**public** repo `Vedang-P/chess-bench-live` (branch `main`) via the GitHub
contents API, every ~2 minutes (the `--monitor` flag of `scripts/run_suite.py`).
The dashboard polls those files over `raw.githubusercontent.com` (CORS `*`).

> The dashboard repo is public on purpose: `raw.githubusercontent.com`
> refuses private repos, and a browser needs no auth. It contains ONLY
> `monitor/*.json` — never code or results.

## Files

- `index.html` — page skeleton
- `styles.css` — dark theme, cards, responsive grid
- `app.js` — polling, KPI rendering, Chart.js charts, live table
- `config.js` — `STATE_URL`, `HISTORY_URL`, `REFRESH_S` (edit for your repo)
- `vercel.json` / `wrangler.toml` — deploy configs

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
that matches the `state.json` schema (see `scripts/run_suite.py` → `Monitor._state`)
and the dashboard renders it. The schema is the contract — keep it stable
across projects.
