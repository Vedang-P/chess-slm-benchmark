# Live proxy (Cloudflare Worker)

GitHub's raw CDN caches files for **5 minutes** (`cache-control: max-age=300`) — useless for a
real-time board feed. This worker proxies the GitHub contents API (always fresh) with a short
server-side cache:

| Endpoint | Returns | Edge cache |
|---|---|---|
| `/live` | monitor/live.json (per-sample telemetry) | 3 s |
| `/state` | monitor/state.json (sweep progress) | 10 s |
| `/history` | monitor/history.jsonl | 10 s |

## Deploy (one-time, ~2 min)

```bash
npx wrangler login                 # browser auth, free account
npx wrangler deploy                # from this directory
npx wrangler secret put GH_TOKEN   # your GitHub PAT (repo scope) — kept
                                   # server-side, never sent to browsers
```

The worker calls `api.github.com` (rate limit 5000/hr with the token — fine at 3 s polling).

## Point the dashboard at it

Set `WORKER_BASE` in `frontend/config.js`:

```js
WORKER_BASE: "https://chess-live.YOUR_SUBDOMAIN.workers.dev",
```

`wrangler deploy` prints the URL. Without the worker, the dashboard falls back to the raw
GitHub URLs — functional, but the live board lags by up to 5 minutes.
