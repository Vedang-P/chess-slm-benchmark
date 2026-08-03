// Dashboard configuration — edit these for your deployment.
const CONFIG = {
  // Cloudflare Worker proxy (recommended): e.g. "https://chess-live.USERNAME.workers.dev"
  // Serves fresh data with a 3s cache. Leave empty to fall back to the raw
  // GitHub URLs below (those are cached ~5 minutes by GitHub's CDN).
  WORKER_BASE: "https://chess-live.vedangpandeyy.workers.dev",
  // Raw GitHub URLs (fallback channel; ~5 min edge cache).
  STATE_URL: "https://raw.githubusercontent.com/Vedang-P/chess-bench-live/main/monitor/state.json",
  HISTORY_URL: "https://raw.githubusercontent.com/Vedang-P/chess-bench-live/main/monitor/history.jsonl",
  LIVE_URL: "https://raw.githubusercontent.com/Vedang-P/chess-bench-live/main/monitor/live.json",
  // Auto-refresh intervals in seconds (0 disables polling).
  REFRESH_S: 15,
  LIVE_REFRESH_S: 2,
  // Fallback branding / links.
  REPO_URL: "https://github.com/Vedang-P/chess-slm-benchmark",
  THEME: "dark",
};
