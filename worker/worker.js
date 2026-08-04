// ChessBench live proxy — Cloudflare Worker.
//
// GitHub's raw CDN caches files for 5 minutes (max-age=300), which is useless
// for a real-time feed. The GitHub API is always fresh, so this worker
// proxies it with a SHORT server-side cache and serves CORS-enabled JSON.
//
// Endpoints: /live  /state  /history
//
// Deploy:
//   npx wrangler login
//   npx wrangler deploy
//   npx wrangler secret put GH_TOKEN   # your GitHub PAT (repo scope) — kept
//                                      # server-side, never exposed to clients
//   (without the secret, the GitHub API allows only 60 req/hr/IP — set the
//    worker cache TTL to >= 60s or add the secret for true live updates)

const REPO = "Vedang-P/chess-bench-live";
const FILES = {
  "/live": "monitor/live.json",
  "/state": "monitor/state.json",
  "/history": "monitor/history.jsonl",
};

function base64ToBytes(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

async function fetchContent(path, token) {
  const headers = { "User-Agent": "chess-live", Accept: "application/vnd.github+json" };
  if (token) headers.Authorization = `Bearer ${token}`;
  const res = await fetch(`https://api.github.com/repos/${REPO}/contents/${path}`, { headers });
  if (!res.ok) return null;
  const data = await res.json();
  if (data.encoding !== "base64" || !data.content) return null;
  return { body: base64ToBytes(data.content), etag: data.sha };
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const file = FILES[url.pathname];
    if (!file) {
      return new Response("chess-live proxy: /live  /state  /history", {
        headers: { "Access-Control-Allow-Origin": "*" },
      });
    }

    // short server-side cache: /live refreshes every 3s, the rest every 10s
    const cacheKey = new Request(`https://cache.local${url.pathname}`, { method: "GET" });
    const cache = caches.default;
    let res = await cache.match(cacheKey);
    if (res) {
      res = new Response(res.body, res);
      res.headers.set("Access-Control-Allow-Origin", "*");
      return res;
    }

    const content = await fetchContent(file, env.GH_TOKEN);
    if (!content) {
      return new Response("monitor file not found", {
        status: 502,
        headers: { "Access-Control-Allow-Origin": "*" },
      });
    }

    // 3s for /live (the runner republishes at most every ~2s, so a 1s TTL
    // just burned GitHub API quota: 1 req/s = 3600/hr against a 5000/hr
    // authenticated budget, and 60/hr unauthenticated)
    const ttl = file === "monitor/live.json" ? 3 : 15;
    res = new Response(content.body, {
      headers: {
        "Content-Type": "application/json; charset=utf-8",
        "Access-Control-Allow-Origin": "*",
        "Cache-Control": `public, max-age=${ttl}`,
        ETag: content.etag,
      },
    });
    ctx.waitUntil(cache.put(cacheKey, res.clone()));
    return res;
  },
};
