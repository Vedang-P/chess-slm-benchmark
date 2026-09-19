import { Chess } from "chess.js";

const HF = "https://huggingface.co";
const REPO = "vedangfake/chess-slm-benchmark";
const RAW = "https://raw.githubusercontent.com/Vedang-P/chess-slm-benchmark/main";
const RUN = "ccgavn-5m-seed0";
const FINAL_STEP = 1_620_000;
const STAGE1 = 320_000;
const CORPUS_TARGET = 920_000_000;
const LIVE_RUNS = ["ccgavn-5m-seed0", "chessbench-full-build", "ccgavn-1b", "puzzle-curriculum"];
const REFERENCE = [
  { name: "Ruoss 9M (teacher)", mate: 98.72, puzzles: 86.13, ref: true },
  { name: "legacy GAVN 5.3M", mate: 85.35, puzzles: 43.83, ref: true },
];

async function hfTree(env, path = "") {
  const out = [];
  let url = `${HF}/api/datasets/${REPO}/tree/main${path ? `/${path}` : ""}?recursive=1`;
  for (let page = 0; page < 12 && url; page++) {
    const r = await fetch(url, { headers: { Authorization: `Bearer ${env.HF_TOKEN}` } });
    if (!r.ok) break;
    const items = await r.json();
    if (Array.isArray(items)) out.push(...items);
    const link = r.headers.get("link") || "";
    const m = link.match(/<([^>]+)>;\s*rel="next"/);
    url = m ? m[1] : null;
  }
  return out;
}

async function hfFile(env, path) {
  const r = await fetch(`${HF}/datasets/${REPO}/resolve/main/${path}`, {
    headers: { Authorization: `Bearer ${env.HF_TOKEN}` },
  });
  if (!r.ok) return null;
  return await r.text();
}

async function hfJson(env, path) {
  const t = await hfFile(env, path);
  if (t === null) return null;
  try { return JSON.parse(t); } catch { return null; }
}

function parseGames(pgnText, limit = 8) {
  const games = [];
  for (const chunk of pgnText.split(/\n\s*\n(?=\[Event )/)) {
    if (games.length >= limit) break;
    const headers = {};
    for (const line of chunk.split("\n")) {
      const m = line.match(/^\[(\w+)\s+"(.*)"\]\s*$/);
      if (m) headers[m[1]] = m[2];
    }
    if (!headers.Event) continue;
    try {
      const replay = new Chess();
      replay.loadPgn(chunk);
      const walk = new Chess();
      const moves = [];
      for (const mv of replay.history({ verbose: true })) {
        walk.move(mv.san);
        moves.push({ san: mv.san, fen: walk.fen(), from: mv.from, to: mv.to });
      }
      if (moves.length) {
        games.push({ event: headers.Event, white: headers.White || "?", black: headers.Black || "?",
                     result: headers.Result || "*", moves });
      }
    } catch { continue; }
  }
  return games;
}

function computeStages(snap) {
  const curve = snap.curve || [];
  const lastStep = curve.length ? curve[curve.length - 1].step : 0;
  const corpus = snap.corpus || {};
  const s = {};
  s.train320 = lastStep >= STAGE1 ? 100 : Math.min(100, (lastStep / STAGE1) * 100);
  s.eval320 = snap.eval320_done ? 100 : (lastStep >= STAGE1 ? 50 : 0);
  s.corpus1b = Math.min(100, ((corpus.labeled_rows || 0) / CORPUS_TARGET) * 100);
  s.train1b = lastStep >= FINAL_STEP ? 100
    : lastStep >= STAGE1 ? Math.max(0, Math.min(100, ((lastStep - STAGE1) / (FINAL_STEP - STAGE1)) * 100)) : 0;
  s.eval1b = snap.eval1b_done ? 100 : 0;
  return s;
}

function appendNotices(prev, next) {
  const notices = prev.notices || [];
  const ts = Date.now();
  const push = (text) => notices.unshift({ t: ts, text });
  if (!notices.length) {
    // seed the feed once so a fresh dashboard is never empty
    const c = next.curve || [];
    const last = c.length ? c[c.length - 1] : null;
    if (last) push(`training at step ${last.step.toLocaleString()} · train ${last.train.toFixed(4)} · dev ${(last.dev ?? 0).toFixed(4)}`);
    for (const [k, v] of Object.entries(next.evals || {})) {
      if (v.mate && v.mate[0] && v.puzzles && v.puzzles[0]) {
        push(`eval complete — ${k.replace(/^eval-results\//, "").replace(/\/eval-summary\.json$/, "")}: MATE ${v.mate[0].pct}% · puzzles ${v.puzzles[0].pct}%`);
      }
    }
    const cp = next.corpus || {};
    if (cp.shards_planned) push(`1B corpus build armed: ${cp.shards_done || 0}/${cp.shards_planned} shards labeled`);
  }
  const ps = prev.stages || {}, ns = next.stages || {};
  const pc = prev.corpus || {}, nc = next.corpus || {};
  const pe = prev.evals || {}, ne = next.evals || {};
  for (const [k, v] of Object.entries(ne)) {
    if (!(k in pe) && v.mate && v.mate[0] && v.puzzles && v.puzzles[0]) {
      const name = k.replace(/^eval-results\//, "").replace(/\/eval-summary\.json$/, "");
      push(`eval complete — ${name}: MATE ${v.mate[0].pct}% · puzzles ${v.puzzles[0].pct}%`);
    }
  }
  if ((ps.train320 || 0) < 100 && (ns.train320 || 0) >= 100) push("CC-GAVN reached 320k steps");
  if ((ps.eval320 || 0) < 100 && (ns.eval320 || 0) >= 100) push("320k frozen evaluation archived");
  for (const mark of [100e6, 250e6, 500e6, 750e6, 920e6]) {
    if ((pc.labeled_rows || 0) < mark && (nc.labeled_rows || 0) >= mark)
      push(`corpus milestone: ${(mark / 1e6).toFixed(0)}M new rows labeled`);
  }
  for (const pct of [25, 50, 75, 100]) {
    const a = ps.train1b || 0, b = ns.train1b || 0;
    if (a < pct && b >= pct && pct < 100) push(`1B continuation reached ${pct}%`);
    if (pct === 100 && a < 100 && b >= 100) push("1B continuation reached 1.62M steps");
  }
  next.notices = notices.slice(0, 40);
}

async function dispatchTick(env) {
  if (!env.GH_TOKEN) return;
  const last = Number((await env.SNAPSHOT.get("last_tick")) || 0);
  if (Date.now() - last < 8 * 60 * 1000) return;
  try {
    const r = await fetch("https://api.github.com/repos/Vedang-P/chess-slm-benchmark/dispatches", {
      method: "POST",
      headers: { Authorization: `Bearer ${env.GH_TOKEN}`, Accept: "application/vnd.github+json",
                 "User-Agent": "chess-slm-monitor" },
      body: JSON.stringify({ event_type: "tick" }),
    });
    if (r.ok || r.status === 204) await env.SNAPSHOT.put("last_tick", String(Date.now()));
  } catch { /* retry next cron */ }
}

async function refresh(env) {
  const prev = (await env.SNAPSHOT.get("snapshot", "json")) || {};
  const snap = { ...prev, errors: [] };
  snap.curve = prev.curve || [];
  snap.evals = prev.evals || {};
  snap.runs = {};
  snap.games = prev.games || [];
  snap.kernels = prev.kernels || [];
  snap.quota = prev.quota || {};
  snap.notices = prev.notices || [];

  let files = [];
  try { files = await hfTree(env); } catch (e) { snap.errors.push(`hf: ${e}`); }
  const paths = new Set(files.map((f) => f.path));

  const known = new Set(snap.curve.map((p) => p.step));
  const metrics = files.map((f) => f.path)
    .filter((p) => /^ccgavn-5m-seed0\/checkpoint-\d+\/metrics\.json$/.test(p))
    .map((p) => ({ p, step: parseInt(p.match(/checkpoint-(\d+)/)[1], 10) }))
    .filter((x) => !known.has(x.step)).sort((a, b) => a.step - b.step);
  for (const { p, step } of metrics.slice(0, 10)) {
    const m = await hfJson(env, p);
    if (m && typeof m.train_loss === "number") snap.curve.push({ step, train: m.train_loss, dev: m.dev_loss ?? null });
  }
  snap.curve.sort((a, b) => a.step - b.step);

  const evalPaths = [...paths].filter((p) => /^eval-results\/.*\/eval-summary\.json$/.test(p));
  for (const p of evalPaths) {
    if (snap.evals[p]) continue;
    const s = await hfJson(env, p);
    if (s) snap.evals[p] = { ...s, fetched_at: new Date().toISOString() };
  }
  snap.eval320_done = evalPaths.some((p) => p.includes("320k"));
  snap.eval1b_done = evalPaths.some((p) => /1b/i.test(p));

  for (const p of [...paths].filter((x) => /^[^/]+\/run-status\.txt$/.test(x))) {
    const top = p.split("/")[0];
    if (!LIVE_RUNS.includes(top)) continue;
    const t = await hfFile(env, p);
    if (t !== null) snap.runs[p] = t.slice(0, 300);
  }

  let rowsMap = {};
  try {
    const r = await fetch(`${RAW}/kernels/build-2b/shard_rows.json`);
    if (r.ok) rowsMap = await r.json();
  } catch { /* estimates only */ }
  const planned = Object.keys(rowsMap);
  const done = planned.filter((s) => paths.has(`chessbench-full-build/shard-${s}/teacher_logp.npy`));
  snap.corpus = {
    target_rows: CORPUS_TARGET,
    labeled_rows: done.reduce((a, s) => a + (rowsMap[s] || 0), 0),
    shards_planned: planned.length || 108,
    shards_done: done.length,
    original_rows: 94_277_038,
  };

  try {
    const [slicesRes, assignRes] = await Promise.all([
      fetch(`${RAW}/kernels/build-2b/shard_slices.json`),
      fetch(`${RAW}/kernels/build-2b/slices.json`),
    ]);
    if (slicesRes.ok && assignRes.ok && planned.length) {
      const sliceLists = await slicesRes.json();
      const assign = await assignRes.json();
      const ba = {};
      for (const [acct, idxs] of Object.entries(assign)) {
        let ps = 0, ds = 0, pr = 0, dr = 0;
        for (const i of idxs) for (const shard of (sliceLists[i] || [])) {
          ps += 1; pr += Number(rowsMap[shard] || 0);
          if (paths.has(`chessbench-full-build/shard-${shard}/teacher_logp.npy`)) {
            ds += 1; dr += Number(rowsMap[shard] || 0);
          }
        }
        ba[acct] = { planned_shards: ps, done_shards: ds, planned_rows: pr, done_rows: dr };
      }
      snap.build_accounts = ba;
    }
  } catch { /* keep previous build_accounts */ }

  if (!snap.games.length) {
    for (const p of [...paths].filter((x) => /^elo-results\/.*\/games-\d+\.pgn$/.test(x)).slice(0, 2)) {
      const t = await hfFile(env, p);
      if (t && snap.games.length < 8) snap.games = snap.games.concat(parseGames(t, 8 - snap.games.length));
      if (snap.games.length >= 8) break;
    }
  }

  snap.reference = REFERENCE;
  snap.stages = computeStages(snap);
  appendNotices(prev, snap);
  snap.updated_at = new Date().toISOString();
  await env.SNAPSHOT.put("snapshot", JSON.stringify(snap));
  return snap;
}

const MASK = { vedanggggg: "acct-1", vedangpandeyyy: "acct-2", softmaxsimp: "acct-3", samaltmannnn: "acct-4", shoumikmitra: "acct-5" };
function masked(o) {
  let s = JSON.stringify(o);
  for (const k in MASK) s = s.split(k).join(MASK[k]);
  return JSON.parse(s);
}

const SEC = {
  "x-content-type-options": "nosniff",
  "referrer-policy": "no-referrer",
  "x-frame-options": "DENY",
  "permissions-policy": "camera=(), microphone=(), geolocation=()",
  "content-security-policy": "default-src 'none'; script-src 'unsafe-inline' https://cdn.jsdelivr.net; style-src 'unsafe-inline' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
};

export default {
  async scheduled(event, env, ctx) {
    ctx.waitUntil(refresh(env));
    ctx.waitUntil(dispatchTick(env));
  },

  async fetch(req, env) {
    const url = new URL(req.url);
    if (url.pathname === "/api/snapshot") {
      const snap = (await env.SNAPSHOT.get("snapshot", "json")) || {};
      snap.stages = computeStages(snap);
      return Response.json(masked(snap), { headers: { ...SEC, "cache-control": "no-store" } });
    }
    if (url.pathname === "/api/ingest" && req.method === "POST") {
      if (req.headers.get("x-ingest-key") !== env.INGEST_KEY) return new Response("forbidden", { status: 403, headers: SEC });
      const body = await req.json().catch(() => null);
      if (!body) return new Response("bad json", { status: 400, headers: SEC });
      const prior = (await env.SNAPSHOT.get("snapshot", "json")) || {};
      const snap = { ...prior };
      for (const k of ["curve", "evals", "corpus", "games", "runs", "kernels", "quota", "live"]) {
        if (k in body) snap[k] = masked(body[k]);
      }
      snap.reference = REFERENCE;
      snap.ingested_at = new Date().toISOString();
      snap.stages = computeStages(snap);
      appendNotices(prior, snap);
      snap.updated_at = snap.ingested_at;
      await env.SNAPSHOT.put("snapshot", JSON.stringify(snap));
      return Response.json({ ok: true, stages: snap.stages }, { headers: SEC });
    }
    if (url.pathname === "/api/tick" || url.pathname === "/api/refresh") {
      if (url.searchParams.get("key") !== env.INGEST_KEY) return new Response("forbidden", { status: 403, headers: SEC });
      if (url.pathname === "/api/tick") {
        await env.SNAPSHOT.delete("last_tick");
        await dispatchTick(env);
        return Response.json({ ok: true }, { headers: SEC });
      }
      const s = await refresh(env);
      return Response.json({ ok: true, stages: s.stages }, { headers: SEC });
    }
    if (url.pathname.startsWith("/api/")) return new Response("not found", { status: 404, headers: SEC });
    return new Response(HTML, { headers: { ...SEC, "content-type": "text/html;charset=utf-8", "cache-control": "no-store" } });
  },
};

const HTML = `<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>chess-slm · mission control</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
:root[data-theme="dark"]{
  --bg:#111111; --surface:#161616; --surface-2:#1e1e1e; --surface-3:#2a2a2a;
  --border:#2a2a29; --ink:#ececea; --ink-2:#a3a3a0; --ink-3:#6f6f6c;
  --grid:#242423; --accent:#3987e5; --live:#d9463f; --ok:#3fa66a; --warn:#c9a227;
  --board-l:#f0d9b5; --board-d:#b58863; --radius:6px; --nav-h:48px;
  --font:"Inter",-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,Helvetica,Arial,sans-serif;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
  color-scheme:dark;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--font);font-size:13.5px;line-height:1.45;-webkit-font-smoothing:antialiased}
a{color:inherit;text-decoration:none}
button{font:inherit;color:inherit;background:none;border:0;padding:0;cursor:pointer}
input,select{font:inherit;color:inherit}
h1,h2,p{margin:0}
.hidden{display:none!important}
.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}
.muted{color:var(--ink-3)}
.dim{color:var(--ink-3)}
.wrap{max-width:1440px;margin:0 auto;padding:0 20px}
/* nav */
.nav{position:sticky;top:0;z-index:20;height:var(--nav-h);background:var(--bg);border-bottom:1px solid var(--border)}
.nav-inner{max-width:1440px;margin:0 auto;padding:0 20px;height:100%;display:flex;align-items:center;gap:22px}
.brand{font-weight:600;white-space:nowrap}
.tabs{display:flex;gap:2px;height:100%}
.tabs a{display:flex;align-items:center;gap:7px;padding:0 9px;color:var(--ink-2);position:relative;white-space:nowrap;cursor:pointer;font-size:13px}
.tabs a:hover{color:var(--ink)}
.tabs a.active{color:var(--ink)}
.tabs a.active::after{content:"";position:absolute;left:9px;right:9px;bottom:-1px;height:2px;background:var(--ink)}
.nav-right{margin-left:auto;display:flex;align-items:center;gap:18px;white-space:nowrap}
.nav-cell{display:flex;flex-direction:column;align-items:flex-end;line-height:1.15}
.nav-cell .k{font-size:10px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-3)}
.nav-cell .v{font-family:var(--mono);font-size:13px}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--ink-3);vertical-align:middle}
.dot.live{background:var(--live)}
@media(max-width:880px){.nav-right .nav-cell.opt{display:none}}
/* sections */
.section{margin:18px 0}
.section-title{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3);margin:22px 0 8px;font-weight:500}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:12px 14px;min-width:0}
.card-head{display:flex;align-items:baseline;justify-content:space-between;gap:12px;margin-bottom:6px}
.card-title{font-weight:600;font-size:13px}
.card-note{color:var(--ink-3);font-size:12px;white-space:nowrap}
.grid{display:grid;gap:10px}
.g2{grid-template-columns:1fr 1fr}
.g3{grid-template-columns:repeat(3,1fr)}
@media(max-width:900px){.g2,.g3{grid-template-columns:1fr}}
/* notices */
.notice{display:flex;gap:14px;padding:7px 0;border-top:1px solid var(--border);font-size:13.5px}
.notice:first-child{border-top:0}
.notice time{flex:0 0 72px;font-family:var(--mono);font-size:12px;color:var(--ink-3);padding-top:1px}
/* run status */
.run{padding:12px 0;border-top:1px solid var(--border)}
.run:first-child{border-top:0;padding-top:2px}
.status-line{display:grid;grid-template-columns:250px 130px 1fr;gap:14px;align-items:center}
@media(max-width:760px){.status-line{grid-template-columns:1fr;gap:6px}}
.status-name{display:inline-flex;align-items:baseline;gap:10px;min-width:0}
.status-name .swatch{width:9px;height:9px;border-radius:50%;align-self:center}
.status-name b{font-size:15px}
.status-name .state{font-size:13px;color:var(--ink-2)}
.big{font-family:var(--mono);font-size:18px;font-weight:600}
.status-sub{font-size:12px;color:var(--ink-3);margin-top:4px}
.run-head{display:flex;align-items:baseline;gap:10px;min-width:0}
.run-name{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-weight:600}
.run-name .acct{color:var(--ink-3);font-weight:400}
.run-state{color:var(--ink-3);font-size:12px;white-space:nowrap}
.run-sum{margin-left:auto;font-family:var(--mono);font-size:12.5px;color:var(--ink-2);white-space:nowrap}
.run-bar{margin-top:7px}
.phase-track{position:relative;height:6px;background:var(--surface-3);border-radius:3px;overflow:hidden}
.phase-fill{position:absolute;inset:0 auto 0 0;background:var(--accent);border-radius:3px}
.phase-fill.warm{background:var(--warn)}
.phase-text{font-size:12px;color:var(--ink-3);margin-top:4px;display:flex;justify-content:space-between;gap:10px}
.stat-row{display:flex;gap:22px;flex-wrap:wrap;margin-top:10px}
.stat{display:flex;flex-direction:column;line-height:1.2}
.stat-label{color:var(--ink-3);font-size:11px;letter-spacing:.05em;text-transform:uppercase}
.stat-val{font-family:var(--mono);font-size:14px}
.delta{color:var(--ink-2);font-size:12px}
.delta.good{color:var(--ok)}.delta.bad{color:var(--live)}
/* tables */
table.tbl{width:100%;border-collapse:collapse;font-size:12.5px;font-family:var(--mono);font-variant-numeric:tabular-nums}
.tbl th,.tbl td{padding:6px 10px;text-align:right;white-space:nowrap;border-top:1px solid var(--border)}
.tbl th{color:var(--ink-3);font-weight:500;font-size:11.5px;border-top:0}
.tbl th:first-child,.tbl td:first-child{text-align:left}
.tbl tbody tr:hover td{background:var(--surface-2)}
/* segmented control */
.seg{display:inline-flex;border:1px solid var(--border);border-radius:4px;overflow:hidden}
.seg button{padding:3px 9px;color:var(--ink-2);border-left:1px solid var(--border);font-size:12px}
.seg button:first-child{border-left:0}
.seg button.active{background:var(--ink);color:var(--bg)}
.seg button:not(.active):hover{color:var(--ink);background:var(--surface-2)}
.controls{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.ctl-label{color:var(--ink-3);margin-right:4px;font-size:12px}
input[type=range]{width:130px;accent-color:var(--accent)}
/* charts */
.chart-card{padding:10px 12px 8px;display:flex;flex-direction:column;min-width:0}
.chart-body{height:170px;margin-top:4px;position:relative}
.chart-card.large .chart-body{height:300px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:980px){.grid2{grid-template-columns:1fr}}
.chart-note{font-size:11px;color:var(--ink-3);margin-top:4px}
.shardgrid{display:flex;flex-wrap:wrap;gap:3px;margin-top:10px}
.shardtile{width:11px;height:11px;border-radius:2px;background:var(--surface-3);display:inline-block}
.shardtile.done{background:var(--ok)}
.chart-legend{display:flex;flex-wrap:wrap;gap:3px 14px;font-size:12px;color:var(--ink-2);padding-bottom:6px}
.legend-item{display:inline-flex;align-items:center;gap:6px}
.swatch{width:7px;height:7px;border-radius:2px}
/* board + games */
.boardwrap{display:grid;grid-template-columns:minmax(280px,440px) 1fr;gap:16px;align-items:start}
@media(max-width:820px){.boardwrap{grid-template-columns:1fr}}
.board{display:grid;grid-template-columns:repeat(8,1fr);grid-template-rows:repeat(8,1fr);aspect-ratio:1/1;border:1px solid var(--border);border-radius:4px;overflow:hidden}
.sq{display:flex;align-items:center;justify-content:center;position:relative}
.sq.l{background:var(--board-l)}.sq.d{background:var(--board-d)}
.sq svg{width:86%;height:86%;display:block}
.sq.from::after,.sq.to::after{content:"";position:absolute;inset:0;pointer-events:none}
.sq{position:relative}
.sq.from,.sq.to{box-shadow:inset 0 0 0 999px rgba(255,213,0,.32)}
.gcontrols{display:flex;align-items:center;gap:6px;margin-top:10px;flex-wrap:wrap}
.icon-btn{display:inline-flex;align-items:center;justify-content:center;width:32px;height:32px;border:1px solid var(--border);border-radius:4px;color:var(--ink-2)}
.icon-btn:hover{color:var(--ink);border-color:var(--ink-3)}
.icon-btn.play{width:auto;padding:0 12px;gap:7px;font-size:13px}
.moves{max-height:470px;overflow:auto;border:1px solid var(--border);border-radius:4px}
.moves table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:12.5px}
.moves td{padding:3px 8px;white-space:nowrap}
.moves .idx{color:var(--ink-3);text-align:right;width:38px}
.moves .san{cursor:pointer;border-radius:3px;padding:1px 6px;display:inline-block;min-width:56px}
.moves .san:hover{background:var(--surface-2)}
.moves .san.on{background:rgba(57,135,229,.18);color:var(--accent)}
select{background:var(--surface);border:1px solid var(--border);border-radius:4px;padding:5px 8px;max-width:320px;font-size:13px}
.foot{color:var(--ink-3);font-size:12px;margin:28px 0 50px;font-family:var(--mono)}
.empty{color:var(--ink-3);padding:10px 0}
</style>
</head>
<body>
<div class="nav"><div class="nav-inner">
  <div class="brand">chess-slm</div>
  <div class="tabs">
    <a data-tab="overview" class="active">overview</a>
    <a data-tab="metrics">metrics</a>
    <a data-tab="games">games</a>
  </div>
  <div class="nav-right" id="navright"></div>
</div></div>

<div class="wrap">
<!-- overview -->
<section id="tab-overview">
  <div class="section">
    <div class="section-title">notices</div>
    <div class="card" id="notices"></div>
  </div>

  <div class="section">
    <div class="section-title" style="margin-top:0">runs</div>
    <div class="card" id="runs"></div>
  </div>

  <div class="section">
    <div class="section-title" style="margin-top:0">pipeline</div>
    <div class="grid g2">
      <div class="card" id="stages"></div>
      <div class="card" id="corpus"></div>
    </div>
  </div>

  <div class="section">
    <div class="section-title" style="margin-top:0">benchmarks</div>
    <div class="grid g3">
      <div class="card chart-card">
        <div class="card-head"><div class="card-title">MATE <span class="muted">4,000 rows</span></div><div class="card-note mono" id="mate-top"></div></div>
        <div class="chart-body"><canvas id="c-mate"></canvas></div>
      </div>
      <div class="card chart-card">
        <div class="card-head"><div class="card-title">puzzles <span class="muted">10K</span></div><div class="card-note mono" id="puz-top"></div></div>
        <div class="chart-body"><canvas id="c-puz"></canvas></div>
      </div>
      <div class="card chart-card">
        <div class="card-head"><div class="card-title">loss <span class="muted">train / dev</span></div><div class="card-note mono" id="loss-top"></div></div>
        <div class="chart-body"><canvas id="c-loss"></canvas></div>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-title" style="margin-top:0">infrastructure</div>
    <div class="grid g2">
      <div class="card"><div class="card-head"><div class="card-title">kernels</div></div><table class="tbl" id="kernels"></table></div>
      <div class="card"><div class="card-head"><div class="card-title">gpu quota</div><div class="card-note">weekly reset</div></div><table class="tbl" id="quota"></table></div>
    </div>
  </div>
</section>

<!-- metrics -->
<section id="tab-metrics" class="hidden">
  <div class="section">
    <div class="card chart-card large">
      <div class="card-head">
        <div class="card-title">training metrics <span class="muted">ccgavn-5m-seed0 · per checkpoint</span></div>
        <div class="controls">
          <span><span class="ctl-label">scale</span><span class="seg"><button id="lin" class="active">linear</button><button id="log">log</button></span></span>
          <span><span class="ctl-label">smoothing</span><input id="smooth" type="range" min="0" max="95" value="0"><span class="mono" id="smoothv">0.00</span></span>
        </div>
      </div>
      <div class="chart-body"><canvas id="c-loss2"></canvas></div>
      <div class="chart-note">shaded bands = steps trained on the puzzle shards (P000/P001) — dev dips there are transient distribution drift, not regressions</div>
    </div>
  </div>
  <div class="section grid2">
    <div class="card chart-card">
      <div class="card-head"><div class="card-title">accuracy vs step <span class="muted">frozen + preview evals</span></div></div>
      <div class="chart-body"><canvas id="c-acc"></canvas></div>
    </div>
    <div class="card chart-card">
      <div class="card-head"><div class="card-title">dev components <span class="muted">distribution / cross-entropy</span></div></div>
      <div class="chart-body"><canvas id="c-comp"></canvas></div>
    </div>
  </div>
  <div class="section grid2">
    <div class="card chart-card">
      <div class="card-head"><div class="card-title">throughput <span class="muted">samples/s</span></div></div>
      <div class="chart-body"><canvas id="c-thr"></canvas></div>
      <div class="chart-note" id="thr-note"></div>
    </div>
    <div class="card" id="corpus-detail"></div>
  </div>
  <div class="section grid2">
    <div class="card">
      <div class="card-head"><div class="card-title">eval history</div><div class="card-note">Δ vs previous · gap to 9M teacher</div></div>
      <table class="tbl" id="evaltable"></table>
    </div>
    <div class="card" id="quota-detail"></div>
  </div>
  <div class="section">
    <div class="card"><div class="card-head"><div class="card-title">checkpoints</div><div class="card-note" id="ckpt-note"></div></div><table class="tbl" id="ckpttable"></table></div>
  </div>
</section>

<!-- games -->
<section id="tab-games" class="hidden">
  <div class="section">
    <div class="card">
      <div class="card-head">
        <div class="card-title">archived games <span class="muted">searchless play vs stockfish</span></div>
        <select id="gamesel"></select>
      </div>
      <div class="boardwrap">
        <div>
          <div class="board" id="board"></div>
          <div class="gcontrols">
            <button class="icon-btn" id="b-first" aria-label="first move"><svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M3 2v10M12 2 6 7l6 5z"/></svg></button>
            <button class="icon-btn" id="b-prev" aria-label="previous move"><svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M10 2 4 7l6 5z"/></svg></button>
            <button class="icon-btn" id="b-next" aria-label="next move"><svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.6"><path d="m4 2 6 5-6 5z"/></svg></button>
            <button class="icon-btn" id="b-last" aria-label="last move"><svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M11 2v10M2 2l6 5-6 5z"/></svg></button>
            <button class="icon-btn play" id="b-play"><svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor"><path d="M2 1.5 10 6l-8 4.5z"/></svg>autoplay</button>
            <span class="mono dim" id="plyinfo"></span>
          </div>
          <div class="dim mono" id="gameresult" style="margin-top:8px"></div>
        </div>
        <div class="moves" id="movelist"></div>
      </div>
    </div>
  </div>
</section>

<div class="foot" id="foot"></div>
</div>

<script>
const PIECE_SVG={"wK":"<svg xmlns=\\"http://www.w3.org/2000/svg\\" viewBox=\\"0 0 45 45\\"><g fill=\\"none\\" fill-rule=\\"evenodd\\" stroke=\\"#000\\" stroke-linecap=\\"round\\" stroke-linejoin=\\"round\\" stroke-width=\\"1.5\\"><path stroke-linejoin=\\"miter\\" d=\\"M22.5 11.63V6M20 8h5\\"/><path fill=\\"#fff\\" stroke-linecap=\\"butt\\" stroke-linejoin=\\"miter\\" d=\\"M22.5 25s4.5-7.5 3-10.5c0 0-1-2.5-3-2.5s-3 2.5-3 2.5c-1.5 3 3 10.5 3 10.5\\"/><path fill=\\"#fff\\" d=\\"M11.5 37c5.5 3.5 15.5 3.5 21 0v-7s9-4.5 6-10.5c-4-6.5-13.5-3.5-16 4V27v-3.5c-3.5-7.5-13-10.5-16-4-3 6 5 10 5 10z\\"/><path d=\\"M11.5 30c5.5-3 15.5-3 21 0m-21 3.5c5.5-3 15.5-3 21 0m-21 3.5c5.5-3 15.5-3 21 0\\"/></g></svg>","wQ":"<svg xmlns=\\"http://www.w3.org/2000/svg\\" viewBox=\\"0 0 45 45\\"><g fill=\\"#fff\\" fill-rule=\\"evenodd\\" stroke=\\"#000\\" stroke-linecap=\\"round\\" stroke-linejoin=\\"round\\" stroke-width=\\"1.5\\"><path d=\\"M8 12a2 2 0 1 1-4 0 2 2 0 1 1 4 0m16.5-4.5a2 2 0 1 1-4 0 2 2 0 1 1 4 0M41 12a2 2 0 1 1-4 0 2 2 0 1 1 4 0M16 8.5a2 2 0 1 1-4 0 2 2 0 1 1 4 0M33 9a2 2 0 1 1-4 0 2 2 0 1 1 4 0\\"/><path stroke-linecap=\\"butt\\" d=\\"M9 26c8.5-1.5 21-1.5 27 0l2-12-7 11V11l-5.5 13.5-3-15-3 15-5.5-14V25L7 14z\\"/><path stroke-linecap=\\"butt\\" d=\\"M9 26c0 2 1.5 2 2.5 4 1 1.5 1 1 .5 3.5-1.5 1-1.5 2.5-1.5 2.5-1.5 1.5.5 2.5.5 2.5 6.5 1 16.5 1 23 0 0 0 1.5-1 0-2.5 0 0 .5-1.5-1-2.5-.5-2.5-.5-2 .5-3.5 1-2 2.5-2 2.5-4-8.5-1.5-18.5-1.5-27 0z\\"/><path fill=\\"none\\" d=\\"M11.5 30c3.5-1 18.5-1 22 0M12 33.5c6-1 15-1 21 0\\"/></g></svg>","wR":"<svg xmlns=\\"http://www.w3.org/2000/svg\\" viewBox=\\"0 0 45 45\\"><g fill=\\"#fff\\" fill-rule=\\"evenodd\\" stroke=\\"#000\\" stroke-linecap=\\"round\\" stroke-linejoin=\\"round\\" stroke-width=\\"1.5\\"><path stroke-linecap=\\"butt\\" d=\\"M9 39h27v-3H9zm3-3v-4h21v4zm-1-22V9h4v2h5V9h5v2h5V9h4v5\\"/><path d=\\"m34 14-3 3H14l-3-3\\"/><path stroke-linecap=\\"butt\\" stroke-linejoin=\\"miter\\" d=\\"M31 17v12.5H14V17\\"/><path d=\\"m31 29.5 1.5 2.5h-20l1.5-2.5\\"/><path fill=\\"none\\" stroke-linejoin=\\"miter\\" d=\\"M11 14h23\\"/></g></svg>","wB":"<svg xmlns=\\"http://www.w3.org/2000/svg\\" viewBox=\\"0 0 45 45\\"><g fill=\\"none\\" fill-rule=\\"evenodd\\" stroke=\\"#000\\" stroke-linecap=\\"round\\" stroke-linejoin=\\"round\\" stroke-width=\\"1.5\\"><g fill=\\"#fff\\" stroke-linecap=\\"butt\\"><path d=\\"M9 36c3.39-.97 10.11.43 13.5-2 3.39 2.43 10.11 1.03 13.5 2 0 0 1.65.54 3 2-.68.97-1.65.99-3 .5-3.39-.97-10.11.46-13.5-1-3.39 1.46-10.11.03-13.5 1-1.35.49-2.32.47-3-.5 1.35-1.94 3-2 3-2z\\"/><path d=\\"M15 32c2.5 2.5 12.5 2.5 15 0 .5-1.5 0-2 0-2 0-2.5-2.5-4-2.5-4 5.5-1.5 6-11.5-5-15.5-11 4-10.5 14-5 15.5 0 0-2.5 1.5-2.5 4 0 0-.5.5 0 2z\\"/><path d=\\"M25 8a2.5 2.5 0 1 1-5 0 2.5 2.5 0 1 1 5 0z\\"/></g><path stroke-linejoin=\\"miter\\" d=\\"M17.5 26h10M15 30h15m-7.5-14.5v5M20 18h5\\"/></g></svg>","wN":"<svg xmlns=\\"http://www.w3.org/2000/svg\\" viewBox=\\"0 0 45 45\\"><g fill=\\"none\\" fill-rule=\\"evenodd\\" stroke=\\"#000\\" stroke-linecap=\\"round\\" stroke-linejoin=\\"round\\" stroke-width=\\"1.5\\"><path fill=\\"#fff\\" d=\\"M22 10c10.5 1 16.5 8 16 29H15c0-9 10-6.5 8-21\\"/><path fill=\\"#fff\\" d=\\"M24 18c.38 2.91-5.55 7.37-8 9-3 2-2.82 4.34-5 4-1.042-.94 1.41-3.04 0-3-1 0 .19 1.23-1 2-1 0-4.003 1-4-4 0-2 6-12 6-12s1.89-1.9 2-3.5c-.73-.994-.5-2-.5-3 1-1 3 2.5 3 2.5h2s.78-1.992 2.5-3c1 0 1 3 1 3\\"/><path fill=\\"#000\\" d=\\"M9.5 25.5a.5.5 0 1 1-1 0 .5.5 0 1 1 1 0m5.433-9.75a.5 1.5 30 1 1-.866-.5.5 1.5 30 1 1 .866.5\\"/></g></svg>","wP":"<svg xmlns=\\"http://www.w3.org/2000/svg\\" viewBox=\\"0 0 45 45\\"><path fill=\\"#fff\\" stroke=\\"#000\\" stroke-linecap=\\"round\\" stroke-width=\\"1.5\\" d=\\"M22.5 9c-2.21 0-4 1.79-4 4 0 .89.29 1.71.78 2.38C17.33 16.5 16 18.59 16 21c0 2.03.94 3.84 2.41 5.03-3 1.06-7.41 5.55-7.41 13.47h23c0-7.92-4.41-12.41-7.41-13.47 1.47-1.19 2.41-3 2.41-5.03 0-2.41-1.33-4.5-3.28-5.62.49-.67.78-1.49.78-2.38 0-2.21-1.79-4-4-4z\\"/></svg>","bK":"<svg xmlns=\\"http://www.w3.org/2000/svg\\" viewBox=\\"0 0 45 45\\"><g fill=\\"none\\" fill-rule=\\"evenodd\\" stroke=\\"#000\\" stroke-linecap=\\"round\\" stroke-linejoin=\\"round\\" stroke-width=\\"1.5\\"><path stroke-linejoin=\\"miter\\" d=\\"M22.5 11.6V6\\"/><path fill=\\"#000\\" stroke-linecap=\\"butt\\" stroke-linejoin=\\"miter\\" d=\\"M22.5 25s4.5-7.5 3-10.5c0 0-1-2.5-3-2.5s-3 2.5-3 2.5c-1.5 3 3 10.5 3 10.5\\"/><path fill=\\"#000\\" d=\\"M11.5 37a22.3 22.3 0 0 0 21 0v-7s9-4.5 6-10.5c-4-6.5-13.5-3.5-16 4V27v-3.5c-3.5-7.5-13-10.5-16-4-3 6 5 10 5 10z\\"/><path stroke-linejoin=\\"miter\\" d=\\"M20 8h5\\"/><path stroke=\\"#ececec\\" d=\\"M32 29.5s8.5-4 6-9.7C34.1 14 25 18 22.5 24.6v2.1-2.1C20 18 9.9 14 7 19.9c-2.5 5.6 4.8 9 4.8 9\\"/><path stroke=\\"#ececec\\" d=\\"M11.5 30c5.5-3 15.5-3 21 0m-21 3.5c5.5-3 15.5-3 21 0m-21 3.5c5.5-3 15.5-3 21 0\\"/></g></svg>","bQ":"<svg xmlns=\\"http://www.w3.org/2000/svg\\" viewBox=\\"0 0 45 45\\"><g fill-rule=\\"evenodd\\" stroke=\\"#000\\" stroke-linecap=\\"round\\" stroke-linejoin=\\"round\\" stroke-width=\\"1.5\\"><g stroke=\\"none\\"><circle cx=\\"6\\" cy=\\"12\\" r=\\"2.75\\"/><circle cx=\\"14\\" cy=\\"9\\" r=\\"2.75\\"/><circle cx=\\"22.5\\" cy=\\"8\\" r=\\"2.75\\"/><circle cx=\\"31\\" cy=\\"9\\" r=\\"2.75\\"/><circle cx=\\"39\\" cy=\\"12\\" r=\\"2.75\\"/></g><path stroke-linecap=\\"butt\\" d=\\"M9 26c8.5-1.5 21-1.5 27 0l2.5-12.5L31 25l-.3-14.1-5.2 13.6-3-14.5-3 14.5-5.2-13.6L14 25 6.5 13.5z\\"/><path stroke-linecap=\\"butt\\" d=\\"M9 26c0 2 1.5 2 2.5 4 1 1.5 1 1 .5 3.5-1.5 1-1.5 2.5-1.5 2.5-1.5 1.5.5 2.5.5 2.5 6.5 1 16.5 1 23 0 0 0 1.5-1 0-2.5 0 0 .5-1.5-1-2.5-.5-2.5-.5-2 .5-3.5 1-2 2.5-2 2.5-4-8.5-1.5-18.5-1.5-27 0z\\"/><path fill=\\"none\\" stroke-linecap=\\"butt\\" d=\\"M11 38.5a35 35 1 0 0 23 0\\"/><path fill=\\"none\\" stroke=\\"#ececec\\" d=\\"M11 29a35 35 1 0 1 23 0m-21.5 2.5h20m-21 3a35 35 1 0 0 22 0m-23 3a35 35 1 0 0 24 0\\"/></g></svg>","bR":"<svg xmlns=\\"http://www.w3.org/2000/svg\\" viewBox=\\"0 0 45 45\\"><g fill-rule=\\"evenodd\\" stroke=\\"#000\\" stroke-linecap=\\"round\\" stroke-linejoin=\\"round\\" stroke-width=\\"1.5\\"><path stroke-linecap=\\"butt\\" d=\\"M9 39h27v-3H9zm3.5-7 1.5-2.5h17l1.5 2.5zm-.5 4v-4h21v4z\\"/><path stroke-linecap=\\"butt\\" stroke-linejoin=\\"miter\\" d=\\"M14 29.5v-13h17v13z\\"/><path stroke-linecap=\\"butt\\" d=\\"M14 16.5 11 14h23l-3 2.5zM11 14V9h4v2h5V9h5v2h5V9h4v5z\\"/><path fill=\\"none\\" stroke=\\"#ececec\\" stroke-linejoin=\\"miter\\" stroke-width=\\"1\\" d=\\"M12 35.5h21m-20-4h19m-18-2h17m-17-13h17M11 14h23\\"/></g></svg>","bB":"<svg xmlns=\\"http://www.w3.org/2000/svg\\" viewBox=\\"0 0 45 45\\"><g fill=\\"none\\" fill-rule=\\"evenodd\\" stroke=\\"#000\\" stroke-linecap=\\"round\\" stroke-linejoin=\\"round\\" stroke-width=\\"1.5\\"><g fill=\\"#000\\" stroke-linecap=\\"butt\\"><path d=\\"M9 36c3.4-1 10.1.4 13.5-2 3.4 2.4 10.1 1 13.5 2 0 0 1.6.5 3 2-.7 1-1.6 1-3 .5-3.4-1-10.1.5-13.5-1-3.4 1.5-10.1 0-13.5 1-1.4.5-2.3.5-3-.5 1.4-2 3-2 3-2z\\"/><path d=\\"M15 32c2.5 2.5 12.5 2.5 15 0 .5-1.5 0-2 0-2 0-2.5-2.5-4-2.5-4 5.5-1.5 6-11.5-5-15.5-11 4-10.5 14-5 15.5 0 0-2.5 1.5-2.5 4 0 0-.5.5 0 2z\\"/><path d=\\"M25 8a2.5 2.5 0 1 1-5 0 2.5 2.5 0 1 1 5 0z\\"/></g><path stroke=\\"#ececec\\" stroke-linejoin=\\"miter\\" d=\\"M17.5 26h10M15 30h15m-7.5-14.5v5M20 18h5\\"/></g></svg>","bN":"<svg xmlns=\\"http://www.w3.org/2000/svg\\" viewBox=\\"0 0 45 45\\"><g fill=\\"none\\" fill-rule=\\"evenodd\\" stroke=\\"#000\\" stroke-linecap=\\"round\\" stroke-linejoin=\\"round\\" stroke-width=\\"1.5\\"><path fill=\\"#000\\" d=\\"M22 10c10.5 1 16.5 8 16 29H15c0-9 10-6.5 8-21\\"/><path fill=\\"#000\\" d=\\"M24 18c.38 2.91-5.55 7.37-8 9-3 2-2.82 4.34-5 4-1.04-.94 1.41-3.04 0-3-1 0 .19 1.23-1 2-1 0-4 1-4-4 0-2 6-12 6-12s1.89-1.9 2-3.5c-.73-1-.5-2-.5-3 1-1 3 2.5 3 2.5h2s.78-2 2.5-3c1 0 1 3 1 3\\"/><path fill=\\"#ececec\\" stroke=\\"#ececec\\" d=\\"M9.5 25.5a.5.5 0 1 1-1 0 .5.5 0 1 1 1 0m5.43-9.75a.5 1.5 30 1 1-.86-.5.5 1.5 30 1 1 .86.5\\"/><path fill=\\"#ececec\\" stroke=\\"none\\" d=\\"m24.55 10.4-.45 1.45.5.15c3.15 1 5.65 2.49 7.9 6.75S35.75 29.06 35.25 39l-.05.5h2.25l.05-.5c.5-10.06-.88-16.85-3.25-21.34s-5.79-6.64-9.19-7.16z\\"/></g></svg>","bP":"<svg xmlns=\\"http://www.w3.org/2000/svg\\" viewBox=\\"0 0 45 45\\"><path stroke=\\"#000\\" stroke-linecap=\\"round\\" stroke-width=\\"1.5\\" d=\\"M22.5 9a4 4 0 0 0-3.22 6.38 6.48 6.48 0 0 0-.87 10.65c-3 1.06-7.41 5.55-7.41 13.47h23c0-7.92-4.41-12.41-7.41-13.47a6.46 6.46 0 0 0-.87-10.65A4.01 4.01 0 0 0 22.5 9z\\"/></svg>"};
const CODE={p:"bP",n:"bN",b:"bB",r:"bR",q:"bQ",k:"bK",P:"wP",N:"wN",B:"wB",R:"wR",Q:"wQ",K:"wK"};
let snap={},charts={},game=null,ply=0,timer=null,logScale=false,smooth=0;
function esc(s){return String(s==null?"":s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));}
function fmt(n,d){d=d==null?0:d;return n==null?"—":Number(n).toLocaleString(undefined,{maximumFractionDigits:d,minimumFractionDigits:d});}
function ago(ts){const s=(Date.now()-ts)/1000;if(s<60)return s.toFixed(0)+"s ago";if(s<3600)return (s/60).toFixed(0)+"m ago";if(s<86400)return (s/3600).toFixed(1)+"h ago";return (s/86400).toFixed(1)+"d ago";}
/* board */
function fenToBoard(fen){
  const rows=(fen||"8/8/8/8/8/8/8/8").split(" ")[0].split("/");let h="";
  rows.forEach((row,r)=>{let f=0;for(const ch of row){
    if(/\\d/.test(ch)){for(let i=0;i<+ch;i++){h+=cell(r,f,"");f++;}}
    else{h+=cell(r,f,ch);f++;}}});
  return h;
}
function cell(r,f,ch){const d=(r+f)%2===1;let inner="";
  if(ch){const code=CODE[ch];if(code)inner=PIECE_SVG[code];}
  return '<div class="sq '+(d?"d":"l")+'" data-sq="'+r+","+f+'">'+inner+"</div>";}
/* navigation / top */
function renderNav(){
  const c=snap.curve||[];const last=c.length?c[c.length-1]:null;
  const corpus=snap.corpus||{};const step=last?last.step:0;
  const live=snap.ingested_at||snap.updated_at;
  document.getElementById("navright").innerHTML=
    '<div class="nav-cell"><span class="k">step</span><span class="v">'+fmt(step)+"</span></div>"+
    '<div class="nav-cell opt"><span class="k">samples</span><span class="v">'+(step*2048/1e9).toFixed(2)+"B</span></div>"+
    '<div class="nav-cell opt"><span class="k">corpus</span><span class="v">'+((corpus.labeled_rows||0)/1e6).toFixed(0)+"M</span></div>"+
    '<div class="nav-cell"><span class="k">updated</span><span class="v"><span class="dot live"></span> '+(live?new Date(live).toLocaleTimeString():"—")+"</span></div>";
}
function renderNotices(){
  const ns=snap.notices||[];
  document.getElementById("notices").innerHTML=ns.length
    ? ns.map(n=>'<div class="notice"><time>'+ago(n.t)+"</time><div>"+esc(n.text)+"</div></div>").join("")
    : '<div class="empty">no events recorded yet</div>';
}
function runBlock(key,status,accounts){
  const parts=key.split("/");
  const acct=parts.length>1?parts[0]:"";const name=parts.length>1?parts.slice(1).join("/"):key;
  const running=/RUNNING|QUEUED/i.test(status);
  const c=snap.curve||[];const last=c.length?c[c.length-1]:null;
  const training=key.indexOf("ccgavn-5m-seed0")>=0;
  const corpus=snap.corpus||{};
  const gateOpen=(corpus.labeled_rows||0)>=(corpus.target_rows||920000000);
  let state;
  if(running)state="running";
  else if(/DONE|COMPLETE/i.test(status))state="complete";
  else if(/error|Traceback|Error/i.test(status))state="failed";
  else if(training&&last&&last.step<1620000)state=gateOpen?"resuming":"awaiting 1B corpus";
  else state="idle";
  let sum="",bar="",stats="";
  if(training&&last){
    const pct=Math.min(100,last.step/1620000*100);
    sum="step "+fmt(last.step)+" / 1,620,000 · "+pct.toFixed(1)+"%";
    bar='<div class="phase-track run-bar"><div class="phase-fill" style="width:'+pct.toFixed(1)+'%"></div></div>';
    stats=statRow([["train loss",last.train.toFixed(4)],["dev loss",last.dev!=null?last.dev.toFixed(4):"—"],
      ["samples",(last.step*2048/1e9).toFixed(2)+"B"],["epochs",(last.step*2048/94.3e6).toFixed(1)]]);
  } else if(key.indexOf("build-2b")>=0&&accounts&&accounts[acct]){
    const a=accounts[acct];const pct=a.planned_rows?Math.min(100,a.done_rows/a.planned_rows*100):0;
    sum=a.done_shards+" / "+a.planned_shards+" shards · "+(a.done_rows/1e6).toFixed(0)+"M rows";
    bar='<div class="phase-track run-bar"><div class="phase-fill warm" style="width:'+pct.toFixed(1)+'%"></div></div>';
  }
  const dot=running?"var(--live)":(state==="awaiting 1B corpus"?"var(--warn)":"var(--ink-3)");
  return '<div class="run"><div class="run-head">'+
    '<span class="swatch" style="background:'+dot+'"></span>'+
    '<span class="run-name">'+(acct?'<span class="acct">'+esc(acct)+"/</span>":"")+esc(name)+"</span>"+
    '<span class="run-state">'+state+"</span>"+
    (sum?'<span class="run-sum">'+sum+"</span>":"")+"</div>"+bar+stats+"</div>";
}
function statRow(pairs){return '<div class="stat-row">'+pairs.map(p=>'<span class="stat"><span class="stat-label">'+p[0]+'</span><span class="stat-val">'+p[1]+"</span></span>").join("")+"</div>";}
function renderRuns(){
  const kernels=snap.kernels||[];const runs=snap.runs||{};const out=[];
  const accounts=snap.build_accounts||{};
  for(const k of kernels)out.push(runBlock(k.account+"/"+k.kernel,k.status||"?",accounts));
  for(const p in runs){const txt=String(runs[p]);
    if(txt.trim().indexOf("DONE")!==0)continue;   // old failure traces stay out of the run list
    out.push(runBlock(p,txt.split("\\n")[0].slice(0,60),accounts));}
  document.getElementById("runs").innerHTML=out.length?out.join(""):'<div class="empty">no run data</div>';
}
function renderStages(){
  const s=snap.stages||{};
  const rows=[["train320","CC-GAVN 320k training"],["eval320","320k frozen eval"],["corpus1b","1B corpus labeling"],["train1b","1B continuation → 1.62M steps"],["eval1b","1B frozen eval"]];
  document.getElementById("stages").innerHTML='<div class="card-head"><div class="card-title">stages</div></div>'+
    rows.map(([k,label])=>{const p=Math.round(s[k]||0);
      return '<div style="margin:9px 0"><div class="phase-text" style="margin:0 0 4px"><span>'+label+'</span><span class="mono">'+p+'%</span></div>'+
        '<div class="phase-track"><div class="phase-fill" style="width:'+p+'%"></div></div></div>';}).join("");
}
let _prevCorpus=null,_curRate=null;
function updateCorpusRate(){
  const c=snap.corpus||{};const now=Date.now();
  if(_prevCorpus&&c.labeled_rows!=null){const dt=(now-_prevCorpus.t)/1000;
    if(dt>30)_curRate=(c.labeled_rows-_prevCorpus.rows)/dt;}
  if(c.labeled_rows!=null)_prevCorpus={t:now,rows:c.labeled_rows};
}
function corpusGateInfo(){
  const c=snap.corpus||{};
  const done=new Set(c.done_tags||[]);
  const remaining=Math.max(0,(c.target_rows||920000000)-(c.labeled_rows||0));
  const remRows=(c.all_tags||[]).filter(x=>!done.has(x)).map(x=>({x:x,r:(c.rows_by_tag||{})[x]||0}))
    .sort((a,b)=>b.r-a.r);
  let acc=0,gateN=0;
  for(const x of remRows){if(acc>=remaining)break;acc+=x.r;gateN++;}
  return {c:c,done:done,remaining:remaining,gateN:gateN,
    gateLabel:remRows.length?("~"+gateN+" of "+(c.shards_planned||0)+" planned"):"—",
    eta:(_curRate&&_curRate>0)?remaining/_curRate:null};
}
function corpusTiles(c,done){
  return (c.all_tags||[]).map(t=>{const r=(c.rows_by_tag||{})[t];
    return '<span class="shardtile'+(done.has(t)?" done":"")+'" title="shard-'+esc(t)+" · "+(r?(r/1e6).toFixed(1):"?")+'M rows"></span>';}).join("");
}
function renderCorpus(){
  const g=corpusGateInfo();const c=g.c;
  const note=(c.all_tags&&c.all_tags.length)
    ? g.gateLabel+" to gate · "+(c.shards_done||0)+" / "+(c.shards_planned||0)+" built"
    : (c.shards_done||0)+" / "+(c.shards_planned||0)+" shards";
  document.getElementById("corpus").innerHTML='<div class="card-head"><div class="card-title">1B corpus</div><div class="card-note mono">'+
    note+'</div></div>'+
    statRow([["labeled",((c.labeled_rows||0)/1e6).toFixed(1)+"M"],["rate",_curRate!=null?(_curRate/1000).toFixed(1)+"k rows/s":"—"],
      ["eta to gate",g.eta!=null?(g.eta/60).toFixed(0)+" min":"—"],["rows to gate",(g.remaining/1e6).toFixed(0)+"M"]])+
    '<div class="shardgrid">'+corpusTiles(c,g.done)+"</div>";
}
function renderInfra(){
  const k=snap.kernels||[];
  document.getElementById("kernels").innerHTML='<thead><tr><th>kernel</th><th>status</th></tr></thead><tbody>'+
    k.map(x=>{const cls=/RUNNING|QUEUED/i.test(x.status)?"var(--live)":"var(--ink-3)";
      return '<tr><td><span class="swatch" style="background:'+cls+';display:inline-block;margin-right:7px"></span>'+esc(x.account+"/"+x.kernel)+"</td><td>"+esc(x.status)+"</td></tr>";}).join("")+"</tbody>";
  const q=snap.quota||{};
  document.getElementById("quota").innerHTML='<thead><tr><th>account</th><th>gpu left</th></tr></thead><tbody>'+
    Object.keys(q).map(a=>"<tr><td>"+esc(a)+'</td><td>'+esc(q[a])+"h</td></tr>").join("")+"</tbody>";
}
function evalRows(){
  const rows=[];
  for(const p in (snap.evals||{})){const s=snap.evals[p];
    const mate=s.mate&&s.mate[0]?s.mate[0].pct:null;const puz=s.puzzles&&s.puzzles[0]?s.puzzles[0].pct:null;
    rows.push({label:p.replace(/^eval-results\\//,"").replace(/\\/eval-summary\\.json$/,""),mate:mate,puz:puz,ref:false});}
  rows.sort((a,b)=>(b.mate||0)-(a.mate||0));
  return rows.concat(snap.reference||[]);
}
function drawBar(id,key,color){
  const el=document.getElementById(id);if(!el||!window.Chart)return;
  const rows=evalRows();
  if(charts[id])charts[id].destroy();
  charts[id]=new Chart(el,{type:"bar",data:{labels:rows.map(r=>r.label),datasets:[{data:rows.map(r=>r[key]),
    backgroundColor:rows.map(r=>r.ref?"#3a3a39":color),borderRadius:2}]},
    options:{animation:false,maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{enabled:true}},
      scales:{x:{ticks:{color:"#6f6f6c",maxRotation:60,font:{size:9}},grid:{display:false}},
        y:{suggestedMin:40,suggestedMax:100,ticks:{color:"#6f6f6c",font:{size:10}},grid:{color:"#242423"}}}}});
}
function smoothArr(a,w){if(!w)return a;const al=1-w;let l=null;return a.map(v=>{l=l==null?v:al*v+(1-al)*l;return l;});}
const puzzleShade={id:"puzzleShade",beforeDatasetsDraw(chart){
  const c=snap.curve||[];const x=chart.scales.x;const a=chart.chartArea;
  if(!x||!a)return;
  const runs=[];let cur=null;
  for(const p of c){const isP=String(p.shard||"").startsWith("P");
    if(isP){if(!cur)cur={s:p.step,e:p.step};else cur.e=p.step;}
    else if(cur){runs.push(cur);cur=null;}}
  if(cur)runs.push(cur);
  if(!runs.length)return;
  const ctx=chart.ctx;ctx.save();ctx.fillStyle="rgba(201,162,39,.13)";
  for(const r of runs){const xa=x.getPixelForValue(r.s),xb=x.getPixelForValue(r.e);
    if(!isFinite(xa)||!isFinite(xb))continue;
    const w=Math.max(2,xb-xa);
    ctx.fillStyle="rgba(201,162,39,.13)";ctx.fillRect(xa,a.top,w,a.bottom-a.top);
    ctx.fillStyle="rgba(201,162,39,.45)";ctx.fillRect(xa,a.top,w,1.5);}
  ctx.restore();}};
function drawLoss(id,log,sm){
  const el=document.getElementById(id);if(!el||!window.Chart)return;
  const c=snap.curve||[];
  const tr=smoothArr(c.map(p=>p.train),sm);
  const dv=smoothArr(c.map(p=>p.dev==null?null:p.dev),sm);
  if(charts[id])charts[id].destroy();
  charts[id]=new Chart(el,{type:"line",data:{labels:c.map(p=>p.step),datasets:[
    {label:"train",data:tr,borderColor:"#3987e5",backgroundColor:"rgba(57,135,229,.08)",pointRadius:0,borderWidth:1.5,fill:true},
    {label:"dev",data:dv,borderColor:"#d9463f",pointRadius:0,borderWidth:1.5,spanGaps:true}]},
    options:{animation:false,maintainAspectRatio:false,interaction:{mode:"index",intersect:false},
      plugins:{legend:{labels:{color:"#a3a3a0",boxWidth:8,font:{size:11}}}},
      scales:{x:{ticks:{color:"#6f6f6c",maxTicksLimit:8,font:{size:10}},grid:{color:"#1a1a1a"}},
        y:{type:log?"logarithmic":"linear",ticks:{color:"#6f6f6c",font:{size:10}},grid:{color:"#242423"}}}},
    plugins:[puzzleShade]});
}
/* metrics tab extras */
function fmtStep(s){return s>=1e6?(s/1e6).toFixed(2)+"M":(s/1000).toFixed(0)+"k";}
function evalSeries(){
  const rows=[];
  for(const p in (snap.evals||{})){const s=snap.evals[p];
    const tail=String(s.checkpoint||"").split("checkpoint-")[1];
    const step=tail?parseInt(tail,10):NaN;
    if(!isFinite(step))continue;
    const mate=s.mate&&s.mate[0]?s.mate[0].pct:null;const puz=s.puzzles&&s.puzzles[0]?s.puzzles[0].pct:null;
    rows.push({step:step,mate:mate,puz:puz,dir:p});}
  rows.sort((a,b)=>a.step-b.step);return rows;
}
function drawAccuracy(){
  const el=document.getElementById("c-acc");if(!el||!window.Chart)return;
  const rows=evalSeries();const refs=snap.reference||[];
  const t=refs.filter(r=>r.ref)[0]||{};
  if(charts["c-acc"])charts["c-acc"].destroy();
  charts["c-acc"]=new Chart(el,{type:"line",data:{labels:rows.map(r=>fmtStep(r.step)),datasets:[
    {label:"MATE",data:rows.map(r=>r.mate),borderColor:"#3987e5",backgroundColor:"#3987e5",pointRadius:3,borderWidth:2,tension:.3},
    {label:"puzzles",data:rows.map(r=>r.puz),borderColor:"#3fa66a",backgroundColor:"#3fa66a",pointRadius:3,borderWidth:2,tension:.3},
    {label:"9M teacher MATE",data:rows.map(()=>t.mate),borderColor:"#6f6f6c",borderDash:[4,4],pointRadius:0,borderWidth:1},
    {label:"9M teacher puzzles",data:rows.map(()=>t.puzzles),borderColor:"#6f6f6c",borderDash:[4,4],pointRadius:0,borderWidth:1}]},
    options:{animation:false,maintainAspectRatio:false,interaction:{mode:"index",intersect:false},
      plugins:{legend:{labels:{color:"#a3a3a0",boxWidth:8,font:{size:10}}}},
      scales:{x:{ticks:{color:"#6f6f6c",font:{size:10}},grid:{color:"#1a1a1a"}},
        y:{suggestedMin:40,suggestedMax:100,ticks:{color:"#6f6f6c",font:{size:10}},grid:{color:"#242423"}}}}});
}
function drawComponents(){
  const el=document.getElementById("c-comp");if(!el||!window.Chart)return;
  const c=snap.curve||[];
  if(charts["c-comp"])charts["c-comp"].destroy();
  charts["c-comp"]=new Chart(el,{type:"line",data:{labels:c.map(p=>p.step),datasets:[
    {label:"dev dist",data:c.map(p=>p.dev_dist==null?null:p.dev_dist),borderColor:"#3987e5",pointRadius:0,borderWidth:1.5,spanGaps:true},
    {label:"dev ce",data:c.map(p=>p.dev_ce==null?null:p.dev_ce),borderColor:"#c9a227",pointRadius:0,borderWidth:1.5,spanGaps:true}]},
    options:{animation:false,maintainAspectRatio:false,interaction:{mode:"index",intersect:false},
      plugins:{legend:{labels:{color:"#a3a3a0",boxWidth:8,font:{size:10}}}},
      scales:{x:{ticks:{color:"#6f6f6c",maxTicksLimit:8,font:{size:10}},grid:{color:"#1a1a1a"}},
        y:{ticks:{color:"#6f6f6c",font:{size:10}},grid:{color:"#242423"}}}}});
}
function drawThroughput(){
  const el=document.getElementById("c-thr");if(!el||!window.Chart)return;
  const c=snap.curve||[];
  const pts=[];
  for(let i=0;i<c.length;i++){
    const p=c[i];
    if(p.samples_per_s!=null){pts.push({x:p.step,y:p.samples_per_s});continue;}
    const q=c[i-1];
    if(q&&q.t&&p.t){const dt=(new Date(p.t)-new Date(q.t))/1000;
      if(dt>10&&p.step>q.step){pts.push({x:p.step,y:(p.step-q.step)*2048/dt});}}
  }
  const note=document.getElementById("thr-note");
  if(!pts.length){if(charts["c-thr"]){charts["c-thr"].destroy();delete charts["c-thr"];}el.style.display="none";if(note)note.textContent="waiting for the next checkpoints";return;}
  el.style.display="";
  const avg=pts.reduce((a,b)=>a+b.y,0)/pts.length;
  if(note)note.textContent="derived from checkpoint timestamps (batch 2048) · avg "+(avg/1000).toFixed(1)+"k samples/s";
  if(charts["c-thr"])charts["c-thr"].destroy();
  charts["c-thr"]=new Chart(el,{type:"line",data:{labels:pts.map(p=>p.x),datasets:[
    {label:"samples/s",data:pts.map(p=>p.y),borderColor:"#3fa66a",pointRadius:0,borderWidth:1.5,fill:true,backgroundColor:"rgba(63,166,106,.08)"}]},
    options:{animation:false,maintainAspectRatio:false,interaction:{mode:"index",intersect:false},
      plugins:{legend:{display:false}},
      scales:{x:{ticks:{color:"#6f6f6c",maxTicksLimit:8,font:{size:10}},grid:{color:"#1a1a1a"}},
        y:{ticks:{color:"#6f6f6c",font:{size:10}},grid:{color:"#242423"}}}}});
}
function renderEvalTable(){
  const rows=evalSeries();const refs=snap.reference||[];const t=refs.filter(r=>r.ref)[0]||{};
  const head='<thead><tr><th>step</th><th>MATE</th><th>Δ</th><th>puzzles</th><th>Δ</th><th>gap→teacher</th><th></th></tr></thead>';
  let body="";
  rows.forEach((r,i)=>{const prev=rows[i-1];
    const dm=(prev&&prev.mate!=null&&r.mate!=null)?r.mate-prev.mate:null;
    const dp=(prev&&prev.puz!=null&&r.puz!=null)?r.puz-prev.puz:null;
    const gap=(t.puzzles!=null&&r.puz!=null)?r.puz-t.puzzles:null;
    const dcls=d=>d==null?"":(d>0?"delta good":(d<0?"delta bad":""));
    const df=(d)=>d==null?"—":(d>0?"+":"")+d.toFixed(2);
    const href="https://huggingface.co/datasets/vedangfake/chess-slm-benchmark/tree/main/"+String(r.dir||"").replace("/eval-summary.json","");
    body+="<tr><td>"+fmt(r.step)+"</td><td>"+(r.mate!=null?r.mate.toFixed(2)+"%":"—")+
      '</td><td class="'+dcls(dm)+'">'+df(dm)+"</td><td>"+(r.puz!=null?r.puz.toFixed(2)+"%":"—")+
      '</td><td class="'+dcls(dp)+'">'+df(dp)+"</td><td>"+(gap!=null?gap.toFixed(2):"—")+
      '</td><td><a class="dim" href="'+href+'" target="_blank" rel="noopener">hf ↗</a></td></tr>';});
  document.getElementById("evaltable").innerHTML=head+"<tbody>"+body+"</tbody>";
}
function renderCorpusDetail(){
  const g=corpusGateInfo();const c=g.c;
  const fmtEta=g.eta!=null?(g.eta/60).toFixed(0)+" min":"—";
  document.getElementById("corpus-detail").innerHTML=
    '<div class="card-head"><div class="card-title">corpus detail</div><div class="card-note mono">'+fmt(c.labeled_rows||0)+" / "+fmt(c.target_rows||0)+'</div></div>'+
    statRow([["rate",_curRate!=null?(_curRate/1000).toFixed(1)+"k rows/s":"—"],["eta to gate",fmtEta],
      ["shards to gate",g.gateLabel],["rows to gate",(g.remaining/1e6).toFixed(0)+"M"]])+
    '<div class="shardgrid">'+corpusTiles(c,g.done)+'</div>'+
    '<div class="chart-note">largest-first order · green = labeled, dark = extra headroom beyond the 920M gate · hover for rows</div>';
}
function renderQuotaDetail(){
  const q=snap.quota||{};const keys=Object.keys(q);
  const total=keys.reduce((a,k)=>a+(parseFloat(q[k])||0),0);
  const reset=new Date("2026-09-26T00:00:00Z");
  const hrsLeft=Math.max(0,(reset-Date.now())/36e5);
  const days=Math.floor(hrsLeft/24),hh=Math.floor(hrsLeft%24);
  document.getElementById("quota-detail").innerHTML=
    '<div class="card-head"><div class="card-title">gpu budget</div><div class="card-note mono">reset in '+days+"d "+hh+'h</div></div>'+
    statRow([["total left",total.toFixed(1)+"h"],["accounts",String(keys.length)],["avg",(total/Math.max(1,keys.length)).toFixed(1)+"h"]])+
    '<table class="tbl"><thead><tr><th>account</th><th>gpu left</th></tr></thead><tbody>'+
    keys.map(a=>"<tr><td>"+esc(a)+'</td><td class="mono">'+esc(q[a])+"h</td></tr>").join("")+"</tbody></table>";
}
function renderCharts(){
  const rows=evalRows();
  const topMate=rows.filter(r=>!r.ref&&r.mate!=null)[0];
  const topPuz=rows.filter(r=>!r.ref&&r.puz!=null).sort((a,b)=>b.puz-a.puz)[0];
  const c=snap.curve||[];const last=c.length?c[c.length-1]:null;
  document.getElementById("mate-top").textContent=topMate?topMate.mate.toFixed(2)+"%":"—";
  document.getElementById("puz-top").textContent=topPuz?topPuz.puz.toFixed(2)+"%":"—";
  document.getElementById("loss-top").textContent=last?(last.train.toFixed(3)+" / "+(last.dev!=null?last.dev.toFixed(3):"—")):"—";
  drawBar("c-mate","mate","#3987e5");drawBar("c-puz","puz","#3987e5");
  drawLoss("c-loss",false,0);drawLoss("c-loss2",logScale,smooth);
  drawAccuracy();drawComponents();drawThroughput();renderEvalTable();renderCorpusDetail();renderQuotaDetail();
}
function renderCkpt(){
  const c=(snap.curve||[]).slice().reverse();
  document.getElementById("ckpt-note").textContent=c.length+" checkpoints";
  document.getElementById("ckpttable").innerHTML='<thead><tr><th>step</th><th>train</th><th>dev</th><th>Δdev</th></tr></thead><tbody>'+
    c.map((p,i)=>{const prev=c[i+1];const d=(prev&&p.dev!=null&&prev.dev!=null)?(p.dev-prev.dev):null;
      return "<tr><td>"+fmt(p.step)+"</td><td>"+p.train.toFixed(4)+"</td><td>"+(p.dev!=null?p.dev.toFixed(4):"—")+
        '</td><td class="'+(d==null?"":(d<0?"delta good":"delta bad"))+'">'+(d==null?"—":(d>0?"+":"")+d.toFixed(4))+"</td></tr>";}).join("")+"</tbody>";
}
/* games */
function selectGame(i){
  const g=(snap.games||[])[i];
  if(!g){game=null;document.getElementById("board").innerHTML="";document.getElementById("movelist").innerHTML='<div class="empty" style="padding:12px">no games archived yet</div>';return;}
  game=g;ply=0;draw();
  document.getElementById("gameresult").textContent=g.white+" vs "+g.black+" — "+g.result+" · "+g.moves.length+" plies";
}
function draw(){
  if(!game)return;
  const start="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
  const fen=ply===0?start:game.moves[ply-1].fen;
  const el=document.getElementById("board");
  el.innerHTML=fenToBoard(fen);
  const mv=ply>0?game.moves[ply-1]:null;
  if(mv&&mv.from&&mv.to){const cellOf=sq=>{const f=sq.charCodeAt(0)-97,r=8-parseInt(sq[1],10);return el.querySelector('[data-sq="'+r+","+f+'"]');};
    const a=cellOf(mv.from),b=cellOf(mv.to);if(a)a.classList.add("from");if(b)b.classList.add("to");}
  const rows=[];
  for(let i=0;i<game.moves.length;i+=2){rows.push("<tr><td class=\\"idx\\">"+(i/2+1)+".</td><td>"+sanSpan(i)+"</td><td>"+sanSpan(i+1)+"</td></tr>");}
  document.getElementById("movelist").innerHTML="<table>"+rows.join("")+"</table>";
  document.getElementById("plyinfo").textContent="ply "+ply+"/"+game.moves.length;
  const cur=document.querySelector(".moves .san.on");if(cur)cur.scrollIntoView({block:"nearest"});
}
function sanSpan(i){if(i>=game.moves.length)return '<span class="dim">—</span>';
  return '<span class="san '+(i===ply-1?"on":"")+'" onclick="jump('+(i+1)+')">'+game.moves[i].san+"</span>";}
function jump(p){if(!game)return;ply=Math.max(0,Math.min(game.moves.length,p));draw();}
window.jump=jump;
/* tabs + wiring */
function switchTab(name){
  document.querySelectorAll(".tabs a").forEach(a=>a.classList.toggle("active",a.dataset.tab===name));
  ["overview","metrics","games"].forEach(t=>document.getElementById("tab-"+t).classList.toggle("hidden",t!==name));
  if(name!=="games")renderCharts();
}
document.querySelectorAll(".tabs a").forEach(a=>a.onclick=()=>{location.hash=a.dataset.tab;switchTab(a.dataset.tab);});
const initTab=(location.hash||"").replace("#","");if(["overview","metrics","games"].includes(initTab))switchTab(initTab);
window.addEventListener("hashchange",()=>{const t=(location.hash||"").replace("#","");if(["overview","metrics","games"].includes(t))switchTab(t);});
document.getElementById("b-prev").onclick=()=>jump(ply-1);
document.getElementById("b-next").onclick=()=>jump(ply+1);
document.getElementById("b-first").onclick=()=>jump(0);
document.getElementById("b-last").onclick=()=>jump(game?game.moves.length:0);
document.getElementById("b-play").onclick=()=>{
  if(timer){clearInterval(timer);timer=null;return;}
  timer=setInterval(()=>{if(!game||ply>=game.moves.length){clearInterval(timer);timer=null;return;}jump(ply+1);},700);
};
document.addEventListener("keydown",e=>{if(e.key==="ArrowLeft")jump(ply-1);if(e.key==="ArrowRight")jump(ply+1);});
document.getElementById("gamesel").onchange=e=>selectGame(+e.target.value);
document.getElementById("smooth").oninput=e=>{smooth=+e.target.value/100;document.getElementById("smoothv").textContent=smooth.toFixed(2);drawLoss("c-loss2",logScale,smooth);};
document.getElementById("lin").onclick=()=>{logScale=false;document.getElementById("lin").classList.add("active");document.getElementById("log").classList.remove("active");drawLoss("c-loss2",logScale,smooth);};
document.getElementById("log").onclick=()=>{logScale=true;document.getElementById("log").classList.add("active");document.getElementById("lin").classList.remove("active");drawLoss("c-loss2",logScale,smooth);};
async function load(){
  const r=await fetch("/api/snapshot?t="+Date.now(),{cache:"no-store"});
  snap=await r.json();
  updateCorpusRate();
  renderNav();renderNotices();renderRuns();renderStages();renderCorpus();renderInfra();renderCharts();renderCkpt();
  const sel=document.getElementById("gamesel");const cur=sel.value;
  sel.innerHTML=(snap.games||[]).map((g,i)=>'<option value="'+i+'">'+esc(g.white+" vs "+g.black+" ("+g.result+")")+"</option>").join("");
  if(snap.games&&snap.games.length){sel.value=(cur&&+cur<snap.games.length)?cur:0;selectGame(+sel.value);}else selectGame(-1);
  document.getElementById("foot").textContent="updated "+new Date(snap.updated_at||Date.now()).toLocaleString()+(snap.errors&&snap.errors.length?" · "+snap.errors.join(" · "):"");
}
load();setInterval(load,60000);
</script>
</body>
</html>`;
