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
      return Response.json(snap, { headers: { "cache-control": "no-store" } });
    }
    if (url.pathname === "/api/ingest" && req.method === "POST") {
      if (req.headers.get("x-ingest-key") !== env.INGEST_KEY) return new Response("forbidden", { status: 403 });
      const body = await req.json().catch(() => null);
      if (!body) return new Response("bad json", { status: 400 });
      const prior = (await env.SNAPSHOT.get("snapshot", "json")) || {};
      const snap = { ...prior };
      for (const k of ["curve", "evals", "corpus", "games", "runs", "kernels", "quota", "live"]) {
        if (k in body) snap[k] = body[k];
      }
      snap.reference = REFERENCE;
      snap.ingested_at = new Date().toISOString();
      snap.stages = computeStages(snap);
      appendNotices(prior, snap);
      snap.updated_at = snap.ingested_at;
      await env.SNAPSHOT.put("snapshot", JSON.stringify(snap));
      return Response.json({ ok: true, stages: snap.stages });
    }
    if (url.pathname === "/api/tick" && url.searchParams.get("key") === env.INGEST_KEY) {
      await env.SNAPSHOT.delete("last_tick");
      await dispatchTick(env);
      return Response.json({ ok: true });
    }
    if (url.pathname === "/api/refresh" && url.searchParams.get("key") === env.INGEST_KEY) {
      const s = await refresh(env);
      return Response.json({ ok: true, stages: s.stages });
    }
    return new Response(HTML, { headers: { "content-type": "text/html;charset=utf-8", "cache-control": "no-store" } });
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
  --board-l:#3a3a39; --board-d:#232322; --radius:6px; --nav-h:48px;
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
.chart-legend{display:flex;flex-wrap:wrap;gap:3px 14px;font-size:12px;color:var(--ink-2);padding-bottom:6px}
.legend-item{display:inline-flex;align-items:center;gap:6px}
.swatch{width:7px;height:7px;border-radius:2px}
/* board + games */
.boardwrap{display:grid;grid-template-columns:minmax(280px,440px) 1fr;gap:16px;align-items:start}
@media(max-width:820px){.boardwrap{grid-template-columns:1fr}}
.board{display:grid;grid-template-columns:repeat(8,1fr);aspect-ratio:1/1;border:1px solid var(--border);border-radius:4px;overflow:hidden}
.sq{display:flex;align-items:center;justify-content:center;font-size:min(6.2vw,32px)}
.sq.l{background:var(--board-l)}.sq.d{background:var(--board-d)}
.sq .pc.w{color:#ececea;text-shadow:0 1px 1px rgba(0,0,0,.55)}
.sq .pc.b{color:#8b8b88;text-shadow:0 1px 1px rgba(0,0,0,.45)}
.sq.from::after,.sq.to::after{content:"";position:absolute;inset:0;pointer-events:none}
.sq{position:relative}
.sq.from{box-shadow:inset 0 0 0 2px rgba(57,135,229,.55)}
.sq.to{box-shadow:inset 0 0 0 2px var(--accent)}
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
    </div>
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
const GLYPH={p:"♟",n:"♞",b:"♝",r:"♜",q:"♛",k:"♚"};
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
  if(ch){const w=ch===ch.toUpperCase();inner='<span class="pc '+(w?"w":"b")+'">'+(GLYPH[ch.toLowerCase()]||"")+"</span>";}
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
  const state=running?"running":(/DONE/i.test(status)?"complete":(/error|Traceback|Error/i.test(status)?"failed":"idle"));
  const c=snap.curve||[];const last=c.length?c[c.length-1]:null;
  let sum="",bar="",stats="";
  if(key.indexOf("ccgavn-5m-seed0")>=0&&last){
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
  return '<div class="run"><div class="run-head">'+
    '<span class="swatch" style="background:'+(running?"var(--live)":"var(--ink-3)")+'"></span>'+
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
function renderCorpus(){
  const c=snap.corpus||{};
  document.getElementById("corpus").innerHTML='<div class="card-head"><div class="card-title">1B corpus</div><div class="card-note mono">'+
    (c.shards_done||0)+" / "+(c.shards_planned||0)+' shards</div></div>'+
    statRow([["labeled",((c.labeled_rows||0)/1e6).toFixed(1)+"M"],["target",((c.target_rows||0)/1e6).toFixed(0)+"M"],
      ["existing",((c.original_rows||0)/1e6).toFixed(0)+"M"]]);
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
function drawLoss(id,log,sm){
  const el=document.getElementById(id);if(!el||!window.Chart)return;
  const c=snap.curve||[];
  const tr=smoothArr(c.map(p=>p.train),sm);
  const dv=smoothArr(c.filter(p=>p.dev!=null).map(p=>p.dev),sm);
  if(charts[id])charts[id].destroy();
  charts[id]=new Chart(el,{type:"line",data:{labels:c.map(p=>p.step),datasets:[
    {label:"train",data:tr,borderColor:"#3987e5",backgroundColor:"rgba(57,135,229,.08)",pointRadius:0,borderWidth:1.5,fill:true},
    {label:"dev",data:dv,borderColor:"#d9463f",pointRadius:0,borderWidth:1.5,spanGaps:true}]},
    options:{animation:false,maintainAspectRatio:false,interaction:{mode:"index",intersect:false},
      plugins:{legend:{labels:{color:"#a3a3a0",boxWidth:8,font:{size:11}}}},
      scales:{x:{ticks:{color:"#6f6f6c",maxTicksLimit:8,font:{size:10}},grid:{color:"#1a1a1a"}},
        y:{type:log?"logarithmic":"linear",ticks:{color:"#6f6f6c",font:{size:10}},grid:{color:"#242423"}}}}});
}
function renderCharts(){
  const rows=evalRows();
  const topMate=rows.filter(r=>!r.ref&&r.mate!=null)[0];
  const topPuz=rows.filter(r=>!r.ref&&r.puz!=null).sort((a,b)=>b.puz-a.puz)[0];
  const c=snap.curve||[];const last=c.length?c[c.length-1]:null;
  document.getElementById("mate-top").textContent=topMate?topMate.mate.toFixed(2)+"%":"—";
  document.getElementById("puz-top").textContent=topPuz?topPuz.puz.toFixed(2)+"%":"—";
  document.getElementById("loss-top").textContent=last?(last.train.toFixed(3)+" / "+(last.dev!=null?last.dev.toFixed(3):"—")):"—";
  drawBar("c-mate","mate","#3987e5");drawBar("c-puz","puzzles","#3987e5");
  drawLoss("c-loss",false,0);drawLoss("c-loss2",logScale,smooth);
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
document.querySelectorAll(".tabs a").forEach(a=>a.onclick=()=>switchTab(a.dataset.tab));
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
  const r=await fetch("/api/snapshot",{cache:"no-store"});
  snap=await r.json();
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
