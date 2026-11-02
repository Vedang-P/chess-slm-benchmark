import { HTML } from "./page.js";

// Live monitor for the cont1 CC-GAVN continuation. The GitHub supervisor posts a
// snapshot every tick; the Worker keeps the per-run curve history in KV and
// serves the merged view. A scheduled trigger also wakes the supervisor.
const KEY = "cont1_monitor_v1";
const GITHUB = "https://api.github.com/repos/Vedang-P/chess-slm-benchmark/dispatches";
const SECURITY = {
  "x-content-type-options": "nosniff",
  "referrer-policy": "no-referrer",
  "x-frame-options": "DENY",
  "content-security-policy": "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
};
const JSON_HEADERS = { ...SECURITY, "cache-control": "no-store" };

async function dispatchTick(env) {
  if (!env.GH_TOKEN) return;
  const previous = Number((await env.SNAPSHOT.get("last_tick")) || 0);
  if (Date.now() - previous < 4 * 60_000) return;
  const response = await fetch(GITHUB, {
    method: "POST",
    headers: { Authorization: "Bearer " + env.GH_TOKEN,
      Accept: "application/vnd.github+json", "User-Agent": "ccgavn-cont1-monitor" },
    body: JSON.stringify({ event_type: "tick" }),
  });
  if (!response.ok && response.status !== 204) throw new Error("GitHub dispatch HTTP " + response.status);
  await env.SNAPSHOT.put("last_tick", String(Date.now()));
}

const num = value => value == null || !Number.isFinite(Number(value)) ? null : Number(value);

function metricPoint(item) {
  const m = item.metrics || {};
  return { step: Number(item.step), train_loss: num(m.train_loss), dev_loss: num(m.dev_loss),
    dev_dist: num(m.dev_dist), val_pct: num(m.val_pct), val_solved: num(m.val_solved),
    tac_hit: num(m.tac_dev_hit), tac_regret: num(m.tac_dev_regret), lr: num(m.lr),
    grad_norm: num(m.grad_norm), samples_per_s: num(m.samples_per_s) };
}

function eventsFrom(previous, current) {
  const at = current.updated_at;
  const events = [];
  const add = (kind, title, detail, tone = "neutral") =>
    events.push({ id: `${at}:${kind}:${title}`, at, title, detail, tone });
  if (!previous?.updated_at) add("start", "Monitor started", "Tracking the cont1 continuation.");
  if (previous?.phase !== current.phase)
    add("phase", `Supervisor: ${current.phase}`, current.supervisor?.reason || "Stage changed.",
      ["blocked", "supervisor-error"].includes(current.phase) ? "bad" : "neutral");
  for (const [run, now] of Object.entries(current.runs || {})) {
    const before = previous?.runs?.[run];
    if ((now.step || 0) > (before?.step || 0))
      add("checkpoint", `${run}: checkpoint ${now.step.toLocaleString()}`,
        "Weights, EMA, optimizer and metrics saved to Hugging Face.", "good");
    const known = new Set((before?.evaluations || []).map(e => e.step));
    for (const e of now.evaluations || []) if (!known.has(e.step))
      add("eval", `${run}: official eval at ${e.step.toLocaleString()}`,
        `${e.puzzles.solved}/10,000 puzzles · ${e.mate.correct}/4,000 MATE`, "good");
    if (/^(HALTED|[A-Za-z]+Error)/.test(now.status || "") && now.status !== before?.status)
      add("failure", `${run}: ${now.status.split("\n")[0].slice(0, 80)}`, now.status.slice(0, 300), "bad");
  }
  return events;
}

async function ingest(env, body) {
  const prior = await env.SNAPSHOT.get(KEY, "json");
  const before = prior?.snapshot || null;
  if (before?.updated_at && body.updated_at < before.updated_at) return { ok: false, stale: true };
  const metrics = new Map(Object.entries(prior?.history?.metrics || {}));
  for (const [run, snap] of Object.entries(body.runs || {})) {
    const points = new Map((metrics.get(run) || []).map(p => [p.step, p]));
    for (const item of snap.recent || []) {
      const point = metricPoint(item);
      if (Number.isFinite(point.step) && point.step > 0) points.set(point.step, point);
    }
    metrics.set(run, [...points.values()].sort((a, b) => a.step - b.step).slice(-1200));
  }
  const observations = [...(prior?.history?.observations || []),
    { at: body.updated_at, phase: body.phase,
      steps: Object.fromEntries(Object.entries(body.runs || {}).map(([r, s]) => [r, s.step])) }].slice(-1000);
  const allEvents = [...(prior?.events || []), ...eventsFrom(before, body)];
  const events = [...new Map(allEvents.map(item => [item.id, item])).values()].slice(-120);
  await env.SNAPSHOT.put(KEY, JSON.stringify({ snapshot: body,
    history: { observations, metrics: Object.fromEntries(metrics) }, events }));
  return { ok: true, runs: Object.keys(body.runs || {}).length };
}

export default {
  async scheduled(_event, env, ctx) { ctx.waitUntil(dispatchTick(env)); },
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/api/snapshot" && request.method === "GET") {
      const bundle = await env.SNAPSHOT.get(KEY, "json");
      return Response.json({ ...(bundle?.snapshot || {}),
        history: bundle?.history || { observations: [], metrics: {} },
        events: bundle?.events || [] }, { headers: JSON_HEADERS });
    }
    if (url.pathname === "/api/ingest" && request.method === "POST") {
      if (!env.INGEST_KEY || request.headers.get("x-ingest-key") !== env.INGEST_KEY)
        return new Response("forbidden", { status: 403, headers: SECURITY });
      const body = await request.json().catch(() => null);
      if (!body || body.run !== "cont1" || !body.updated_at || !body.runs || !Array.isArray(body.accounts))
        return new Response("invalid cont1 snapshot", { status: 400, headers: SECURITY });
      const result = await ingest(env, body);
      return Response.json(result, { status: result.stale ? 409 : 200, headers: JSON_HEADERS });
    }
    if (url.pathname.startsWith("/api/")) return new Response("not found", { status: 404, headers: SECURITY });
    return new Response(HTML, { headers: { ...SECURITY,
      "content-type": "text/html; charset=utf-8", "cache-control": "no-cache" } });
  },
};
