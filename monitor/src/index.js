import { HTML } from "./page.js";

const RUN = "ccgavn-5m-puzzles-v1-seed0";
const KEY = "tactical_monitor_v1";
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
      Accept: "application/vnd.github+json", "User-Agent": "ccgavn-tactical-monitor" },
    body: JSON.stringify({ event_type: "tick" }),
  });
  if (!response.ok && response.status !== 204) throw new Error("GitHub dispatch HTTP " + response.status);
  await env.SNAPSHOT.put("last_tick", String(Date.now()));
}

function eventsFrom(previous, current) {
  const at = current.updated_at;
  const events = [];
  const add = (kind, title, detail, tone = "neutral") =>
    events.push({ id: `${at}:${kind}:${title}`, at, title, detail, tone });
  if (!previous?.updated_at) add("start", "Monitor started", "Tracking the frozen puzzle continuation.");
  if (previous?.phase !== current.phase) add("phase", `Stage: ${current.phase}`,
    current.supervisor?.reason || "Pipeline stage changed.",
    ["blocked", "supervisor-error"].includes(current.phase) ? "bad" : "neutral");
  if (current.demo?.complete && !previous?.demo?.complete) add("demo", "End-to-end demo passed",
    "Stockfish labels, 9M teacher, learner resume, and smoke evaluation completed.", "good");
  for (const [key, label] of [["root_shards", "Stockfish shards"], ["teacher_shards", "teacher shards"]]) {
    const oldCount = Number(previous?.corpus?.[key] || 0);
    const count = Number(current.corpus?.[key] || 0);
    if (count > oldCount) add(key, `${count.toLocaleString()} ${label}`,
      `${(count - oldCount).toLocaleString()} new complete shards.`, "good");
  }
  const oldStep = Number(previous?.training?.step || 0);
  const step = Number(current.training?.step || 0);
  if (step > oldStep && step > Number(current.training?.base_step || 0))
    add("checkpoint", `Checkpoint ${step.toLocaleString()}`,
      "Full model, optimizer, RNG, config, and metrics saved to Hugging Face.", "good");
  const known = new Set((previous?.evaluations || []).map(item => item.step));
  for (const result of current.evaluations || []) if (!known.has(result.step))
    add("evaluation", `Frozen evaluation at ${result.step.toLocaleString()}`,
      `${result.puzzles.solved}/10,000 puzzles · ${result.mate.correct}/4,000 MATE.`, "good");
  const status = String(current.training?.status || "");
  if (/\bFAILED\b/.test(status) && status !== previous?.training?.status)
    add("failure", "Learner reported a failure", status.slice(-350), "bad");
  return events;
}

function observation(s) {
  return { at: s.updated_at, phase: s.phase,
    roots: Number(s.corpus?.root_count || 0),
    teacher_roots: Number(s.corpus?.teacher_roots || 0),
    step: Number(s.training?.step || 0), evaluations: (s.evaluations || []).length };
}

function metricPoint(item) {
  const m = item.metrics || {};
  const num = value => value == null || !Number.isFinite(Number(value)) ? null : Number(value);
  return { step: Number(item.step), loss: num(m.train_loss), dev_loss: num(m.dev_loss),
    dev_dist: num(m.dev_dist), dev_ce: num(m.dev_ce), lr: num(m.lr),
    grad_norm: num(m.grad_norm), samples_per_s: num(m.samples_per_s) };
}

async function ingest(env, body) {
  const prior = await env.SNAPSHOT.get(KEY, "json");
  const before = prior?.snapshot || null;
  if (before?.updated_at && body.updated_at < before.updated_at) return { ok: false, stale: true };
  const oldHistory = prior?.history || { observations: [], metrics: [] };
  const observations = [...oldHistory.observations, observation(body)].slice(-2016);
  const points = new Map((oldHistory.metrics || []).map(item => [item.step, item]));
  for (const item of body.training.checkpoints || []) {
    const point = metricPoint(item);
    if (Number.isFinite(point.step) && point.step > 0) points.set(point.step, point);
  }
  const metrics = [...points.values()].sort((a, b) => a.step - b.step).slice(-1500);
  const allEvents = [...(prior?.events || []), ...eventsFrom(before, body)];
  const events = [...new Map(allEvents.map(item => [item.id, item])).values()].slice(-100);
  await env.SNAPSHOT.put(KEY, JSON.stringify({ snapshot: body,
    history: { observations, metrics }, events }));
  return { ok: true, observations: observations.length, metrics: metrics.length };
}

export default {
  async scheduled(_event, env, ctx) { ctx.waitUntil(dispatchTick(env)); },
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/api/snapshot" && request.method === "GET") {
      const bundle = await env.SNAPSHOT.get(KEY, "json");
      return Response.json({ ...(bundle?.snapshot || {}),
        history: bundle?.history || { observations: [], metrics: [] },
        events: bundle?.events || [] }, { headers: JSON_HEADERS });
    }
    if (url.pathname === "/api/ingest" && request.method === "POST") {
      if (!env.INGEST_KEY || request.headers.get("x-ingest-key") !== env.INGEST_KEY)
        return new Response("forbidden", { status: 403, headers: SECURITY });
      const body = await request.json().catch(() => null);
      if (!body || body.run !== RUN || !body.updated_at || !body.corpus ||
          !body.training || !Array.isArray(body.accounts) || !Array.isArray(body.evaluations))
        return new Response("invalid tactical snapshot", { status: 400, headers: SECURITY });
      const result = await ingest(env, body);
      return Response.json(result, { status: result.stale ? 409 : 200, headers: JSON_HEADERS });
    }
    if (url.pathname.startsWith("/api/")) return new Response("not found", { status: 404, headers: SECURITY });
    return new Response(HTML, { headers: { ...SECURITY,
      "content-type": "text/html; charset=utf-8", "cache-control": "no-cache" } });
  },
};
