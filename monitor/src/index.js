import { HTML } from "./page.js";

const GITHUB = "https://api.github.com/repos/Vedang-P/chess-slm-benchmark/dispatches";
const RUN = "ccgavn-rl-root-v3-seed0";
const MONITOR_KEY = "rl_monitor_v3";
const SECURITY = {
  "x-content-type-options": "nosniff",
  "referrer-policy": "no-referrer",
  "x-frame-options": "DENY",
  "content-security-policy": "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
};
const JSON_HEADERS = { ...SECURITY, "cache-control": "no-store" };
const MAX_OBSERVATIONS = 2016; // One week at a five minute cadence.
const MAX_METRICS = 500;
const MAX_EVENTS = 90;

async function dispatchTick(env) {
  if (!env.GH_TOKEN) return;
  const previous = Number((await env.SNAPSHOT.get("last_tick")) || 0);
  if (Date.now() - previous < 4 * 60_000) return;
  const response = await fetch(GITHUB, {
    method: "POST",
    headers: {
      Authorization: "Bearer " + env.GH_TOKEN,
      Accept: "application/vnd.github+json",
      "User-Agent": "ccgavn-rl-monitor",
    },
    body: JSON.stringify({ event_type: "tick" }),
  });
  if (!response.ok && response.status !== 204) {
    throw new Error("GitHub dispatch HTTP " + response.status);
  }
  await env.SNAPSHOT.put("last_tick", String(Date.now()));
}

function statusClass(value) {
  const text = String(value || "").toUpperCase();
  if (/\b(?:FAILED|ERROR|INCOMPLETE)\b/.test(text)) return "error";
  if (/\b(?:DONE|COMPLETE|SUCCEEDED)\b/.test(text)) return "done";
  if (/\bRUNNING\b/.test(text)) return "running";
  if (/\b(?:QUEUED|PENDING)\b/.test(text)) return "queued";
  return "waiting";
}

function eventsFrom(previous, current) {
  const events = [];
  const at = current.updated_at;
  const push = (type, title, detail, tone = "neutral") =>
    events.push({ id: at + ":" + type + ":" + title, at, type, title, detail, tone });
  if (!previous?.updated_at) push("system", "Monitor history started", "Tracking observed pipeline states from this point.");
  for (const label of current.labels) {
    const before = previous?.labels?.find(item => item.account === label.account);
    if (before && label.count > before.count) {
      push("labels", label.account + " uploaded labels",
        label.count.toLocaleString() + " / " + label.target.toLocaleString() +
        " exact roots (" + (label.count - before.count).toLocaleString() + " new).", "good");
    }
    if (before && label.kernel !== before.kernel) {
      push("worker", label.account + " verifier " + String(label.kernel).toLowerCase(),
        "Kaggle kernel status observed by the supervisor.", statusClass(label.kernel) === "error" ? "bad" : "neutral");
    }
    if (before && statusClass(label.status) === "error" && before.status !== label.status) {
      push("error", label.account + " verifier reported an error", label.status, "bad");
    }
  }
  for (const account of current.accounts) {
    const before = previous?.accounts?.find(item => item.account === account.account);
    if (before && account.learner !== before.learner) {
      push("worker", account.account + " RL worker " + String(account.learner).toLowerCase(),
        "Kaggle learner kernel status observed by the supervisor.",
        statusClass(account.learner) === "error" ? "bad" : "neutral");
    }
  }
  const oldStep = Number(previous?.training?.step || 0);
  const step = Number(current.training?.step || 0);
  if (step > oldStep) {
    push("checkpoint", "RL checkpoint " + step.toLocaleString(),
      "Complete weights, optimizer, and metrics available on Hugging Face.", "good");
  }
  if (previous && current.label_count >= current.label_target &&
      previous.label_count < previous.label_target) {
    push("milestone", "Verifier labels complete",
      current.label_count.toLocaleString() + " exact positions across five accounts.", "good");
  }
  if (previous && current.evaluation && !previous.evaluation) {
    push("result", "Frozen evaluation complete",
      "4,000 MATE positions and 10,000 puzzles scored.", "good");
  }
  if (previous && statusClass(current.training?.status) === "error" &&
      previous.training?.status !== current.training.status) {
    push("error", "RL worker reported an error",
      String(current.training.status).split("\n").at(-1), "bad");
  }
  if (previous && statusClass(current.evaluation_status) === "error" &&
      previous.evaluation_status !== current.evaluation_status) {
    push("error", "Evaluation reported an error",
      String(current.evaluation_status).split("\n").at(-1), "bad");
  }
  return events;
}

function observation(snapshot) {
  return {
    at: snapshot.updated_at,
    labels: snapshot.labels.map(item => Number(item.count || 0)),
    label_count: Number(snapshot.label_count || 0),
    step: Number(snapshot.training?.step || 0),
    phase: snapshot.phase,
  };
}

function metricPoint(item, observedAt) {
  const metrics = item.metrics || {};
  const step = Number(item.step);
  if (!Number.isFinite(step) || step <= 0) return null;
  const finite = value => value == null ? null : Number.isFinite(Number(value)) ? Number(value) : null;
  return {
    step,
    observed_at: observedAt,
    loss: finite(metrics.loss),
    pg: finite(metrics.pg),
    kl: finite(metrics.kl),
    value: finite(metrics.value),
    grad_norm: finite(metrics.grad_norm),
    amp_scale: finite(metrics.amp_scale),
    dev_top1: finite(metrics.dev?.top1),
    dev_regret: finite(metrics.dev?.mean_regret),
    dev_roots: finite(metrics.dev?.roots),
  };
}

async function ingest(env, body) {
  const bundle = await env.SNAPSHOT.get(MONITOR_KEY, "json");
  const [previous, oldHistory, oldEvents] = bundle
    ? [bundle.snapshot, bundle.history, bundle.events]
    : [null, null, null];
  if (previous?.updated_at && body.updated_at < previous.updated_at) {
    return { ok: false, stale: true };
  }
  const history = oldHistory || { observations: [], metrics: [] };
  const currentObservation = observation(body);
  const observations = history.observations || [];
  if (observations.at(-1)?.at !== currentObservation.at) observations.push(currentObservation);
  history.observations = observations.slice(-MAX_OBSERVATIONS);
  const metrics = new Map((history.metrics || []).map(item => [item.step, item]));
  for (const checkpoint of body.training.checkpoints || []) {
    const point = metricPoint(checkpoint, body.updated_at);
    if (point && !metrics.has(point.step)) metrics.set(point.step, point);
  }
  history.metrics = [...metrics.values()].sort((a, b) => a.step - b.step).slice(-MAX_METRICS);
  const events = [...(oldEvents || []), ...eventsFrom(previous, body)];
  const unique = [...new Map(events.map(item => [item.id, item])).values()].slice(-MAX_EVENTS);
  await env.SNAPSHOT.put(MONITOR_KEY, JSON.stringify({ snapshot: body, history, events: unique }));
  return { ok: true, observations: history.observations.length, metrics: history.metrics.length };
}

export default {
  async scheduled(_event, env, ctx) {
    ctx.waitUntil(dispatchTick(env));
  },
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/api/snapshot" && request.method === "GET") {
      const bundle = await env.SNAPSHOT.get(MONITOR_KEY, "json");
      const [snapshot, history, events] = bundle
        ? [bundle.snapshot, bundle.history, bundle.events]
        : [null, null, null];
      return Response.json({ ...(snapshot || {}), history: history || { observations: [], metrics: [] },
        events: events || [] }, { headers: JSON_HEADERS });
    }
    if (url.pathname === "/api/ingest" && request.method === "POST") {
      if (!env.INGEST_KEY || request.headers.get("x-ingest-key") !== env.INGEST_KEY) {
        return new Response("forbidden", { status: 403, headers: SECURITY });
      }
      const body = await request.json().catch(() => null);
      if (!body || body.run !== RUN || !Array.isArray(body.labels) ||
          !Array.isArray(body.accounts) || !body.training || !body.updated_at) {
        return new Response("invalid RL snapshot", { status: 400, headers: SECURITY });
      }
      const result = await ingest(env, body);
      return Response.json(result, { status: result.stale ? 409 : 200, headers: JSON_HEADERS });
    }
    if (url.pathname.startsWith("/api/")) {
      return new Response("not found", { status: 404, headers: SECURITY });
    }
    return new Response(HTML, { headers: { ...SECURITY,
      "content-type": "text/html; charset=utf-8", "cache-control": "no-cache" } });
  },
};
