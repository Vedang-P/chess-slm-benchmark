/* ChessBench — tournament monitor logic. Reads monitor/state.json + history.jsonl. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const fmtPct = (v) => (v === null || v === undefined || v === "" ? "—" : `${(v * 100).toFixed(1)}%`);
  const fmtNum = (v) => (v === null || v === undefined || v === "" ? "—" : String(v));
  const num = (v) => (typeof v === "number" ? v : null);
  const escapeHtml = (value) => String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");

  let state = null;
  let history = [];
  let timer = null;
  let clockTimer = null;
  let fetchFailed = false;
  let stateRequest = 0;
  let liveRequest = 0;
  let stateController = null;
  let liveController = null;
  const charts = {};

  // ---------------- status ----------------
  function status(kind, text) {
    const pill = $("statusPill");
    pill.className = "lamp " + kind;
    $("statusText").textContent = text;
  }

  // ---------------- data ----------------
  // worker-first endpoints; raw GitHub as fallback (cached ~5 min).
  const feedUrl = (kind) => {
    if (CONFIG.WORKER_BASE) {
      const map = { state: "/state", history: "/history", live: "/live" };
      return CONFIG.WORKER_BASE.replace(/\/$/, "") + map[kind];
    }
    return { state: CONFIG.STATE_URL, history: CONFIG.HISTORY_URL, live: CONFIG.LIVE_URL }[kind];
  };

  async function fetchText(url, signal) {
    const res = await fetch(url, { cache: "no-store", signal });
    if (!res.ok) throw new Error(`${res.status} ${url}`);
    return res.text();
  }

  async function fetchFeed(kind, signal) {
    try {
      return await fetchText(feedUrl(kind), signal);
    } catch (e) {
      if (!CONFIG.WORKER_BASE) throw e;
      const raw = { state: CONFIG.STATE_URL, history: CONFIG.HISTORY_URL, live: CONFIG.LIVE_URL }[kind];
      return await fetchText(raw, signal); // worker unreachable -> raw fallback
    }
  }

  async function load() {
    const request = ++stateRequest;
    if (stateController) stateController.abort();
    const controller = new AbortController();
    stateController = controller;
    try {
      const nextState = JSON.parse(await fetchFeed("state", controller.signal));
      if (request !== stateRequest) return;
      let nextHistory = history;
      try {
        const parsedHistory = (await fetchFeed("history", controller.signal)).trim().split("\n")
          .filter(Boolean).map((l) => JSON.parse(l));
        const runId = nextState.run_id || nextState.started_at;
        const runStart = parseTimestamp(nextState.started_at);
        nextHistory = parsedHistory.filter((entry) => {
          if (entry.run_id && runId) return entry.run_id === runId;
          const timestamp = parseTimestamp(entry.ts);
          return Number.isFinite(runStart) && Number.isFinite(timestamp) && timestamp >= runStart;
        });
      } catch (error) {
        if (controller.signal.aborted || request !== stateRequest) return;
        console.warn("history feed unavailable; keeping the last history", error);
      }
      if (request !== stateRequest) return;
      state = nextState;
      history = nextHistory;
      fetchFailed = false;
      const ageS = ageSeconds(state.updated_at);
      if (!Number.isFinite(ageS)) status("error", "invalid timestamp");
      else if (ageS < -5) status("error", "clock skew · feed is in the future");
      else if (ageS > Math.max(CONFIG.REFRESH_S * 4, 180)) status("stale", "stale · " + Math.round(ageS / 60) + "m ago");
      else status("live", "live · " + timeAgo(state.updated_at));
      render();
    } catch (e) {
      if (controller.signal.aborted || request !== stateRequest) return;
      fetchFailed = true;
      status("error", "no signal");
      if (!state) showNoSignal(true);
      console.warn(e);
    }
  }

  function showNoSignal(failed) {
    $("notice").hidden = false;
    $("notice").className = "notice" + (failed ? " error" : "");
    $("noticeTitle").textContent = failed ? "No signal from the monitor" : "Waiting for a run";
    $("noticeBody").textContent = failed
      ? "Could not reach the results feed. The page keeps retrying; check the worker and the public monitor repo if this persists."
      : "No run is publishing yet. Start a run with --live-push (run_mate_eval.py) or --monitor (run_suite.py) and results land here within a minute.";
    $("noticeMeta").textContent = "";
    $("noticeMeta").appendChild(document.createTextNode("auto-retrying every " + CONFIG.REFRESH_S + "s · "));
    const a = $("retryLink");
    a.href = "#";
    a.textContent = "retry now";
    a.onclick = (ev) => { ev.preventDefault(); load(); };
    $("noticeMeta").appendChild(a);
  }

  function timeAgo(iso) {
    const s = ageSeconds(iso);
    if (!Number.isFinite(s)) return "unknown age";
    if (s < -5) return `clock skew · ${Math.round(Math.abs(s))}s ahead`;
    if (s < 5) return "just now";
    if (s < 60) return `${Math.round(s)}s ago`;
    if (s < 3600) return `${Math.round(s / 60)}m ago`;
    return `${Math.round(s / 3600)}h ago`;
  }

  function parseTimestamp(value) {
    if (typeof value !== "string" || !value.trim()) return NaN;
    const text = value.trim();
    if (/(?:Z|[+-]\d\d:\d\d)$/i.test(text)) return Date.parse(text);
    // Legacy snapshots omit a timezone. Prefer the interpretation that is not
    // in the future, since Kaggle and the browser may use different zones.
    const local = Date.parse(text);
    const utc = Date.parse(`${text}Z`);
    if (local > Date.now() + 5000 && utc <= Date.now() + 5000) return utc;
    return local;
  }

  function ageSeconds(value) {
    const timestamp = parseTimestamp(value);
    return Number.isFinite(timestamp) ? (Date.now() - timestamp) / 1000 : NaN;
  }

  // ---------------- clock (the signature) ----------------
  function tickClock() {
    const el = $("clockDigits");
    const face = el.closest(".clock-face");
    const start = state && state.started_at ? parseTimestamp(state.started_at) : NaN;
    if (!Number.isFinite(start)) { el.textContent = "00:00:00"; return; }
    const s = Math.max(0, Math.floor((Date.now() - start) / 1000));
    const h = String(Math.floor(s / 3600)).padStart(2, "0");
    const m = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
    const sec = String(s % 60).padStart(2, "0");
    el.textContent = `${h}:${m}:${sec}`;
    const progress = state && state.progress ? state.progress : {};
    face.classList.toggle("running", state && !state.last_error && state.stage === "sweep" && progress.fraction < 1);
  }

  // ---------------- helpers ----------------
  const avg = (vals) => {
    const ok = vals.filter((v) => typeof v === "number");
    return ok.length ? ok.reduce((a, b) => a + b, 0) / ok.length : null;
  };

  const weightedAvg = (entries) => {
    const ok = entries.filter((entry) => typeof entry.value === "number" && entry.weight > 0);
    const weight = ok.reduce((sum, entry) => sum + entry.weight, 0);
    return weight ? ok.reduce((sum, entry) => sum + entry.value * entry.weight, 0) / weight : null;
  };

  function cellsFor(task, variant) {
    return (state.cells || []).filter((c) => (!task || c.task === task) && (!variant || c.variant === variant));
  }

  function winMetric(cell, field) { return cell && cell.win ? cell.win[field] ?? null : null; }

  function modelWinAvg(task, variant, field) {
    const per = {};
    for (const c of cellsFor(task, variant)) {
      const v = winMetric(c, field);
      if (typeof v === "number") (per[c.model] = per[c.model] || []).push({ value: v, weight: num(c.win && c.win.n) || 1 });
    }
    return Object.fromEntries(Object.entries(per).map(([m, entries]) => [m, weightedAvg(entries)]));
  }

  function modelMetricAvgWhere(models, predicate, field, condition = "win") {
    const per = {};
    for (const c of state.cells || []) {
      if (predicate && !predicate(c)) continue;
      const metric = c[condition] || {};
      const value = num(metric[field]);
      if (value !== null) (per[c.model] = per[c.model] || []).push({ value, weight: num(metric.n) || 1 });
    }
    return Object.fromEntries(
      models.map((model) => [model, weightedAvg(per[model] || [])]).filter(([, value]) => value !== null),
    );
  }

  function weightedCellMetricAvg(cells, field, condition = "win") {
    return weightedAvg(cells.map((cell) => {
      const metric = cell[condition] || {};
      return { value: num(metric[field]), weight: num(metric.n) || 1 };
    }));
  }

  // The monitor publishes two shapes of run. `run_kind` says which; older
  // snapshots are inferred from `mode`. Rendering a MATE run through the
  // sweep layout is what produced "LEGAL MOVE RATE 79%" for a task with no
  // legality, and four permanently blank cards.
  function runKind() {
    if (state && state.run_kind) return state.run_kind;
    if (state && state.mode === "mate") return "mate-selection";
    return "sweep";
  }
  const isMateRun = () => runKind() === "mate-selection";

  /* Snapshots published before the MATE metrics existed carried a single fake
     sweep "cell" whose `legal_rate` actually held accuracy. Read what those
     snapshots genuinely contain (n and accuracy) and leave everything the old
     payload never recorded as null, so the cards show "—" rather than a
     number derived from a field that meant something else. */
  function legacyMateStats() {
    const cell = ((state && state.cells) || [])[0];
    const win = cell && cell.win;
    if (!win || typeof win.compliance_strict !== "number") return null;
    const n = win.n || 0;
    return {
      legacy: true,
      n, n_attempted: n,
      accuracy: win.compliance_strict,
      correct: Math.round(win.compliance_strict * n),
      answered: null, answer_rate: null,
      wrong: null, no_answer: null, parse_error: null, api_error: null,
      picked_a: null, picked_b: null, truth_a: null, truth_b: null,
      accuracy_truth_a: null, accuracy_truth_b: null,
      no_answer_reasons: {},
      mean_latency_s: null, mean_output_tokens: null,
      mean_reasoning_tokens: null, positions_per_hour: null,
    };
  }

  const mateStats = () => (state && state.mate) || legacyMateStats();

  function progressOf() {
    const p = (state && state.progress) || {};
    return {
      done: p.done ?? p.cells_done ?? 0,
      total: p.total ?? p.cells_total ?? 0,
      failed: p.failed ?? p.cells_failed ?? 0,
      fraction: typeof p.fraction === "number" ? p.fraction : 0,
    };
  }

  /* Positions, not cells, are the number that actually moves while you watch:
     one sweep cell is 40+ positions and can sit still for many minutes.
     Older snapshots have no positions_* fields, so fall back to summing the
     per-cell n — that still tells you how many positions have been scored. */
  function positionsOf() {
    const p = (state && state.progress) || {};
    const summed = ((state && state.cells) || [])
      .reduce((sum, c) => sum + (num(c.win && c.win.n) || num(c.n) || 0), 0);
    const done = typeof p.positions_done === "number" ? p.positions_done : summed;
    const total = typeof p.positions_total === "number" ? p.positions_total : null;
    return { done, total, fraction: total ? done / total : null };
  }

  const fmtInt = (v) => (typeof v === "number" ? v.toLocaleString() : "—");
  const fmtDur = (min) => {
    if (typeof min !== "number" || !Number.isFinite(min) || min < 0) return null;
    if (min < 60) return `${Math.round(min)}m`;
    const h = Math.floor(min / 60);
    return `${h}h ${Math.round(min % 60)}m`;
  };

  // one scoreboard card
  function card({ label, value, sub, wide, progress }) {
    const bar = typeof progress === "number"
      ? `<div class="rule-track"><div class="rule-fill" style="width:${(progress * 100).toFixed(1)}%"></div></div>`
      : "";
    return `<div class="sb-cell${wide ? " sb-wide" : ""}">
      <span class="sb-label">${escapeHtml(label)}</span>
      <span class="sb-value${wide ? " sb-big" : ""}">${escapeHtml(value)}</span>
      <span class="sb-sub">${escapeHtml(sub ?? "")}</span>
      ${bar}
    </div>`;
  }

  function cellLabel(cell) {
    if (!cell) return "—";
    return `${cell.model} · ${cell.task} · ${cell.variant || "—"}`;
  }

  function bestOf(models, map) {
    const entries = models.map((m) => [m, map(m)]).filter(([, v]) => v !== null);
    entries.sort((a, b) => b[1] - a[1]);
    return entries;
  }

  // ---------------- render ----------------
  function render() {
    // "has data" depends on the run kind: a MATE run publishes no cells at
    // all, so gating on cells.length left it permanently on the no-signal
    // screen once the sweep-shaped payload went away.
    const m = mateStats();
    const hasData = !!state && (isMateRun()
      ? !!m && (m.n_attempted || 0) > 0
      : (state.cells || []).length > 0);
    $("notice").hidden = hasData;
    if (!hasData) {
      showNoSignal(false);
      $("scoreboard").innerHTML = card({
        label: "round", value: "IDLE",
        sub: state && state.mode ? `${state.mode} mode` : "no signal yet",
      });
      $("charts").innerHTML = "";
      $("errorBanner").hidden = true;
      $("clockDigits").textContent = "00:00:00";
      $("entrants").hidden = true;
      $("cellsBody").innerHTML = `<tr><td colspan="10" class="empty">nothing scored yet — awaiting the first push</td></tr>`;
      $("tableCount").textContent = "—";
      $("footRepo").textContent = state ? `repo: ${state.repo || "—"}` : "no signal";
      $("footRun").textContent = "—";
      return;
    }
    $("wordmarkSub").textContent = isMateRun()
      ? `MATE move selection · ${(state.models || []).join(", ") || "?"}`
      : `tactical sweep · ${(state.models || []).length} models`;
    renderScoreboard();
    renderError();
    renderEntrants();
    try { renderCharts(); } catch (e) { console.warn("charts failed:", e); }
    renderTable();
    $("footRepo").textContent = `repo: ${state.repo || "—"}`;
    $("footRun").textContent = isMateRun()
      ? `mate-selection · ${state.config && state.config.thinking_enabled ? "thinking on" : "direct mode"}`
      : `sweep · ${state.mode || "?"} mode`;
    $("rawLink").href = CONFIG.STATE_URL;
    tickClock();
    if (!clockTimer) clockTimer = setInterval(tickClock, 1000);
  }

  function renderScoreboard() {
    $("scoreboard").innerHTML = isMateRun() ? mateScoreboard() : sweepScoreboard();
  }

  function runCard() {
    const p = progressOf();
    const complete = p.fraction >= 1 || state.stage === "complete";
    const cur = state.current;
    // "SWEEP" is the sweep runner's internal stage name; a MATE run is not a
    // sweep and should not be labelled one.
    const stage = complete ? "COMPLETE" : isMateRun() ? "RUNNING" : (state.stage || "sweep").toUpperCase();
    const sub = isMateRun()
      ? `${(state.models || [])[0] || "?"} · ${state.config && state.config.thinking_enabled ? "thinking on" : "direct"}`
      : `${(state.models || []).length} models · ${state.mode || "?"}`;
    // the sweep cursor is only informative when there is more than one cell;
    // a MATE run would print the same model × task on every refresh
    const now = cur && !isMateRun()
      ? `<span class="now-line">now: ${escapeHtml(cur.model)} × ${escapeHtml(cur.task)}</span>`
      : "";
    return `<div class="sb-cell">
      <span class="sb-label">round</span>
      <span class="sb-value">${stage}</span>
      <span class="sb-sub">${escapeHtml(sub)}</span>
      ${now}
    </div>`;
  }

  // ---- MATE selection run -------------------------------------------------
  const OLD = "not recorded in this snapshot";

  function mateScoreboard() {
    const m = mateStats() || {};
    const p = progressOf();
    const eta = fmtDur(state.eta_min);
    const unanswered = (m.no_answer || 0) + (m.parse_error || 0);
    const ratio = (x, d) => (typeof x === "number" && d ? x / d : null);
    const bRate = ratio(m.picked_b, m.n);
    const truthBRate = ratio(m.truth_b, m.n);
    const cards = [
      runCard(),
      card({
        label: "positions", value: fmtInt(p.done), wide: true,
        sub: `of ${fmtInt(p.total)}${eta ? ` · eta ${eta}` : ""}`,
        progress: p.fraction,
      }),
      card({
        label: "accuracy vs expert", value: fmtPct(m.accuracy),
        sub: `${fmtInt(m.correct)} / ${fmtInt(m.n)} · 50% is chance`,
      }),
      card({
        label: "answer rate", value: fmtPct(m.answer_rate),
        sub: m.legacy ? OLD : unanswered
          ? `${fmtInt(unanswered)} unanswered · ${Object.entries(m.no_answer_reasons || {}).map(([k, v]) => `${k} ${v}`).join(", ") || "no reason recorded"}`
          : "every position answered",
      }),
      card({
        label: "choice bias", value: bRate === null ? "—" : `${(bRate * 100).toFixed(0)}% B`,
        sub: m.legacy ? OLD
          : `picked A ${fmtInt(m.picked_a)} · B ${fmtInt(m.picked_b)} · expert B ${truthBRate === null ? "—" : (truthBRate * 100).toFixed(0) + "%"}`,
      }),
      card({
        label: "accuracy by expert label",
        value: m.legacy ? "—" : `${fmtPct(m.accuracy_truth_a)} / ${fmtPct(m.accuracy_truth_b)}`,
        sub: m.legacy ? OLD : `truth A (n=${fmtInt(m.truth_a)}) / truth B (n=${fmtInt(m.truth_b)})`,
      }),
      card({
        label: "throughput",
        value: m.positions_per_hour ? `${Math.round(m.positions_per_hour)}/h` : "—",
        sub: m.legacy ? OLD : m.mean_latency_s ? `${m.mean_latency_s.toFixed(1)}s per position` : "measuring…",
      }),
      card({
        label: "tokens per answer",
        value: typeof m.mean_output_tokens === "number"
          ? Math.round(m.mean_output_tokens).toLocaleString() : "—",
        sub: m.legacy ? OLD : m.mean_reasoning_tokens
          ? `+ ${Math.round(m.mean_reasoning_tokens).toLocaleString()} reasoning`
          : "output tokens · no reasoning reported",
      }),
      card({
        label: "api errors",
        value: m.legacy ? "—" : fmtInt(m.api_error || 0),
        sub: m.legacy ? OLD
          : (m.api_error || 0) ? "gateway failures · excluded from accuracy" : "no transport failures",
      }),
    ];
    return cards.join("");
  }

  // ---- tactical sweep -----------------------------------------------------
  function sweepScoreboard() {
    const p = progressOf();
    const cells = state.cells || [];
    const legalCells = cells.filter((c) => typeof winMetric(c, "legal_rate") === "number");
    // variant-agnostic: these were pinned to the "grid" variant, which the
    // FEN-only study never produces, so every card read "—"
    const strictFor = (task) => avg(Object.values(modelWinAvg(task, null, "compliance_strict")));
    const nModels = (task) => Object.keys(modelWinAvg(task, null, "compliance_strict")).length;
    const eta = fmtDur(state.eta_min);
    const pos = positionsOf();
    return [
      runCard(),
      card({
        // the headline is POSITIONS SCORED: "cells completed 3" told you
        // almost nothing about how far a run had actually got
        label: "positions scored", value: fmtInt(pos.done), wide: true,
        sub: `${pos.total ? `of ${fmtInt(pos.total)} · ` : ""}`
          + `${fmtInt(p.done)} of ${fmtInt(p.total)} cells`
          + `${p.failed ? ` · ${p.failed} failed` : ""}${eta ? ` · eta ${eta}` : ""}`,
        progress: pos.fraction ?? p.fraction,
      }),
      card({
        label: "legal move rate",
        value: fmtPct(weightedCellMetricAvg(legalCells, "legal_rate")),
        sub: `${legalCells.length} cells`,
      }),
      card({ label: "mate-in-1 strict", value: fmtPct(strictFor("mate1-lichess")),
             sub: `${nModels("mate1-lichess")} models` }),
      card({ label: "mate-in-2 strict", value: fmtPct(strictFor("mate2-lichess")),
             sub: `${nModels("mate2-lichess")} models` }),
      card({ label: "stockfish top-1", value: fmtPct(strictFor("bestmove-8x8")),
             sub: `${nModels("bestmove-8x8")} models` }),
    ].join("");
  }

  function renderError() {
    const has = !!state.last_error;
    $("errorBanner").hidden = !has;
    $("errorText").textContent = has ? state.last_error : "";
  }

  function renderEntrants() {
    const models = state.models || [];
    // A one-model run has no leaderboard. The section used to render a
    // single "not started" row forever during every MATE run.
    if (isMateRun() || models.length < 2) {
      $("entrants").hidden = true;
      return;
    }
    $("entrants").hidden = false;
    const cells = state.cells || [];
    const byModel = modelMetricAvgWhere(models, () => true, "legal_rate");
    const ranked = bestOf(models, (m) => byModel[m] ?? null);
    const ordered = [
      ...ranked.map(([model]) => model),
      ...models.filter((model) => byModel[model] === undefined),
    ];
    const p = progressOf();
    const expectedPerModel = models.length ? (p.total || 0) / models.length : 0;
    const done = new Set(models.filter((model) => {
      const completed = cells.filter((c) => c.model === model && c.done).length;
      return expectedPerModel > 0 && completed >= expectedPerModel;
    }));
    $("entrantCount").textContent = `${cells.length} cells`;
    $("entrantRows").innerHTML = ordered.map((m, i) => {
      const cellsDone = cells.filter((c) => c.model === m && c.done).length;
      const frac = expectedPerModel ? Math.min(1, cellsDone / expectedPerModel) : 0;
      const score = byModel[m] ?? null;
      const safeModel = escapeHtml(m);
      return `
      <div class="entrant${done.has(m) ? " done" : ""}">
        <span class="entrant-rank">${score === null ? "—" : String(i + 1).padStart(2, "0")}</span>
        <span class="entrant-name" title="${safeModel}">${safeModel}</span>
        <span class="entrant-mark">${score === null ? "not started" : fmtPct(score) + " legal"}</span>
        <span class="entrant-prog" style="--p:${(frac * 100).toFixed(0)}%"></span>
      </div>`;
    }).join("");
  }

  // ---------------- charts ----------------
  // direct value labels on bars and line ends (data is read, not decoded)
  const valueLabels = {
    id: "valueLabels",
    afterDatasetsDraw(chart) {
      const ctx = chart.ctx;
      ctx.save();
      ctx.font = "9px 'IBM Plex Mono', monospace";
      ctx.textAlign = "center";
      for (let d = 0; d < chart.data.datasets.length; d++) {
        const ds = chart.data.datasets[d];
        const meta = chart.getDatasetMeta(d);
        for (let i = 0; i < ds.data.length; i++) {
          const v = ds.data[i];
          if (typeof v !== "number" || v <= 0.001) continue;
          const el = meta.data[i];
          if (!el) continue;
          ctx.fillStyle = "#9d968a";
          ctx.fillText((v * 100).toFixed(0) + "%", el.x, el.y - 4);
        }
      }
      ctx.restore();
    },
  };
  const endLabel = {
    id: "endLabel",
    afterDatasetsDraw(chart) {
      const meta = chart.getDatasetMeta(0);
      const last = meta.data[meta.data.length - 1];
      const ds = chart.data.datasets[0];
      const v = ds.data[ds.data.length - 1];
      if (!last || typeof v !== "number") return;
      const ctx = chart.ctx;
      ctx.save();
      ctx.font = "600 10px 'IBM Plex Mono', monospace";
      ctx.fillStyle = "#e5484d";
      ctx.textAlign = "left";
      ctx.fillText((v * 100).toFixed(1) + "%", last.x + 6, last.y + 3);
      ctx.restore();
    },
  };

  function sortedModels(models, map) {
    return bestOf(models, (m) => map[m] ?? null).map(([m]) => m);
  }

  const baseOpts = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: { labels: { color: "#9d968a", boxWidth: 6, boxHeight: 6, usePointStyle: true, font: { family: "IBM Plex Mono", size: 10 } } },
      tooltip: {
        backgroundColor: "#0d0c0b", borderColor: "#3d3934", borderWidth: 1,
        titleColor: "#e9e6df", bodyColor: "#9d968a", padding: 9, cornerRadius: 0,
        callbacks: { label: (c) => ` ${c.dataset.label}: ${(c.parsed.y * 100).toFixed(1)}%` },
      },
    },
    scales: {
      x: { ticks: { color: "#6c655a", font: { family: "IBM Plex Mono", size: 9.5 }, maxRotation: 40 }, grid: { display: false }, border: { display: false } },
      y: { beginAtZero: true, max: 1, ticks: { color: "#6c655a", font: { family: "IBM Plex Mono", size: 9.5 }, callback: (v) => (v * 100).toFixed(0) + "%" }, grid: { color: "rgba(255,255,255,0.05)" }, border: { display: false } },
    },
  };

  function mkChart(id, cfg) {
    if (charts[id]) charts[id].destroy();
    charts[id] = new Chart($(id).getContext("2d"), cfg);
  }

  function barDataset(label, data, color) {
    return { label, data, backgroundColor: color, borderColor: color, borderWidth: 1, borderRadius: 0, maxBarThickness: 26 };
  }

  function chartStatus(id, text) {
    const el = $(id);
    if (el) el.textContent = text;
  }

  function chartValues(models, predicate, field) {
    return modelMetricAvgWhere(models, predicate, field);
  }

  function orderedMetricModels(models, maps) {
    const valueFor = (model) => maps
      .map((map) => map[model])
      .find((value) => value !== undefined && value !== null) ?? -1;
    return models
      .filter((model) => maps.some((map) => map[model] !== undefined && map[model] !== null))
      .sort((a, b) => valueFor(b) - valueFor(a));
  }

  // The figure set depends on the run kind, so the markup is generated too.
  // Rebuild only when the layout changes — replacing the canvases on every
  // 15s poll would orphan the Chart.js instances bound to them.
  let chartLayout = null;

  function buildChartLayout(figures) {
    const key = figures.map((f) => f.id).join(",");
    if (key === chartLayout) return false;
    for (const id of Object.keys(charts)) { charts[id].destroy(); delete charts[id]; }
    $("charts").innerHTML = figures.map((f) => `
      <figure class="chart-figure${f.wide ? " chart-span2" : ""}">
        <figcaption>${escapeHtml(f.title)} <span class="fig-note">${escapeHtml(f.note)}</span></figcaption>
        <div class="chart-box"><canvas id="${f.id}"></canvas><p class="chart-status" id="${f.id}Status"></p></div>
      </figure>`).join("");
    chartLayout = key;
    return true;
  }

  const rateOpts = (xTitle) => ({
    ...baseOpts,
    scales: {
      ...baseOpts.scales,
      x: { ...baseOpts.scales.x, title: { display: !!xTitle, text: xTitle || "", color: "#6c655a", font: { family: "IBM Plex Mono", size: 9.5 } } },
    },
  });

  function renderCharts() {
    if (isMateRun()) return renderMateCharts();
    return renderSweepCharts();
  }

  // ---- MATE selection -----------------------------------------------------
  function renderMateCharts() {
    const m = mateStats() || {};
    if (m.legacy) {
      // a legacy snapshot has no per-outcome or A/B counts to plot
      buildChartLayout([{ id: "chartMateAccuracy", title: "Accuracy over the run", note: "vs 50% chance", wide: true }]);
      chartStatus("chartMateAccuracyStatus",
        `legacy snapshot: ${fmtPct(m.accuracy)} over ${fmtInt(m.n)} positions. `
        + "Choice distribution, outcome breakdown and B-preference need a run "
        + "published by the current runner.");
      if (typeof Chart === "undefined") return;
      const hist = history.filter((h) => typeof h.legal_avg === "number");
      mkChart("chartMateAccuracy", {
        type: "line",
        data: { labels: hist.map((h) => h.cells_done), datasets: [{
          label: "accuracy", data: hist.map((h) => h.legal_avg), borderColor: "#e5484d",
          borderWidth: 1.6, backgroundColor: "rgba(229,72,77,0.06)", fill: true,
          tension: 0.3, pointRadius: 0 }] },
        options: rateOpts("positions scored"),
        plugins: [endLabel],
      });
      return;
    }
    buildChartLayout([
      { id: "chartMateAccuracy", title: "Accuracy over the run", note: "vs 50% chance", wide: true },
      { id: "chartMateChoice", title: "Choice distribution", note: "model vs expert" },
      { id: "chartMateOutcome", title: "Outcome breakdown", note: "share of scored positions" },
      { id: "chartMateBias", title: "B-preference over the run", note: "drift check", wide: true },
    ]);

    const hist = history.filter((h) => typeof h.accuracy === "number");
    chartStatus("chartMateAccuracyStatus", hist.length
      ? `${hist.length} monitor samples · latest ${fmtPct(hist[hist.length - 1].accuracy)} over ${fmtInt(m.n)} positions`
      : "Waiting for the first monitor sample.");
    chartStatus("chartMateChoiceStatus", m.n
      ? `model picked B in ${fmtPct(m.n ? m.picked_b / m.n : null)} of ${fmtInt(m.n)} positions; the expert answer is B in ${fmtPct(m.n ? m.truth_b / m.n : null)}`
      : "Waiting for scored positions.");
    chartStatus("chartMateOutcomeStatus", m.n
      ? `${fmtInt(m.n)} scored · ${fmtInt(m.api_error || 0)} api errors excluded`
      : "Waiting for scored positions.");
    chartStatus("chartMateBiasStatus", hist.length
      ? "a flat line well away from the expert's B-rate means a position-independent preference, not chess reasoning"
      : "Waiting for the first monitor sample.");

    if (typeof Chart === "undefined") return;

    const chanceLine = {
      id: "chanceLine",
      afterDatasetsDraw(chart) {
        const y = chart.scales.y.getPixelForValue(0.5);
        const ctx = chart.ctx;
        ctx.save();
        ctx.setLineDash([3, 3]);
        ctx.strokeStyle = "rgba(233,230,223,0.35)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(chart.chartArea.left, y);
        ctx.lineTo(chart.chartArea.right, y);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "#6c655a";
        ctx.font = "9px 'IBM Plex Mono', monospace";
        ctx.textAlign = "left";
        ctx.fillText("chance", chart.chartArea.left + 4, y - 4);
        ctx.restore();
      },
    };

    mkChart("chartMateAccuracy", {
      type: "line",
      data: {
        labels: hist.map((h) => h.done ?? h.cells_done),
        datasets: [
          { label: "accuracy", data: hist.map((h) => h.accuracy), borderColor: "#e5484d",
            borderWidth: 1.6, backgroundColor: "rgba(229,72,77,0.06)", fill: true, tension: 0.3, pointRadius: 0 },
          { label: "answer rate", data: hist.map((h) => h.answer_rate ?? null), borderColor: "rgba(90,171,130,0.9)",
            borderWidth: 1.2, borderDash: [4, 3], fill: false, tension: 0.3, pointRadius: 0 },
        ],
      },
      options: rateOpts("positions scored"),
      plugins: [chanceLine, endLabel],
    });

    mkChart("chartMateChoice", {
      type: "bar",
      data: {
        labels: ["candidate A", "candidate B"],
        datasets: [
          barDataset("model picked", [m.n ? m.picked_a / m.n : 0, m.n ? m.picked_b / m.n : 0], "rgba(242,169,59,0.8)"),
          barDataset("expert answer", [m.n ? m.truth_a / m.n : 0, m.n ? m.truth_b / m.n : 0], "rgba(233,230,223,0.5)"),
        ],
      },
      options: baseOpts,
      plugins: [valueLabels],
    });

    const n = m.n || 1;
    mkChart("chartMateOutcome", {
      type: "bar",
      data: {
        labels: ["correct", "wrong", "no answer", "parse error"],
        datasets: [barDataset("share", [
          (m.correct || 0) / n, (m.wrong || 0) / n, (m.no_answer || 0) / n, (m.parse_error || 0) / n,
        ], "rgba(90,171,130,0.75)")],
      },
      options: { ...baseOpts, plugins: { ...baseOpts.plugins, legend: { display: false } } },
      plugins: [valueLabels],
    });

    const biasHist = history.filter((h) => typeof h.picked_b_rate === "number");
    mkChart("chartMateBias", {
      type: "line",
      data: {
        labels: biasHist.map((h) => h.done ?? h.cells_done),
        datasets: [
          { label: "model picks B", data: biasHist.map((h) => h.picked_b_rate), borderColor: "#f2a93b",
            borderWidth: 1.6, backgroundColor: "rgba(242,169,59,0.07)", fill: true, tension: 0.3, pointRadius: 0 },
          { label: "expert B rate", data: biasHist.map(() => (m.n ? m.truth_b / m.n : null)),
            borderColor: "rgba(233,230,223,0.45)", borderWidth: 1, borderDash: [4, 3], fill: false, pointRadius: 0 },
        ],
      },
      options: rateOpts("positions scored"),
    });
  }

  // ---- tactical sweep -----------------------------------------------------
  function renderSweepCharts() {
    buildChartLayout([
      { id: "chartLegal", title: "Parsing and legality", note: "completed position cells", wide: true },
      { id: "chartTactics", title: "Tactics pipeline", note: "parsed · legal · strict score" },
      { id: "chartStock", title: "Move strength", note: "bestmove task · Stockfish" },
      { id: "chartHistory", title: "Progress over time", note: "average legal rate", wide: true },
    ]);

    const models = state.models || [];
    const position = (c) => c.win && Object.keys(c.win).length > 0;
    const positionCells = (state.cells || []).filter(position);

    const legParsed = chartValues(models, position, "parse_rate");
    const legLegal = chartValues(models, position, "legal_rate");
    const legOrder = orderedMetricModels(models, [legLegal, legParsed]);
    chartStatus("chartLegalStatus", positionCells.length
      ? `${positionCells.length} completed cells · parsed ${fmtPct(weightedCellMetricAvg(positionCells, "parse_rate"))} · legal ${fmtPct(weightedCellMetricAvg(positionCells, "legal_rate"))}`
      : "Waiting for completed position cells.");

    const tactical = (c) => position(c) && (c.task || "").startsWith("mate");
    const tacticalCells = (state.cells || []).filter(tactical);
    const tacParsed = chartValues(models, tactical, "parse_rate");
    const tacLegal = chartValues(models, tactical, "legal_rate");
    const m1 = chartValues(models, (c) => tactical(c) && c.task === "mate1-lichess", "compliance_strict");
    const m2 = chartValues(models, (c) => tactical(c) && c.task === "mate2-lichess", "compliance_strict");
    const tacOrder = orderedMetricModels(models, [m1, m2, tacLegal, tacParsed]);
    const decidedTactical = tacticalCells.filter((c) => typeof c.win.compliance_of_legal === "number").length;
    chartStatus("chartTacticsStatus", tacticalCells.length
      ? `${tacticalCells.length} completed cells · ${decidedTactical} with a legal answer · strict score counts rejected answers as 0`
      : "Waiting for mate-in-1 and mate-in-2 cells.");

    const stockfish = (c) => position(c) && c.task === "bestmove-8x8";
    const stockCells = (state.cells || []).filter(stockfish);
    const stockParsed = chartValues(models, stockfish, "parse_rate");
    const stockLegal = chartValues(models, stockfish, "legal_rate");
    const stockTop = chartValues(models, stockfish, "compliance_strict");
    const stockOrder = orderedMetricModels(models, [stockTop, stockLegal, stockParsed]);
    chartStatus("chartStockStatus", stockCells.length
      ? `${stockCells.length} completed cells · top-1 is strict over all samples · reference is Stockfish`
      : "Waiting for bestmove-8x8 cells.");

    chartStatus("chartHistoryStatus", history.length
      ? `${history.length} monitor samples · latest legal rate ${fmtPct(history[history.length - 1].legal_avg)}`
      : "Waiting for the first monitor sample.");

    if (typeof Chart === "undefined") return; // CDN blocked — statuses still explain the data

    mkChart("chartLegal", {
      type: "bar",
      data: { labels: legOrder, datasets: [
        barDataset("parsed", legOrder.map((m) => legParsed[m] ?? 0), "rgba(233,230,223,0.55)"),
        barDataset("legal", legOrder.map((m) => legLegal[m] ?? 0), "rgba(90,171,130,0.75)"),
      ] },
      options: baseOpts,
      plugins: [valueLabels],
    });

    mkChart("chartTactics", {
      type: "bar",
      data: { labels: tacOrder, datasets: [
        barDataset("parsed", tacOrder.map((m) => tacParsed[m] ?? 0), "rgba(233,230,223,0.45)"),
        barDataset("legal", tacOrder.map((m) => tacLegal[m] ?? 0), "rgba(242,169,59,0.7)"),
        barDataset("mate-in-1 strict", tacOrder.map((m) => m1[m] ?? 0), "rgba(229,72,77,0.75)"),
        barDataset("mate-in-2 strict", tacOrder.map((m) => m2[m] ?? 0), "rgba(90,171,130,0.7)"),
      ] },
      options: baseOpts,
      plugins: [valueLabels],
    });

    mkChart("chartStock", {
      type: "bar",
      data: { labels: stockOrder, datasets: [
        barDataset("parsed", stockOrder.map((m) => stockParsed[m] ?? 0), "rgba(233,230,223,0.45)"),
        barDataset("legal", stockOrder.map((m) => stockLegal[m] ?? 0), "rgba(90,171,130,0.7)"),
        barDataset("top-1 vs Stockfish", stockOrder.map((m) => stockTop[m] ?? 0), "rgba(242,169,59,0.8)"),
      ] },
      options: baseOpts,
      plugins: [valueLabels],
    });

    mkChart("chartHistory", {
      type: "line",
      data: {
        labels: history.map((h) => h.cells_done),
        datasets: [{
          label: "avg legal rate",
          data: history.map((h) => h.legal_avg),
          borderColor: "#e5484d",
          borderWidth: 1.5,
          backgroundColor: "rgba(229,72,77,0.06)",
          fill: true,
          tension: 0.3,
          pointRadius: 0,
        }],
      },
      options: rateOpts("cells completed"),
      plugins: [endLabel],
    });
  }

  // ---------------- table ----------------
  // column headers follow the run kind: a MATE choice has no legality, and a
  // sweep cell has no A/B split. One fixed header row meant every run showed
  // at least one column that could never have a value.
  const TABLE_HEADS = {
    "mate-selection": ["#", "model", "slice", "form", "what it measures", "n",
                       "answered", "picked B", "accuracy", ""],
    sweep: ["#", "model", "task", "form", "note", "n",
            "parsed", "legal", "strict score", ""],
  };

  function renderTable() {
    const heads = TABLE_HEADS[isMateRun() ? "mate-selection" : "sweep"];
    $("cellsHead").innerHTML = heads
      .map((h, i) => `<th${i === 0 || i >= 5 ? ' class="num"' : ""}>${escapeHtml(h)}</th>`)
      .join("");
    if (isMateRun()) return renderMateTable();
    return renderSweepTable();
  }

  function renderMateTable() {
    const m = mateStats() || {};
    $("tableCount").textContent = `${fmtInt(m.n)} scored positions`;
    const rate = (x, d) => (typeof x === "number" && d ? x / d : null);
    if (m.legacy) {
      $("cellsBody").innerHTML = `<tr>
        <td class="num mono dim">01</td>
        <td class="mono">${escapeHtml((state.models || [])[0] || "—")}</td>
        <td>overall</td>
        <td><span class="dim">strategy</span></td>
        <td><span class="dim">legacy snapshot — only n and accuracy were published</span></td>
        <td class="num mono">${fmtInt(m.n)}</td>
        <td class="num mono dim">—</td>
        <td class="num mono dim">—</td>
        <td class="num mono ${m.accuracy > 0.55 ? "pos" : m.accuracy >= 0.45 ? "warn" : "neg"}">${fmtPct(m.accuracy)}</td>
        <td><span class="cell-done">${state.stage === "complete" ? "done" : "running"}</span></td>
      </tr>`;
      return;
    }
    const rows = [
      ["overall", m.n, m.answer_rate, rate(m.picked_b, m.n), m.accuracy,
       "every scored position"],
      ["expert answer = A", m.truth_a, null, null, m.accuracy_truth_a,
       "accuracy where MoveA is the expert choice"],
      ["expert answer = B", m.truth_b, null, null, m.accuracy_truth_b,
       "accuracy where MoveB is the expert choice"],
      ["no answer / unparseable", (m.no_answer || 0) + (m.parse_error || 0), null, null, null,
       "the model produced nothing scorable"],
      ["api errors", m.api_error || 0, null, null, null,
       "gateway failures — excluded from every rate above"],
    ];
    $("cellsBody").innerHTML = rows.map(([label, n, answered, bRate, acc, note], i) => `
      <tr>
        <td class="num mono dim">${String(i + 1).padStart(2, "0")}</td>
        <td class="mono">${escapeHtml((state.models || [])[0] || "—")}</td>
        <td>${escapeHtml(label)}</td>
        <td><span class="dim">strategy</span></td>
        <td><span class="dim">${escapeHtml(note)}</span></td>
        <td class="num mono">${fmtInt(n)}</td>
        <td class="num mono ${typeof answered !== "number" ? "dim" : answered > 0.9 ? "pos" : "warn"}">${typeof answered === "number" ? fmtPct(answered) : "—"}</td>
        <td class="num mono ${typeof bRate !== "number" ? "dim" : Math.abs(bRate - 0.5) > 0.1 ? "warn" : ""}">${typeof bRate === "number" ? fmtPct(bRate) : "—"}</td>
        <td class="num mono ${typeof acc !== "number" ? "dim" : acc > 0.55 ? "pos" : acc >= 0.45 ? "warn" : "neg"}">${typeof acc === "number" ? fmtPct(acc) : "—"}</td>
        <td>${i === 0 ? `<span class="cell-done">${state.stage === "complete" ? "done" : "running"}</span>` : ""}</td>
      </tr>`).join("");
  }

  function renderSweepTable() {
    const body = $("cellsBody");
    const cells = state.cells || [];
    $("tableCount").textContent = `${cells.length} cells`;
    if (!cells.length) {
      body.innerHTML = `<tr><td colspan="10" class="empty">no scored cells yet — awaiting the first push</td></tr>`;
      return;
    }
    const rows = [];
    let idx = 0;
    for (const c of cells) {
      idx += 1;
      const n = String(idx).padStart(2, "0");
      const m = c.win;
      if (!m || !Object.keys(m).length) continue;
      const parse = num(m.parse_rate), legal = num(m.legal_rate), comp = num(m.compliance_strict);
      const attempted = num(m.n_attempted);
      const apiErr = num(m.api_error) || 0;
      rows.push(`<tr>
        <td class="num mono dim">${n}</td>
        <td class="mono">${escapeHtml(c.model)}</td>
        <td>${escapeHtml(c.task)}</td>
        <td><span class="dim">${escapeHtml(c.variant)}</span></td>
        <td><span class="dim">${apiErr ? `${apiErr} api errors excluded` : "scored"}</span></td>
        <td class="num mono">${fmtNum(m.n)}${attempted && attempted !== m.n ? `<span class="dim"> / ${attempted}</span>` : ""}</td>
        <td class="num mono ${parse > 0.5 ? "pos" : "neg"}">${fmtPct(parse)}</td>
        <td class="num mono ${legal > 0.3 ? "pos" : legal > 0 ? "warn" : "neg"}">${fmtPct(legal)}</td>
        <td class="num mono ${comp > 0.3 ? "pos" : comp > 0 ? "warn" : "dim"}">${fmtPct(comp)}</td>
        <td><span class="cell-done">done</span></td>
      </tr>`);
    }
    body.innerHTML = rows.join("") || `<tr><td colspan="10" class="empty">no scored cells yet</td></tr>`;
  }

  // ---------------- live board ----------------
  // Lichess cburnett SVG pieces (window.CHESS_PIECES from pieces.js).
  let live = null;
  let liveLatest = null;   // the newest published sample (sync target)
  let lastLiveKey = null;
  const replay = [];       // ordered history: replay[0] is the oldest
  let navigated = false;   // user browsing history manually

  function navHistory() {
    // ordered samples: replay (oldest..newest) + current live
    return [...replay, live].filter(Boolean);
  }

  function navIndex() {
    const list = navHistory();
    const i = list.indexOf(live);
    return i >= 0 ? i : list.length - 1;
  }

  function navTo(index) {
    const list = navHistory();
    if (index < 0 || index >= list.length) return;
    live = list[index];
    navigated = index < list.length - 1;
    renderLive();
    renderReplay();
  }

  function navSync() {
    if (liveLatest) { live = liveLatest; navigated = false; }
    else { navigated = false; }
    renderLive();
    renderReplay();
  }

  function updateNavButtons() {
    const list = navHistory();
    const i = navIndex();
    $("navPrev").disabled = i <= 0;
    $("navNext").disabled = i >= list.length - 1;
    $("navSync").classList.toggle("active", !navigated);
  }

  function fenPieces(fen) {
    if (!fen) return null;
    const place = (fen.split(" ")[0] || "").split("/");
    if (place.length !== 8) return null;
    const out = {};
    for (let r = 0; r < 8; r++) {
      let c = 0;
      for (const ch of place[r]) {
        if (/\d/.test(ch)) c += +ch;
        else { out[`${String.fromCharCode(97 + c)}${8 - r}`] = ch; c++; }
      }
    }
    return out;
  }

  function positionOf(sample) {
    return sample && sample.position ? sample.position : sample || {};
  }

  function listedPieces(sample) {
    const position = positionOf(sample);
    if (!Array.isArray(position.pieces)) return null;
    const out = {};
    for (const p of position.pieces) {
      if (!p || !p.sq || !p.kind || !p.color) continue;
      out[p.sq] = p.color === "w" ? p.kind : p.kind.toLowerCase();
    }
    return out;
  }

  function positionFen(sample) {
    const position = positionOf(sample);
    return position.fen || sample.fen || null;
  }

  function piecesMap(sample) {
    const listed = listedPieces(sample);
    const fen = fenPieces(positionFen(sample));
    const listedOk = listed && Object.keys(listed).length > 0;
    // Grid/list prompts are generated from the piece record; FEN prompts are
    // verified against that same record before rendering. An empty piece list
    // (e.g. FEN-only sources like MATE) must fall back to the FEN board.
    if (sample.cell && sample.cell.variant !== "fen" && listedOk) return listed;
    if (fen) return fen;
    if (listedOk) return listed;
    return null;
  }

  function mapSignature(map) {
    return Object.entries(map || {}).sort(([a], [b]) => a.localeCompare(b))
      .map(([sq, piece]) => `${sq}:${piece}`).join("|");
  }

  function renderedGrid(sample) {
    const position = positionOf(sample);
    const n = position.n || sample.n || 8;
    const pieces = listedPieces(sample) || {};
    const lines = ["   " + Array.from({ length: n }, (_, c) => String.fromCharCode(97 + c)).join(" ")];
    for (let r = n - 1; r >= 0; r--) {
      const row = [];
      for (let c = 0; c < n; c++) {
        const sq = `${String.fromCharCode(97 + c)}${r + 1}`;
        const piece = pieces[sq];
        row.push(piece ? `${piece === piece.toUpperCase() ? "w" : "b"}${piece.toUpperCase()}` : "..");
      }
      lines.push(`${String(r + 1).padStart(2, " ")} ${row.join(" ")}`);
    }
    return lines.join("\n");
  }

  function boardIntegrity(sample) {
    const errors = [];
    const fenMap = fenPieces(positionFen(sample));
    const pieceMap = listedPieces(sample);
    if (fenMap && pieceMap && mapSignature(fenMap) !== mapSignature(pieceMap)) {
      errors.push("FEN and piece-list snapshots disagree");
    }
    const prompt = sample.prompt || "";
    const variant = sample.cell && sample.cell.variant;
    if (variant === "fen") {
      const match = prompt.match(/The position in FEN notation:\s*([^\n]+)/);
      const promptFen = match && match[1].trim().split(/\s+/);
      const liveFen = positionFen(sample) && positionFen(sample).trim().split(/\s+/);
      if (promptFen && liveFen && (promptFen[0] !== liveFen[0] || promptFen[1] !== liveFen[1])) {
        errors.push("the FEN in the prompt differs from the board snapshot");
      }
    } else if (variant === "grid" && prompt && !prompt.includes(renderedGrid(sample))) {
      errors.push("the grid in the prompt differs from the piece snapshot");
    }
    return errors;
  }

  function renderBoard(sample) {
    const el = $("liveBoard");
    const n = sample && sample.n ? sample.n : 8;
    const pieces = piecesMap(sample) || {};
    const move = sample && sample.move ? sample.move : null;
    const fromSq = move ? move.slice(0, 2) : null;
    const toSq = move ? move.slice(2, 4) : null;
    const verdictClass = sample && sample.status === "legal"
      ? (sample.compliance ? "correct" : "wrong") : null;
    // Explicit rows keep empty ranks the same height as ranks with pieces.
    const track = `repeat(${n}, minmax(0, 1fr))`;
    el.style.gridTemplateColumns = track;
    el.style.gridTemplateRows = track;
    let html = "";
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        const sq = `${String.fromCharCode(97 + c)}${n - r}`;
        const piece = pieces[sq];
        const dark = (r + c) % 2 === 1;
        let cls = `board-cell ${dark ? "dark" : "light"}`;
        if (sq === fromSq || sq === toSq) cls += " last-move";
        if (sq === toSq && verdictClass === "correct") cls += " hl-to-correct";
        if (sq === toSq && verdictClass === "wrong") cls += " hl-to-wrong";
        const coords = [];
        if (c === 0) coords.push(`<span class="board-coord rank">${n - r}</span>`);
        if (r === n - 1) coords.push(`<span class="board-coord file">${String.fromCharCode(97 + c)}</span>`);
        const pieceHtml = piece && window.CHESS_PIECES
          ? `<span class="piece-svg">${window.CHESS_PIECES[(piece === piece.toUpperCase() ? "w" : "b") + piece.toUpperCase()] || ""}</span>`
          : "";
        html += `<div class="${cls}" data-sq="${sq}">${pieceHtml}${coords.join("")}</div>`;
      }
    }
    el.innerHTML = html;
    el.dataset.n = n;
    renderArrows(sample);
  }


  // lichess arrows — ported verbatim from chessground/src/svg.ts +
  // lila's analyse brushes. Straight line with round caps + an SVG marker
  // arrowhead, in square units (viewBox 0 0 8 8).
  const CG_BRUSHES = {
    green: { key: "g", color: "#15781B", opacity: 1, lineWidth: 10 },
    blue: { key: "b", color: "#003088", opacity: 1, lineWidth: 10 },
    paleGreen: { key: "q", color: "#15781B", opacity: 0.4, lineWidth: 8 },
    paleBlue: { key: "p", color: "#003088", opacity: 0.4, lineWidth: 8 },
  };
  const CG_ARROW_MARGIN = 10 / 64;
  const CG_NS = "http://www.w3.org/2000/svg";
  const cgLineWidth = (brush) => (brush.lineWidth || 10) / 64;
  const cgOpacity = (brush, pendingErase) =>
    (brush.opacity || 1) * (pendingErase ? 0.6 : 1);

  function sqCenterUnits(sq, n) {
    if (!/^[a-h][1-8]$/.test(sq || "")) return null;
    const file = sq.charCodeAt(0) - 97;
    const rank = +sq[1];
    // chessground coords: square centers in units, y down, rank 8 at top
    return [file + 0.5, n - rank + 0.5];
  }

  function ensureArrowDefs(svg) {
    let defs = svg.querySelector("defs");
    if (!defs) {
      defs = document.createElementNS(CG_NS, "defs");
      svg.appendChild(defs);
    }
    for (const brush of Object.values(CG_BRUSHES)) {
      if (defs.querySelector(`#arrowhead-${brush.key}`)) continue;
      const marker = document.createElementNS(CG_NS, "marker");
      marker.setAttribute("id", `arrowhead-${brush.key}`);
      marker.setAttribute("orient", "auto");
      marker.setAttribute("overflow", "visible");
      marker.setAttribute("markerWidth", "4");
      marker.setAttribute("markerHeight", "4");
      marker.setAttribute("refX", "2.05");
      marker.setAttribute("refY", "2");
      const path = document.createElementNS(CG_NS, "path");
      path.setAttribute("d", "M0,0 V4 L3,2 Z");
      path.setAttribute("fill", brush.color);
      marker.appendChild(path);
      defs.appendChild(marker);
    }
  }

  function cgArrowLine(from, to, brush) {
    // chessground renderArrow: line shortened by arrowMargin at the dest,
    // marker-end arrowhead, round caps
    const dx = to[0] - from[0];
    const dy = to[1] - from[1];
    const angle = Math.atan2(dy, dx);
    const xo = Math.cos(angle) * CG_ARROW_MARGIN;
    const yo = Math.sin(angle) * CG_ARROW_MARGIN;
    const line = document.createElementNS(CG_NS, "line");
    line.setAttribute("stroke", brush.color);
    line.setAttribute("stroke-width", String(cgLineWidth(brush)));
    line.setAttribute("stroke-linecap", "round");
    line.setAttribute("marker-end", `url(#arrowhead-${brush.key})`);
    line.setAttribute("opacity", String(cgOpacity(brush, false)));
    line.setAttribute("x1", String(from[0]));
    line.setAttribute("y1", String(from[1]));
    line.setAttribute("x2", String(to[0] - xo));
    line.setAttribute("y2", String(to[1] - yo));
    return line;
  }

  function oracleMoves(sample) {
    const o = (sample && sample.oracle) || {};
    const moves = [];
    if (o.best_move) moves.push(o.best_move);
    if (Array.isArray(o.mate_moves) && o.mate_moves.length) moves.push(o.mate_moves[0]);
    if (o.first_move && o.first_move !== o.best_move) moves.push(o.first_move);
    // MATE move-selection shape: truth_label picks the expert's candidate
    if (o.truth_label === "A" && o.candidate_a) moves.push(o.candidate_a);
    if (o.truth_label === "B" && o.candidate_b) moves.push(o.candidate_b);
    return [...new Set(moves)];
  }

  function clearArrows() {
    const overlay = $("boardArrows");
    if (overlay) overlay.innerHTML = "";
  }

  function renderArrows(sample) {
    let overlay = $("boardArrows");
    if (!overlay) {
      overlay = document.createElementNS(CG_NS, "svg");
      overlay.id = "boardArrows";
      overlay.setAttribute("class", "board-arrows");
      $("liveBoard").appendChild(overlay);
    }
    const n = sample && sample.n ? sample.n : 8;
    const modelMove = sample && sample.move ? sample.move : null;
    const oracle = oracleMoves(sample);
    overlay.setAttribute("viewBox", `0 0 ${n} ${n}`);
    overlay.setAttribute("width", "100%");
    overlay.setAttribute("height", "100%");
    overlay.innerHTML = "";
    ensureArrowDefs(overlay);

    // reference arrows (green; paled when the model picked the same move so
    // both stay visible, lichess-style)
    for (const m of oracle) {
      const from = sqCenterUnits(m.slice(0, 2), n);
      const to = sqCenterUnits(m.slice(2, 4), n);
      if (!from || !to) continue;
      const brush = m === modelMove ? CG_BRUSHES.paleGreen : CG_BRUSHES.green;
      overlay.appendChild(cgArrowLine(from, to, brush));
    }
    // model move (blue)
    if (modelMove) {
      const from = sqCenterUnits(modelMove.slice(0, 2), n);
      const to = sqCenterUnits(modelMove.slice(2, 4), n);
      if (from && to) overlay.appendChild(cgArrowLine(from, to, CG_BRUSHES.blue));
    }
  }

  function sampleDone(sample) {
    return sample && (sample.finished === true
      || ["legal", "illegal", "parse_error", "no_answer", "correct", "wrong"].includes(sample.status));
  }

  function verdictInfo(sample) {
    if (!sample || !sampleDone(sample)) {
      return { cls: "neutral", title: "waiting", detail: "the model is still generating" };
    }
    if (sample.status === "correct") {
      return { cls: "correct", title: "matches reference", detail: "the model chose the expert's move" };
    }
    if (sample.status === "wrong") {
      return { cls: "wrong", title: "chose the wrong candidate", detail: "the model's choice did not match the reference" };
    }
    if (sample.status === "legal") {
      if (sample.compliance === true) {
        return { cls: "correct", title: "matches reference", detail: "legal move and objective satisfied" };
      }
      if (sample.compliance === false) {
        return { cls: "wrong", title: "legal, not the reference", detail: "the move was legal but missed the task oracle" };
      }
      return { cls: "neutral", title: "legal move", detail: "this task has no objective match score" };
    }
    if (sample.status === "illegal") {
      return { cls: "wrong", title: "rejected as illegal", detail: "the answer cannot count as a legal move" };
    }
    if (sample.status === "parse_error") {
      return { cls: "neutral", title: "could not parse", detail: "the response did not contain a valid move" };
    }
    if (sample.status === "no_answer") {
      return { cls: "neutral", title: "no answer", detail: "the model returned no move" };
    }
    return { cls: "neutral", title: "unscored", detail: "no scoring result was published" };
  }

  function cellKey(cell) {
    return cell ? [cell.model, cell.task, cell.variant].join("|") : "";
  }

  function updateLiveSync() {
    const el = $("liveSync");
    if (!live) {
      el.className = "live-sync warn";
      el.textContent = "no sample published yet";
      return;
    }
    const ageS = live.updated_at ? ageSeconds(live.updated_at) : Infinity;
    const fresh = Number.isFinite(ageS) && ageS <= Math.max(CONFIG.LIVE_REFRESH_S * 4, 30);
    const record = live.record_id || live.position_id || "unknown";
    const complete = state && state.stage === "complete";
    // A MATE run has one task and one model: comparing the sample's "cell"
    // against a sweep cursor always disagreed and always showed the warn
    // state. Freshness is the only thing worth reporting here.
    if (isMateRun()) {
      el.className = "live-sync " + (fresh || complete ? "ok" : "warn");
      el.textContent = fresh
        ? `live · published ${timeAgo(live.updated_at)} · position ${record}`
        : complete
          ? `run finished · last scored position ${record} (${timeAgo(live.updated_at)})`
          : `feed is ${timeAgo(live.updated_at)} — the runner may have stalled · position ${record}`;
      return;
    }
    const sameCell = cellKey(state && state.current) === cellKey(live.cell || {});
    el.className = "live-sync " + (sameCell && fresh ? "ok" : "warn");
    el.textContent = sameCell && fresh
      ? `same sweep cell · published ${timeAgo(live.updated_at)} · board is record ${record}`
      : `board is ${timeAgo(live.updated_at)} · ${sameCell ? "same cell, stale sample" : `last published sample; sweep cursor is ${cellLabel(state && state.current)}`}`;
  }

  let reasoningSynced = true; // auto-follow the bottom of the thinking box

  function reasoningAtBottom() {
    const el = $("liveReasoning");
    if (!el) return true;
    return el.scrollHeight - el.scrollTop - el.clientHeight < 24;
  }

  function reasoningSyncToBottom() {
    const el = $("liveReasoning");
    if (el) el.scrollTop = el.scrollHeight;
    reasoningSynced = true;
    renderLive();
  }

  function renderLive() {
    const mate = isMateRun();
    const p = progressOf();
    $("liveNow").textContent = state && state.current
      ? (mate ? `scoring position ${fmtInt(p.done + 1)} of ${fmtInt(p.total)}` : `sweep now · ${cellLabel(state.current)}`)
      : (mate
        ? (state && state.stage === "complete" ? "run complete · showing the last scored position" : "between positions")
        : "no sweep cell in progress");
    if (!live) {
      $("liveCaption").textContent = "no live signal yet";
      $("liveSync").className = "live-sync warn";
      $("liveSync").textContent = "no published sample to compare with the sweep cursor";
      $("liveIntegrity").hidden = true;
      $("liveVerdict").hidden = true;
      return;
    }
    const cell = live.cell || {};
    const kind = live.task_kind || (live.oracle && live.oracle.kind) || null;
    $("liveMeta").textContent =
      `last sample ${live.sample_idx}${live.sample_total ? " / " + live.sample_total : ""} · ${live.phase || (sampleDone(live) ? "scored" : "generating")} · ${cell.task || "unknown task"} · ${cell.variant || "unknown representation"} · ${live.record_id || live.position_id || "unknown position"}`;
    updateLiveSync();

    $("livePromptLabel").textContent = live.model_input ? "exact model input" : "task prompt · model template not published";
    $("livePrompt").textContent = live.model_input || live.prompt || "No model input was published for this sample.";
    const reasoning = live.reasoning || "";
    const generating = live.phase === "generating" && !sampleDone(live);
    const thinkingLive = generating && reasoning.length > 0;

    // thinking box: autoscrolls ONLY while the reader is at the bottom
    // (or has hit the sync button); scrolling up freezes the viewport
    const reasonEl = $("liveReasoning");
    const wasAtBottom = reasoningAtBottom();
    $("liveReasoningLabel").textContent = thinkingLive
      ? `model thinking (live) · ${reasoning.length.toLocaleString()} chars`
      : reasoning
        ? `model thinking · ${reasoning.length.toLocaleString()} chars`
        : generating ? "model thinking · no tokens yet" : "model thinking";
    $("liveReasoning").textContent = reasoning || (generating ? "waiting for the first thought token…" : "no thinking published for this sample.");
    $("reasoningSync").classList.toggle("active", reasoningSynced && wasAtBottom);
    if (reasoningSynced && wasAtBottom) reasonEl.scrollTop = reasonEl.scrollHeight;

    // output box: the model's final answer
    $("liveGenerationLabel").textContent = live.output
      ? "model output (final answer)"
      : generating ? "model output · pending" : "model output";
    $("liveGenerationNote").textContent = generating
      ? (thinkingLive ? `live reasoning · refreshed every ~60s on the feed, box updates every 3s` : "thinking is in progress; the answer will appear when published")
      : live.output
        ? "the final answer after reasoning; the full chain of thought is preserved in the box above"
        : "reasoning ran to completion but no answer was emitted (flagged as a fallback if one was forced)";
    $("liveThinking").textContent = live.output || (sampleDone(live) ? "no answer was emitted" : "…");
    $("liveThinking").classList.toggle("thinking", !sampleDone(live));
    $("liveThinking").scrollTop = $("liveThinking").scrollHeight;

    const c = live.correct || {};
    const verdict = verdictInfo(live);
    const oracle = live.oracle || {};
    const referenceLabel = kind === "bestmove"
      ? "Stockfish reference"
      : kind === "mate1" || kind === "mate2"
        ? "mating reference"
        : kind === "mate_selection"
          ? "expert choice"
          : c.move
            ? "oracle reference"
            : "reference answer";
    $("liveReferenceLabel").textContent = referenceLabel;
    $("liveModelMove").textContent = live.move
      ? (kind === "mate_selection" && live.label ? `Move${live.label}: ${live.move}` : live.move)
      : "—";
    $("liveModelMove").classList.toggle("empty", !live.move);
    $("liveModelStatus").textContent = sampleDone(live)
      ? live.status === "legal" ? "parsed and legal" : (live.status || "unscored").replaceAll("_", " ")
      : "pending final answer";
    const expertCandidate = oracle.truth_label === "A" ? oracle.candidate_a
      : oracle.truth_label === "B" ? oracle.candidate_b : null;
    const refMove = expertCandidate || c.move || oracle.best_move
      || (Array.isArray(oracle.mate_moves) && oracle.mate_moves[0])
      || oracle.first_move || "—";
    $("liveStockMove").textContent = refMove;
    $("liveStockMove").classList.toggle("empty", !refMove);
    $("liveStockMove").title = c.note || oracle.cp != null ? `eval ${oracle.cp}cp` : "";
    $("liveReferenceNote").textContent =
      (kind === "mate_selection" && oracle.truth_label
        ? `expert picked Move${oracle.truth_label} · candidates A ${oracle.candidate_a || "?"} / B ${oracle.candidate_b || "?"}`
        : "")
      || c.note
      || (oracle.cp != null ? `Stockfish eval ${oracle.cp}cp (depth ${oracle.depth || "?"})` : "")
      || (kind === "mate1" ? "any move delivering checkmate wins" : "")
      || (kind === "mate2" ? "the first move of the forced mate line" : "")
      || "No reference move was published.";

    const vEl = $("liveVerdict");
    vEl.hidden = false;
    vEl.className = "live-verdict " + verdict.cls;
    vEl.innerHTML = `<strong></strong><span></span>`;
    vEl.querySelector("strong").textContent = verdict.title;
    vEl.querySelector("span").textContent = verdict.detail;
    $("liveDot").className = "live-dot" + (sampleDone(live) ? "" : " on");

    const integrityErrors = boardIntegrity(live);
    if (integrityErrors.length) {
      $("liveIntegrity").hidden = false;
      $("liveIntegrity").className = "live-integrity error";
      $("liveIntegrity").textContent = `BOARD HIDDEN · ${integrityErrors.join(" · ")}`;
      $("liveBoard").style.gridTemplateColumns = "1fr";
      $("liveBoard").style.gridTemplateRows = "1fr";
      $("liveBoard").innerHTML = `<div class="board-empty">Position data does not match the exact prompt. The dashboard will not show a potentially misleading board.</div>`;
      clearArrows();
      $("liveCaption").textContent = "board withheld until the monitor publishes a consistent sample";
      return;
    }
    $("liveIntegrity").hidden = false;
    $("liveIntegrity").className = "live-integrity ok";
    $("liveIntegrity").textContent = `position snapshot verified · ${live.record_id || live.position_id || "unknown record"}`;
    renderBoard(live);
    $("liveCaption").textContent = sampleDone(live) ? `completed · ${live.updated_at || ""}` : "generating…";
  }

  function renderReplay() {
    const el = $("liveReplay");
    if (!replay.length) { el.innerHTML = `<span class="live-caption">recently scored positions will appear here</span>`; return; }
    el.innerHTML = replay.slice().reverse().map((s, i) => {
      const v = verdictInfo(s);
      const mark = !v ? "·" : v.cls === "correct" ? "✓" : v.cls === "wrong" ? "✗" : "△";
      const active = s === live ? " active" : "";
      const label = isMateRun()
        ? escapeHtml(String(s.record_id || s.position_id || "").replace(/^mate-sel-/, "#"))
        : `${escapeHtml(String(s.cell && s.cell.model || "").split("-")[0])} · ${escapeHtml(s.cell && s.cell.task || "")}`;
      return `<button class="replay-chip${active}" data-i="${replay.length - 1 - i}">
        <span class="r-mark ${v ? v.cls : "warn"}">${mark}</span>
        <span>${label}</span>
      </button>`;
    }).join("");
  }

  async function loadLive() {
    const request = ++liveRequest;
    if (liveController) liveController.abort();
    const controller = new AbortController();
    liveController = controller;
    try {
      const l = JSON.parse(await fetchFeed("live", controller.signal));
      if (request !== liveRequest) return;
      const incomingTime = l.updated_at ? parseTimestamp(l.updated_at) : 0;
      const currentTime = live && live.updated_at ? parseTimestamp(live.updated_at) : 0;
      if (live && incomingTime && currentTime && incomingTime < currentTime) return;
      const key = `${l.cell ? l.cell.model + l.cell.task + l.cell.variant : ""}|${l.position_id || ""}|${l.sample_idx || ""}`;
      if (key !== lastLiveKey) {
        if (live && live.position_id && live !== l) replay.push(live);
        if (replay.length > 40) replay.shift();
        lastLiveKey = key;
        liveLatest = l;
        if (!navigated) live = l;
      } else if (
        l.updated_at !== live.updated_at
        || l.phase !== live.phase
        || l.status !== live.status
        || l.output !== live.output
        || l.reasoning !== live.reasoning
      ) {
        liveLatest = l;
        if (!navigated || live === liveLatest) live = l; // same position, refreshed content or lifecycle phase
      }
      renderLive();
      renderReplay();
    } catch (e) {
      if (controller.signal.aborted || request !== liveRequest) return;
      /* live is best-effort; the existing sample remains visible but ages */
    }
  }

  // ---------------- boot ----------------
  $("refreshBtn").addEventListener("click", load);
  $("navPrev").addEventListener("click", () => navTo(navIndex() - 1));
  $("navNext").addEventListener("click", () => navTo(navIndex() + 1));
  $("navSync").addEventListener("click", navSync);
  setInterval(updateNavButtons, 500);
  $("liveReasoning").addEventListener("scroll", () => {
    reasoningSynced = reasoningAtBottom();
    $("reasoningSync").classList.toggle("active", reasoningSynced);
  });
  $("reasoningSync").addEventListener("click", reasoningSyncToBottom);
  $("liveReplay").addEventListener("click", (ev) => {
    const chip = ev.target.closest(".replay-chip");
    if (!chip) return;
    const i = +chip.dataset.i;
    const list = navHistory();
    const target = list[list.length - 1 - i];
    if (target) { live = target; navigated = true; renderLive(); renderReplay(); }
  });
  document.addEventListener("visibilitychange", () => { if (!document.hidden) { load(); loadLive(); } });
  if (CONFIG.REFRESH_S > 0) timer = setInterval(load, CONFIG.REFRESH_S * 1000);
  if (CONFIG.LIVE_REFRESH_S > 0) setInterval(loadLive, CONFIG.LIVE_REFRESH_S * 1000);
  setInterval(updateLiveSync, 1000);
  load();
  loadLive();
})();
