/* ChessBench — tournament monitor logic. Reads monitor/state.json + history.jsonl. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const fmtPct = (v) => (v === null || v === undefined || v === "" ? "—" : `${(v * 100).toFixed(1)}%`);
  const fmtNum = (v) => (v === null || v === undefined || v === "" ? "—" : String(v));
  const num = (v) => (typeof v === "number" ? v : null);

  let state = null;
  let history = [];
  let timer = null;
  let clockTimer = null;
  let fetchFailed = false;
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

  async function fetchText(url) {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`${res.status} ${url}`);
    return res.text();
  }

  async function fetchFeed(kind) {
    try {
      return await fetchText(feedUrl(kind));
    } catch (e) {
      if (!CONFIG.WORKER_BASE) throw e;
      const raw = { state: CONFIG.STATE_URL, history: CONFIG.HISTORY_URL, live: CONFIG.LIVE_URL }[kind];
      return await fetchText(raw); // worker unreachable -> raw fallback
    }
  }

  async function load() {
    try {
      state = JSON.parse(await fetchFeed("state"));
      try {
        history = (await fetchFeed("history")).trim().split("\n")
          .filter(Boolean).map((l) => JSON.parse(l));
      } catch { history = []; }
      fetchFailed = false;
      const ageS = (Date.now() - new Date(state.updated_at).getTime()) / 1000;
      if (ageS > Math.max(CONFIG.REFRESH_S * 4, 180)) status("stale", "stale · " + Math.round(ageS / 60) + "m ago");
      else status("live", "live · " + timeAgo(state.updated_at));
      render();
    } catch (e) {
      fetchFailed = true;
      status("error", "no signal");
      if (!state) showNoSignal(true);
      console.warn(e);
    }
  }

  function showNoSignal(failed) {
    $("notice").hidden = false;
    $("notice").className = "notice" + (failed ? " error" : "");
    $("noticeTitle").textContent = failed ? "No signal from the monitor" : "Waiting for the sweep";
    $("noticeBody").textContent = failed
      ? "Could not reach the results feed (raw.githubusercontent.com). Check your connection — the feed is served by GitHub and the page keeps retrying."
      : "No results have been published yet. Data lands here every ~2 minutes once kaggle_run.ipynb is running on Kaggle.";
    $("noticeMeta").textContent = "";
    $("noticeMeta").appendChild(document.createTextNode("auto-retrying every " + CONFIG.REFRESH_S + "s · "));
    const a = $("retryLink");
    a.href = "#";
    a.textContent = "retry now";
    a.addEventListener("click", (ev) => { ev.preventDefault(); load(); });
    $("noticeMeta").appendChild(a);
  }

  function timeAgo(iso) {
    const s = (Date.now() - new Date(iso).getTime()) / 1000;
    if (s < 5) return "just now";
    if (s < 60) return `${Math.round(s)}s ago`;
    if (s < 3600) return `${Math.round(s / 60)}m ago`;
    return `${Math.round(s / 3600)}h ago`;
  }

  // ---------------- clock (the signature) ----------------
  function tickClock() {
    const el = $("clockDigits");
    const face = el.closest(".clock-face");
    const start = state && state.started_at ? new Date(state.started_at).getTime() : null;
    if (!start) { el.textContent = "00:00:00"; return; }
    const s = Math.max(0, Math.floor((Date.now() - start) / 1000));
    const h = String(Math.floor(s / 3600)).padStart(2, "0");
    const m = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
    const sec = String(s % 60).padStart(2, "0");
    el.textContent = `${h}:${m}:${sec}`;
    face.classList.toggle("running", state && !state.last_error && state.stage === "sweep");
  }

  // ---------------- helpers ----------------
  const avg = (vals) => {
    const ok = vals.filter((v) => typeof v === "number");
    return ok.length ? ok.reduce((a, b) => a + b, 0) / ok.length : null;
  };

  function cellsFor(task, variant) {
    return (state.cells || []).filter((c) => (!task || c.task === task) && (!variant || c.variant === variant));
  }

  function winMetric(cell, field) { return cell && cell.win ? cell.win[field] ?? null : null; }

  function modelWinAvg(task, variant, field) {
    const per = {};
    for (const c of cellsFor(task, variant)) {
      const v = winMetric(c, field);
      if (typeof v === "number") (per[c.model] = per[c.model] || []).push(v);
    }
    return Object.fromEntries(Object.entries(per).map(([m, vs]) => [m, avg(vs)]));
  }

  function modelMetricAvgWhere(models, predicate, field, condition = "win") {
    const per = {};
    for (const c of state.cells || []) {
      if (predicate && !predicate(c)) continue;
      const metric = c[condition] || {};
      const value = num(metric[field]);
      if (value !== null) (per[c.model] = per[c.model] || []).push(value);
    }
    return Object.fromEntries(
      models.map((model) => [model, avg(per[model] || [])]).filter(([, value]) => value !== null),
    );
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
    // no-signal notice: shown only when there is genuinely no data at all
    const hasData = !!state && (state.cells || []).length > 0;
    $("notice").hidden = hasData;
    if (!hasData) {
      showNoSignal(false);
      $("stageTag").textContent = "IDLE";
      $("runMeta").textContent = state && state.mode ? `${state.mode} mode` : "no signal yet";
      $("clockDigits").textContent = "00:00:00";
      $("entrantRows").innerHTML = `<div class="empty" style="padding:26px">no model scores yet — awaiting completed position cells</div>`;
      $("cellsBody").innerHTML = `<tr><td colspan="9" class="empty">no scored cells yet — awaiting the first push</td></tr>`;
      $("tableCount").textContent = "0 cells";
      $("footRepo").textContent = state ? `repo: ${state.repo || "—"}` : "no signal";
      return;
    }
    renderScoreboard();
    renderError();
    renderEntrants();
    try { renderCharts(); } catch (e) { console.warn("charts failed:", e); }
    renderTable();
    $("footRepo").textContent = `repo: ${state.repo || "—"} · ${state.mode || "?"} mode`;
    $("rawLink").href = CONFIG.STATE_URL;
    tickClock();
    if (!clockTimer) clockTimer = setInterval(tickClock, 1000);
  }

  function renderScoreboard() {
    const p = state.progress || {};
    const frac = p.fraction || 0;
    $("progressFill").style.width = `${(frac * 100).toFixed(1)}%`;
    $("stageTag").textContent = (state.stage || "sweep").toUpperCase();
    $("runMeta").textContent = `${(state.models || []).length} models · ${state.mode || "?"}`;
    const cur = state.current;
    $("currentCell").hidden = !cur;
    if (cur) $("currentCell").textContent = `now: ${cur.model} × ${cur.task}`;

    const cells = state.cells || [];
    const legalVals = cells.map((c) => winMetric(c, "legal_rate")).filter((v) => typeof v === "number");
    const mateVals = modelWinAvg("mate1-lichess", "grid", "compliance_strict");
    const stockVals = modelWinAvg("bestmove-8x8", "grid", "compliance_strict");
    const gameWins = cells.filter((c) => c.game && c.game.win_rate != null).map((c) => c.game.win_rate);

    $("kCells").textContent = `${p.cells_done ?? 0}`;
    $("kCellsSub").textContent = `of ${p.cells_total ?? "?"} cells`;
    $("kLegal").textContent = fmtPct(avg(legalVals));
    $("kLegalSub").textContent = `${legalVals.length} cells`;
    $("kTactics").textContent = fmtPct(avg(Object.values(mateVals)));
    $("kTacticsSub").textContent = `${Object.keys(mateVals).length} models · strict`;
    $("kStock").textContent = fmtPct(avg(Object.values(stockVals)));
    $("kStockSub").textContent = `${Object.keys(stockVals).length} models · strict`;
    $("kGames").textContent = fmtPct(avg(gameWins));
    $("kGamesSub").textContent = `${gameWins.length} game cells`;

    let gridVals = [], fenVals = [];
    for (const task of ["cap-legal-8x8", "mate1-lichess", "mate2-lichess", "bestmove-8x8"]) {
      for (const c of cellsFor(task, "grid")) { const v = winMetric(c, "legal_rate"); if (typeof v === "number") gridVals.push(v); }
      for (const c of cellsFor(task, "fen")) { const v = winMetric(c, "legal_rate"); if (typeof v === "number") fenVals.push(v); }
    }
    const delta = avg(gridVals) !== null && avg(fenVals) !== null ? avg(gridVals) - avg(fenVals) : null;
    $("kDivergence").textContent = delta === null ? "—" : (delta >= 0 ? "+" : "") + (delta * 100).toFixed(1) + "%";
  }

  function renderError() {
    const has = !!state.last_error;
    $("errorBanner").hidden = !has;
    $("errorText").textContent = has ? state.last_error : "";
  }

  function renderEntrants() {
    const models = state.models || [];
    const cells = state.cells || [];
    const byModel = modelMetricAvgWhere(models, (c) => !c.game, "legal_rate");
    const ranked = bestOf(models, (m) => byModel[m] ?? null);
    const ordered = [
      ...ranked.map(([model]) => model),
      ...models.filter((model) => byModel[model] === undefined),
    ];
    const expectedPerModel = models.length && state.progress
      ? (state.progress.cells_total || 0) / models.length
      : 0;
    const done = new Set(models.filter((model) => {
      const completed = cells.filter((c) => c.model === model && c.done).length;
      return expectedPerModel > 0 && completed >= expectedPerModel;
    }));
    if (!ordered.length) {
      $("entrantRows").innerHTML = `<div class="empty" style="padding:26px">no model scores yet — awaiting completed position cells</div>`;
      return;
    }
    $("entrantRows").innerHTML = ordered.map((m, i) => {
      const games = cells.filter((c) => c.model === m && c.done).length;
      const frac = expectedPerModel ? Math.min(1, games / expectedPerModel) : 0;
      const score = byModel[m] ?? null;
      return `
      <div class="entrant${done.has(m) ? " done" : ""}">
        <span class="entrant-rank">${score === null ? "—" : String(i + 1).padStart(2, "0")}</span>
        <span class="entrant-name" title="${m}">${m}</span>
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

  function renderCharts() {
    const models = state.models || [];
    const position = (c) => !c.game && c.win && Object.keys(c.win).length > 0;
    const positionCells = (state.cells || []).filter(position);

    const legParsed = chartValues(models, position, "parse_rate");
    const legLegal = chartValues(models, position, "legal_rate");
    const legOrder = orderedMetricModels(models, [legLegal, legParsed]);
    chartStatus(
      "chartLegalStatus",
      positionCells.length
        ? `${positionCells.length} completed cells · parsed ${fmtPct(avg(positionCells.map((c) => c.win.parse_rate)))} · legal ${fmtPct(avg(positionCells.map((c) => c.win.legal_rate)))}`
        : "Waiting for completed position cells.",
    );

    const tactical = (c) => position(c) && (c.task || "").startsWith("mate");
    const tacticalCells = (state.cells || []).filter(tactical);
    const tacParsed = chartValues(models, tactical, "parse_rate");
    const tacLegal = chartValues(models, tactical, "legal_rate");
    const m1 = chartValues(models, (c) => tactical(c) && c.task === "mate1-lichess", "compliance_strict");
    const m2 = chartValues(models, (c) => tactical(c) && c.task === "mate2-lichess", "compliance_strict");
    const tacOrder = orderedMetricModels(models, [m1, m2, tacLegal, tacParsed]);
    const decidedTactical = tacticalCells.filter((c) => typeof c.win.compliance_of_legal === "number").length;
    chartStatus(
      "chartTacticsStatus",
      tacticalCells.length
        ? `${tacticalCells.length} completed cells · ${decidedTactical} with a legal answer · strict score counts rejected answers as 0`
        : "Waiting for mate-in-1 and mate-in-2 cells.",
    );

    const stockfish = (c) => position(c) && c.task === "bestmove-8x8";
    const stockCells = (state.cells || []).filter(stockfish);
    const stockParsed = chartValues(models, stockfish, "parse_rate");
    const stockLegal = chartValues(models, stockfish, "legal_rate");
    const stockTop = chartValues(models, stockfish, "compliance_strict");
    const stockOrder = orderedMetricModels(models, [stockTop, stockLegal, stockParsed]);
    chartStatus(
      "chartStockStatus",
      stockCells.length
        ? `${stockCells.length} completed cells · top-1 is strict over all samples · reference is Stockfish`
        : "Waiting for bestmove-8x8 cells.",
    );

    chartStatus(
      "chartHistoryStatus",
      history.length
        ? `${history.length} monitor samples · latest legal rate ${fmtPct(history[history.length - 1].legal_avg)}`
        : "Waiting for the first monitor sample.",
    );

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
      options: { ...baseOpts, scales: { ...baseOpts.scales, x: { ...baseOpts.scales.x, title: { display: true, text: "cells completed", color: "#6c655a", font: { family: "IBM Plex Mono", size: 9.5 } } } } },
      plugins: [endLabel],
    });
  }

  // ---------------- table ----------------
  function renderTable() {
    const body = $("cellsBody");
    const cells = state.cells || [];
    $("tableCount").textContent = `${cells.length} cells`;
    if (!cells.length) {
      body.innerHTML = `<tr><td colspan="9" class="empty">no scored cells yet — awaiting the first push</td></tr>`;
      return;
    }
    const rows = [];
    let idx = 0;
    for (const c of cells) {
      idx += 1;
      const n = String(idx).padStart(2, "0");
      if (c.game) {
        const g = c.game;
        rows.push(`<tr>
          <td class="num mono dim">${n}</td>
          <td class="mono">${c.model}</td>
          <td>${c.task}</td>
          <td><span class="dim">${c.variant}</span></td>
          <td class="num mono">${fmtNum(g.n)}</td>
          <td class="num mono dim">—</td>
          <td class="num mono ${g.legal_rate > 0.3 ? "pos" : g.legal_rate > 0 ? "warn" : "neg"}">${fmtPct(g.legal_rate)}</td>
          <td class="num mono ${g.win_rate > 0.3 ? "pos" : g.win_rate > 0 ? "warn" : "dim"}">${fmtPct(g.win_rate)}</td>
          <td><span class="cell-done">done</span></td>
        </tr>`);
        continue;
      }
      const m = c.win;
      if (!m) continue;
      const parse = num(m.parse_rate), legal = num(m.legal_rate), comp = num(m.compliance_of_legal);
      rows.push(`<tr>
        <td class="num mono dim">${n}</td>
        <td class="mono">${c.model}</td>
        <td>${c.task}</td>
        <td><span class="dim">${c.variant}</span></td>
        <td class="num mono">${fmtNum(m.n)}</td>
        <td class="num mono ${parse > 0.5 ? "pos" : "neg"}">${fmtPct(parse)}</td>
        <td class="num mono ${legal > 0.3 ? "pos" : legal > 0 ? "warn" : "neg"}">${fmtPct(legal)}</td>
        <td class="num mono ${comp > 0.3 ? "pos" : comp > 0 ? "warn" : "dim"}">${fmtPct(comp)}</td>
        <td><span class="cell-done">done</span></td>
      </tr>`);
    }
    body.innerHTML = rows.join("") || `<tr><td colspan="9" class="empty">no scored cells yet</td></tr>`;
  }

  // ---------------- live board ----------------
  // Lichess cburnett SVG pieces (window.CHESS_PIECES from pieces.js).
  const GAME_TASKS = ["playout-5x5", "ttt", "c4"];
  let live = null;
  let lastLiveKey = null;
  const replay = [];

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
    // Grid/list prompts are generated from the piece record; FEN prompts are
    // verified against that same record before rendering.
    if (sample.cell && sample.cell.variant !== "fen" && listed) return listed;
    if (sample.cell && sample.cell.variant === "fen" && fen) return fen;
    if (listed) return listed;
    if (fen) return fen;
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
        html += `<div class="${cls}">${pieceHtml}${coords.join("")}</div>`;
      }
    }
    el.innerHTML = html;
  }

  function sampleDone(sample) {
    return sample && (sample.finished === true
      || ["legal", "illegal", "parse_error", "no_answer"].includes(sample.status));
  }

  function verdictInfo(sample) {
    if (!sample || !sampleDone(sample)) {
      return { cls: "neutral", title: "waiting", detail: "the model is still generating" };
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

  function renderLive() {
    const sweep = state && state.current ? `sweep now · ${cellLabel(state.current)}` : "sweep position unavailable";
    $("liveNow").textContent = sweep;
    if (!live) {
      $("liveCaption").textContent = "no live signal yet";
      $("liveSync").className = "live-sync warn";
      $("liveSync").textContent = "no published sample to compare with the sweep cursor";
      $("liveIntegrity").hidden = true;
      $("liveVerdict").hidden = true;
      return;
    }
    const cell = live.cell || {};
    const kind = live.task_kind;
    const isGame = GAME_TASKS.includes(cell.task);
    const sameCell = cellKey(state && state.current) === cellKey(cell);
    const ageS = live.updated_at ? (Date.now() - new Date(live.updated_at).getTime()) / 1000 : Infinity;
    const fresh = Number.isFinite(ageS) && ageS <= Math.max(CONFIG.LIVE_REFRESH_S * 4, 30);
    $("liveMeta").textContent =
      `last sample ${live.sample_idx}${live.sample_total ? " / " + live.sample_total : ""} · ${cell.task || "unknown task"} · ${cell.variant || "unknown representation"} · ${live.record_id || live.position_id || "unknown position"}`;
    $("liveSync").className = "live-sync " + (sameCell && fresh ? "ok" : "warn");
    $("liveSync").textContent = sameCell && fresh
      ? `same sweep cell · published ${timeAgo(live.updated_at)} · board is record ${live.record_id || live.position_id || "unknown"}`
      : `board is ${timeAgo(live.updated_at)} · ${sameCell ? "same cell, stale sample" : `last published sample; sweep cursor is ${cellLabel(state && state.current)}`}`;

    $("livePrompt").textContent = live.prompt || "No prompt was published for this sample.";
    const cot = live.cot_requested === true || /think step by step/i.test(live.prompt || "");
    $("liveGenerationLabel").textContent = cot ? "reasoning + final answer" : "model output · answer-only";
    $("liveGenerationNote").textContent = cot
      ? "raw generated text; no hidden reasoning is inferred"
      : "this run did not request chain-of-thought; the prompt requires an answer-only response";
    $("liveThinking").textContent = live.output || (sampleDone(live) ? "No output was returned." : "waiting for the first generated token…");
    $("liveThinking").classList.toggle("thinking", !sampleDone(live));
    $("liveThinking").scrollTop = $("liveThinking").scrollHeight;

    const c = live.correct || {};
    const verdict = verdictInfo(live);
    const referenceLabel = kind === "bestmove"
      ? "Stockfish reference"
      : c.move
        ? "oracle reference"
        : kind === "cap"
          ? "rule"
          : "reference answer";
    $("liveReferenceLabel").textContent = referenceLabel;
    $("liveModelMove").textContent = live.move || "—";
    $("liveModelMove").classList.toggle("empty", !live.move);
    $("liveModelStatus").textContent = sampleDone(live)
      ? live.status === "legal" ? "parsed and legal" : (live.status || "unscored").replaceAll("_", " ")
      : "pending final answer";
    $("liveStockMove").textContent = c.move || (kind === "cap" ? "any legal move" : "—");
    $("liveStockMove").classList.toggle("empty", !c.move);
    $("liveStockMove").title = c.note || "";
    $("liveReferenceNote").textContent = c.note || "No reference move was published.";

    const vEl = $("liveVerdict");
    vEl.hidden = false;
    vEl.className = "live-verdict " + verdict.cls;
    vEl.innerHTML = `<strong></strong><span></span>`;
    vEl.querySelector("strong").textContent = verdict.title;
    vEl.querySelector("span").textContent = verdict.detail;
    $("liveDot").className = "live-dot" + (sampleDone(live) ? "" : " on");

    if (isGame) {
      $("liveIntegrity").hidden = true;
      $("liveBoard").style.gridTemplateColumns = "1fr";
      $("liveBoard").style.gridTemplateRows = "1fr";
      $("liveBoard").innerHTML = `<div class="board-empty">This is a full-game task. The monitor publishes the outcome, not each position.</div>`;
      $("liveCaption").textContent = "full-game cells stream outcomes only — per-move telemetry comes back with position tasks";
      return;
    }

    const integrityErrors = boardIntegrity(live);
    if (integrityErrors.length) {
      $("liveIntegrity").hidden = false;
      $("liveIntegrity").className = "live-integrity error";
      $("liveIntegrity").textContent = `BOARD HIDDEN · ${integrityErrors.join(" · ")}`;
      $("liveBoard").style.gridTemplateColumns = "1fr";
      $("liveBoard").style.gridTemplateRows = "1fr";
      $("liveBoard").innerHTML = `<div class="board-empty">Position data does not match the exact prompt. The dashboard will not show a potentially misleading board.</div>`;
      $("liveCaption").textContent = "board withheld until the monitor publishes a consistent sample";
      return;
    }
    $("liveIntegrity").hidden = false;
    $("liveIntegrity").className = "live-integrity ok";
    $("liveIntegrity").textContent = `position verified · ${live.record_id || live.position_id || "unknown record"}`;
    renderBoard(live);
    $("liveCaption").textContent = sampleDone(live) ? `completed · ${live.updated_at || ""}` : "generating…";
  }

  function renderReplay() {
    const el = $("liveReplay");
    if (!replay.length) { el.innerHTML = `<span class="live-caption">recent games will appear here</span>`; return; }
    el.innerHTML = replay.slice().reverse().map((s, i) => {
      const v = verdictInfo(s);
      const mark = !v ? "·" : v.cls === "correct" ? "✓" : v.cls === "wrong" ? "✗" : "△";
      const active = s === live ? " active" : "";
      return `<button class="replay-chip${active}" data-i="${replay.length - 1 - i}">
        <span class="r-mark ${v ? v.cls : "warn"}">${mark}</span>
        <span>${s.cell ? s.cell.model.split("-")[0] : ""} · ${s.cell ? s.cell.task : ""}</span>
      </button>`;
    }).join("");
  }

  async function loadLive() {
    try {
      const l = JSON.parse(await fetchFeed("live"));
      const incomingTime = l.updated_at ? new Date(l.updated_at).getTime() : 0;
      const currentTime = live && live.updated_at ? new Date(live.updated_at).getTime() : 0;
      if (live && incomingTime && currentTime && incomingTime < currentTime) return;
      const key = `${l.cell ? l.cell.model + l.cell.task + l.cell.variant : ""}|${l.position_id || ""}|${l.sample_idx || ""}`;
      if (key !== lastLiveKey) {
        if (live && live.position_id && live !== l) replay.push(live);
        if (replay.length > 40) replay.shift();
        lastLiveKey = key;
        live = l;
      } else if (l.updated_at !== live.updated_at) {
        live = l; // same position, refreshed content
      }
      renderLive();
      renderReplay();
    } catch (e) { /* live is best-effort; page keeps working */ }
  }

  // ---------------- boot ----------------
  $("refreshBtn").addEventListener("click", load);
  $("liveReplay").addEventListener("click", (ev) => {
    const chip = ev.target.closest(".replay-chip");
    if (!chip) return;
    const i = +chip.dataset.i;
    if (replay[i]) { live = replay[i]; renderLive(); renderReplay(); }
  });
  document.addEventListener("visibilitychange", () => { if (!document.hidden) { load(); loadLive(); } });
  if (CONFIG.REFRESH_S > 0) timer = setInterval(load, CONFIG.REFRESH_S * 1000);
  if (CONFIG.LIVE_REFRESH_S > 0) setInterval(loadLive, CONFIG.LIVE_REFRESH_S * 1000);
  load();
  loadLive();
})();
