/* ChessBench live dashboard — logic. Reads monitor/state.json + history.jsonl
   from the public live repo; renders KPIs, charts, ranking and the cell table. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const fmtPct = (v) => (v === null || v === undefined || v === "" ? "—" : `${(v * 100).toFixed(1)}%`);
  const fmtNum = (v) => (v === null || v === undefined || v === "" ? "—" : String(v));
  const num = (v) => (typeof v === "number" ? v : null);

  let state = null;
  let history = [];
  let timer = null;
  const charts = {};

  // ---------------- status ----------------
  function status(kind, text) {
    const pill = $("statusPill");
    pill.className = "status-pill " + kind;
    $("statusText").textContent = text;
  }

  // ---------------- data ----------------
  async function fetchText(url) {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`${res.status} ${url}`);
    return res.text();
  }

  async function load() {
    try {
      state = JSON.parse(await fetchText(CONFIG.STATE_URL));
      try {
        history = (await fetchText(CONFIG.HISTORY_URL)).trim().split("\n")
          .filter(Boolean).map((l) => JSON.parse(l));
      } catch { history = []; }
      const ageS = (Date.now() - new Date(state.updated_at).getTime()) / 1000;
      if (ageS > Math.max(CONFIG.REFRESH_S * 4, 180)) status("stale", `stale · ${Math.round(ageS / 60)}m ago`);
      else status("live", "live · " + timeAgo(state.updated_at));
      render();
    } catch (e) {
      status("error", "cannot reach monitor");
      console.warn(e);
    }
  }

  function timeAgo(iso) {
    const s = (Date.now() - new Date(iso).getTime()) / 1000;
    if (s < 5) return "just now";
    if (s < 60) return `${Math.round(s)}s ago`;
    if (s < 3600) return `${Math.round(s / 60)}m ago`;
    return `${Math.round(s / 3600)}h ago`;
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
  function gameMetric(cell, field) { return cell && cell.game ? cell.game[field] ?? null : null; }
  function modelWinAvg(task, variant, field) {
    const per = {};
    for (const c of cellsFor(task, variant)) {
      const v = winMetric(c, field);
      if (typeof v === "number") (per[c.model] = per[c.model] || []).push(v);
    }
    return Object.fromEntries(Object.entries(per).map(([m, vs]) => [m, avg(vs)]));
  }
  function bestOf(models, map) {
    const entries = models.map((m) => [m, map(m)]).filter(([, v]) => v !== null);
    entries.sort((a, b) => b[1] - a[1]);
    return entries;
  }

  // ---------------- render ----------------
  function render() {
    renderRun();
    renderKpis();
    renderError();
    renderModelChips();
    renderRanking();
    renderCharts();
    renderTable();
    $("footRepo").textContent = `repo: ${state.repo || "—"} · ${state.mode || "?"} mode`;
    $("rawLink").href = CONFIG.STATE_URL;
  }

  function renderRun() {
    const p = state.progress || {};
    const frac = p.fraction || 0;
    $("progressFill").style.width = `${(frac * 100).toFixed(1)}%`;
    $("progressText").textContent = `${p.cells_done ?? 0} / ${p.cells_total ?? "?"} cells complete`;
    $("etaText").textContent = state.eta_min != null ? `ETA ~${state.eta_min} min` : "ETA —";
    $("stageTag").textContent = (state.stage || "sweep").toUpperCase();
    $("runMeta").textContent = `${state.mode || "?"} mode · ${(state.models || []).length} models · started ${state.started_at || "—"}`;
    $("updatedText").textContent = "↻ " + timeAgo(state.updated_at);
  }

  function renderModelChips() {
    const models = state.models || [];
    const done = new Set(cellsFor(null, null).filter((c) => c.done).map((c) => c.model));
    $("modelChips").innerHTML = models
      .map((m) => `<span class="model-chip${done.has(m) ? " done" : ""}">${done.has(m) ? "✓" : "·"} ${m}</span>`)
      .join("");
  }

  function renderKpis() {
    const cells = state.cells || [];
    const legalVals = cells.map((c) => winMetric(c, "legal_rate")).filter((v) => typeof v === "number");
    const mateVals = modelWinAvg("mate1-lichess", "grid", "compliance_of_legal");
    const stockVals = modelWinAvg("bestmove-8x8", "grid", "compliance_of_legal");
    const gameWins = cells.filter((c) => c.game && c.game.win_rate != null).map((c) => c.game.win_rate);

    $("kCells").textContent = fmtNum(state.progress?.cells_done);
    $("kCellsSub").textContent = `of ${state.progress?.cells_total ?? "?"}`;
    $("kLegal").textContent = fmtPct(avg(legalVals));
    $("kLegalSub").textContent = `${legalVals.length} cells`;
    $("kTactics").textContent = fmtPct(avg(Object.values(mateVals)));
    $("kTacticsSub").textContent = `${Object.keys(mateVals).length} models`;
    $("kStock").textContent = fmtPct(avg(Object.values(stockVals)));
    $("kStockSub").textContent = `${Object.keys(stockVals).length} models`;
    $("kGames").textContent = fmtPct(avg(gameWins));
    $("kGamesSub").textContent = `${gameWins.length} game cells`;

    // representation gap: avg legal(grid) - avg legal(fen) across 8x8 tasks
    let gridVals = [], fenVals = [];
    for (const task of ["cap-legal-8x8", "mate1-lichess", "mate2-lichess", "bestmove-8x8"]) {
      for (const c of cellsFor(task, "grid")) { const v = winMetric(c, "legal_rate"); if (typeof v === "number") gridVals.push(v); }
      for (const c of cellsFor(task, "fen")) { const v = winMetric(c, "legal_rate"); if (typeof v === "number") fenVals.push(v); }
    }
    const delta = avg(gridVals) !== null && avg(fenVals) !== null ? avg(gridVals) - avg(fenVals) : null;
    $("kDivergence").textContent = delta === null ? "—" : (delta >= 0 ? "+" : "") + (delta * 100).toFixed(1) + "%";
    $("kDivergenceSub").textContent = "grid − fen legality";
  }

  function renderError() {
    const has = !!state.last_error;
    $("errorBanner").hidden = !has;
    $("errorText").textContent = has ? state.last_error : "";
  }

  // ---------------- ranking ----------------
  function renderRanking() {
    const models = state.models || [];
    const el = $("ranking");
    const byModel = modelWinAvg("cap-legal-8x8", "grid", "legal_rate");
    const ranked = bestOf(models, (m) => byModel[m] ?? null);
    if (!ranked.length) {
      el.innerHTML = `<div class="empty">no legality data yet — waiting for the first cells</div>`;
      return;
    }
    el.innerHTML = ranked.map(([m, v], i) => `
      <div class="rank-row">
        <div class="rank-badge">${i + 1}</div>
        <div class="rank-info">
          <div class="rank-name"><span>${m}</span><span class="rank-val">legal move rate</span></div>
          <div class="rank-track"><div class="rank-fill" style="width:${(v * 100).toFixed(1)}%"></div></div>
        </div>
        <div class="rank-legal">${fmtPct(v)}</div>
      </div>`).join("");
  }

  // ---------------- charts ----------------
  const baseOpts = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: { labels: { color: "#a39dc4", boxWidth: 8, boxHeight: 8, usePointStyle: true, font: { family: "Inter", size: 11 } } },
      tooltip: {
        backgroundColor: "rgba(16,14,29,0.95)", borderColor: "rgba(255,255,255,0.12)", borderWidth: 1,
        titleColor: "#f2effb", bodyColor: "#a39dc4", padding: 10, cornerRadius: 10,
        callbacks: { label: (c) => ` ${c.dataset.label}: ${(c.parsed.y * 100).toFixed(1)}%` },
      },
    },
    scales: {
      x: { ticks: { color: "#645e87", font: { family: "JetBrains Mono", size: 10 }, maxRotation: 40 }, grid: { display: false } },
      y: { beginAtZero: true, max: 1, ticks: { color: "#645e87", font: { family: "JetBrains Mono", size: 10 }, callback: (v) => (v * 100).toFixed(0) + "%" }, grid: { color: "rgba(255,255,255,0.05)" }, border: { display: false } },
    },
  };

  function mkChart(id, cfg) {
    if (charts[id]) charts[id].destroy();
    charts[id] = new Chart($(id).getContext("2d"), cfg);
  }

  function barDataset(label, data, from, to) {
    const ctx = document.createElement("canvas").getContext("2d");
    const grad = ctx.createLinearGradient(0, 0, 0, 250);
    grad.addColorStop(0, to); grad.addColorStop(1, from);
    return { label, data, backgroundColor: grad, borderColor: to, borderWidth: 0, borderRadius: 6, maxBarThickness: 34 };
  }

  function renderCharts() {
    const cells = state.cells || [];
    if (!cells.length) return;
    const models = state.models || [];

    // 1. legality: grid vs fen by model
    const legGrid = modelWinAvg("cap-legal-8x8", "grid", "legal_rate");
    const legFen = modelWinAvg("cap-legal-8x8", "fen", "legal_rate");
    mkChart("chartLegal", {
      type: "bar",
      data: { labels: models, datasets: [
        barDataset("grid", models.map((m) => legGrid[m] ?? null), "rgba(55,214,160,0.14)", "#37d6a0"),
        barDataset("fen", models.map((m) => legFen[m] ?? null), "rgba(139,123,255,0.14)", "#8b7bff"),
      ] },
      options: baseOpts,
    });

    // 2. tactics: mate1 + mate2 (grid)
    const m1 = modelWinAvg("mate1-lichess", "grid", "compliance_of_legal");
    const m2 = modelWinAvg("mate2-lichess", "grid", "compliance_of_legal");
    mkChart("chartTactics", {
      type: "bar",
      data: { labels: models, datasets: [
        barDataset("mate-in-1", models.map((m) => m1[m] ?? null), "rgba(139,123,255,0.14)", "#8b7bff"),
        barDataset("mate-in-2", models.map((m) => m2[m] ?? null), "rgba(77,214,232,0.14)", "#4dd6e8"),
      ] },
      options: baseOpts,
    });

    // 3. Stockfish top-1 vs legal rate
    const stTop = modelWinAvg("bestmove-8x8", "grid", "compliance_of_legal");
    const stLegal = modelWinAvg("bestmove-8x8", "grid", "legal_rate");
    mkChart("chartStock", {
      type: "bar",
      data: { labels: models, datasets: [
        barDataset("top-1 vs Stockfish", models.map((m) => stTop[m] ?? null), "rgba(196,181,255,0.14)", "#c4b5ff"),
        barDataset("legal rate", models.map((m) => stLegal[m] ?? null), "rgba(55,214,160,0.14)", "#37d6a0"),
      ] },
      options: baseOpts,
    });

    // 4. history line
    mkChart("chartHistory", {
      type: "line",
      data: {
        labels: history.map((h) => h.cells_done),
        datasets: [{
          label: "avg legal rate",
          data: history.map((h) => h.legal_avg),
          borderColor: "#8b7bff", borderWidth: 2,
          backgroundColor: "rgba(139,123,255,0.12)",
          fill: true, tension: 0.35, pointRadius: 2.5, pointBackgroundColor: "#8b7bff",
        }],
      },
      options: { ...baseOpts, scales: { ...baseOpts.scales, x: { ...baseOpts.scales.x, title: { display: true, text: "cells done", color: "#645e87", font: { family: "Inter", size: 10 } } } } },
    });
  }

  // ---------------- table ----------------
  function renderTable() {
    const body = $("cellsBody");
    const cells = state.cells || [];
    $("tableCount").textContent = `${cells.length} cells`;
    if (!cells.length) {
      body.innerHTML = `<tr><td colspan="9" class="empty">no data yet — awaiting the first push</td></tr>`;
      return;
    }
    const rows = [];
    for (const c of cells) {
      if (c.game) {
        const g = c.game;
        rows.push(`<tr>
          <td class="mono">${c.model}</td>
          <td>${c.task}</td>
          <td><span class="tag">${c.variant}</span></td>
          <td>game</td>
          <td class="num mono">${fmtNum(g.n)}</td>
          <td class="num mono dim">—</td>
          <td class="num mono ${g.legal_rate > 0.3 ? "pos" : g.legal_rate > 0 ? "warn" : "neg"}">${fmtPct(g.legal_rate)}</td>
          <td class="num mono ${g.win_rate > 0.3 ? "pos" : g.win_rate > 0 ? "warn" : "dim"}">win ${fmtPct(g.win_rate)}</td>
          <td><span class="cell-status">done</span></td>
        </tr>`);
        continue;
      }
      for (const cond of ["win"]) {
        const m = c[cond];
        if (!m) continue;
        const parse = num(m.parse_rate), legal = num(m.legal_rate), comp = num(m.compliance_of_legal);
        rows.push(`<tr>
          <td class="mono">${c.model}</td>
          <td>${c.task}</td>
          <td><span class="tag">${c.variant}</span></td>
          <td>${cond}</td>
          <td class="num mono">${fmtNum(m.n)}</td>
          <td class="num mono ${parse > 0.5 ? "pos" : "neg"}">${fmtPct(parse)}</td>
          <td class="num mono ${legal > 0.3 ? "pos" : legal > 0 ? "warn" : "neg"}">${fmtPct(legal)}</td>
          <td class="num mono ${comp > 0.3 ? "pos" : comp > 0 ? "warn" : "dim"}">${fmtPct(comp)}</td>
          <td><span class="cell-status">done</span></td>
        </tr>`);
      }
    }
    body.innerHTML = rows.join("") || `<tr><td colspan="9" class="empty">no data yet</td></tr>`;
  }

  // ---------------- boot ----------------
  $("refreshBtn").addEventListener("click", load);
  document.addEventListener("visibilitychange", () => { if (!document.hidden) load(); });
  if (CONFIG.REFRESH_S > 0) timer = setInterval(load, CONFIG.REFRESH_S * 1000);
  load();
})();
