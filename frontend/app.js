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
  const charts = {};

  // ---------------- status ----------------
  function status(kind, text) {
    const pill = $("statusPill");
    pill.className = "lamp " + kind;
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
      if (ageS > Math.max(CONFIG.REFRESH_S * 4, 180)) status("stale", "stale · " + Math.round(ageS / 60) + "m ago");
      else status("live", "live · " + timeAgo(state.updated_at));
      render();
    } catch (e) {
      status("error", "no signal");
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

  function bestOf(models, map) {
    const entries = models.map((m) => [m, map(m)]).filter(([, v]) => v !== null);
    entries.sort((a, b) => b[1] - a[1]);
    return entries;
  }

  // ---------------- render ----------------
  function render() {
    renderScoreboard();
    renderError();
    renderEntrants();
    renderCharts();
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
    $("runMeta").textContent = `${(state.models || []).length} entrants · ${state.mode || "?"}`;

    const cells = state.cells || [];
    const legalVals = cells.map((c) => winMetric(c, "legal_rate")).filter((v) => typeof v === "number");
    const mateVals = modelWinAvg("mate1-lichess", "grid", "compliance_of_legal");
    const stockVals = modelWinAvg("bestmove-8x8", "grid", "compliance_of_legal");
    const gameWins = cells.filter((c) => c.game && c.game.win_rate != null).map((c) => c.game.win_rate);

    $("kCells").textContent = `${p.cells_done ?? 0}`;
    $("kCellsSub").textContent = `of ${p.cells_total ?? "?"} games`;
    $("kLegal").textContent = fmtPct(avg(legalVals));
    $("kLegalSub").textContent = `${legalVals.length} cells`;
    $("kTactics").textContent = fmtPct(avg(Object.values(mateVals)));
    $("kTacticsSub").textContent = `${Object.keys(mateVals).length} models`;
    $("kStock").textContent = fmtPct(avg(Object.values(stockVals)));
    $("kStockSub").textContent = `${Object.keys(stockVals).length} models`;
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
    const byModel = modelWinAvg("cap-legal-8x8", "grid", "legal_rate");
    const ranked = bestOf(models, (m) => byModel[m] ?? null);
    const done = new Set(cellsFor(null, null).filter((c) => c.done).map((c) => c.model));
    if (!ranked.length) {
      $("entrantRows").innerHTML = `<div class="empty" style="padding:26px">no entrants scored yet — awaiting the first games</div>`;
      return;
    }
    $("entrantRows").innerHTML = ranked.map(([m, v], i) => `
      <div class="entrant${done.has(m) ? " done" : ""}">
        <span class="entrant-rank">${String(i + 1).padStart(2, "0")}</span>
        <span class="entrant-name">${m}</span>
        <span class="entrant-mark">${fmtPct(v)}</span>
      </div>`).join("");
  }

  // ---------------- charts ----------------
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

  function renderCharts() {
    if (!(state.cells || []).length) return;
    const models = state.models || [];

    const legGrid = modelWinAvg("cap-legal-8x8", "grid", "legal_rate");
    const legFen = modelWinAvg("cap-legal-8x8", "fen", "legal_rate");
    mkChart("chartLegal", {
      type: "bar",
      data: { labels: models, datasets: [
        barDataset("grid", models.map((m) => legGrid[m] ?? null), "rgba(229,72,77,0.75)"),
        barDataset("fen", models.map((m) => legFen[m] ?? null), "rgba(242,169,59,0.7)"),
      ] },
      options: baseOpts,
    });

    const m1 = modelWinAvg("mate1-lichess", "grid", "compliance_of_legal");
    const m2 = modelWinAvg("mate2-lichess", "grid", "compliance_of_legal");
    mkChart("chartTactics", {
      type: "bar",
      data: { labels: models, datasets: [
        barDataset("mate-in-1", models.map((m) => m1[m] ?? null), "rgba(229,72,77,0.75)"),
        barDataset("mate-in-2", models.map((m) => m2[m] ?? null), "rgba(233,230,223,0.55)"),
      ] },
      options: baseOpts,
    });

    const stTop = modelWinAvg("bestmove-8x8", "grid", "compliance_of_legal");
    const stLegal = modelWinAvg("bestmove-8x8", "grid", "legal_rate");
    mkChart("chartStock", {
      type: "bar",
      data: { labels: models, datasets: [
        barDataset("top-1 vs stockfish", models.map((m) => stTop[m] ?? null), "rgba(242,169,59,0.75)"),
        barDataset("legal rate", models.map((m) => stLegal[m] ?? null), "rgba(90,171,130,0.7)"),
      ] },
      options: baseOpts,
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
      options: { ...baseOpts, scales: { ...baseOpts.scales, x: { ...baseOpts.scales.x, title: { display: true, text: "games played", color: "#6c655a", font: { family: "IBM Plex Mono", size: 9.5 } } } } },
    });
  }

  // ---------------- table ----------------
  function renderTable() {
    const body = $("cellsBody");
    const cells = state.cells || [];
    $("tableCount").textContent = `${cells.length} games`;
    if (!cells.length) {
      body.innerHTML = `<tr><td colspan="9" class="empty">no games recorded yet — awaiting the first push</td></tr>`;
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
    body.innerHTML = rows.join("") || `<tr><td colspan="9" class="empty">no games recorded yet</td></tr>`;
  }

  // ---------------- boot ----------------
  $("refreshBtn").addEventListener("click", load);
  document.addEventListener("visibilitychange", () => { if (!document.hidden) load(); });
  if (CONFIG.REFRESH_S > 0) timer = setInterval(load, CONFIG.REFRESH_S * 1000);
  load();
})();
