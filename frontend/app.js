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
      $("entrantRows").innerHTML = `<div class="empty" style="padding:26px">no entrants scored yet — awaiting the first games</div>`;
      $("cellsBody").innerHTML = `<tr><td colspan="9" class="empty">no games recorded yet — awaiting the first push</td></tr>`;
      $("tableCount").textContent = "0 games";
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
    $("runMeta").textContent = `${(state.models || []).length} entrants · ${state.mode || "?"}`;
    const cur = state.current;
    $("currentCell").hidden = !cur;
    if (cur) $("currentCell").textContent = `now: ${cur.model} × ${cur.task}`;

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
    const cells = state.cells || [];
    const byModel = modelWinAvg("cap-legal-8x8", "grid", "legal_rate");
    const ranked = bestOf(models, (m) => byModel[m] ?? null);
    const done = new Set(cells.filter((c) => c.done).map((c) => c.model));
    const total = new Set(cells.map((c) => `${c.model}|${c.task}|${c.variant}`));
    if (!ranked.length) {
      $("entrantRows").innerHTML = `<div class="empty" style="padding:26px">no entrants scored yet — awaiting the first games</div>`;
      return;
    }
    $("entrantRows").innerHTML = ranked.map(([m, v], i) => {
      const games = cells.filter((c) => c.model === m && c.done).length;
      const frac = total.size ? Math.min(1, games / total.size) : 0;
      return `
      <div class="entrant${done.has(m) ? " done" : ""}">
        <span class="entrant-rank">${String(i + 1).padStart(2, "0")}</span>
        <span class="entrant-name">${m}</span>
        <span class="entrant-mark">${fmtPct(v)}</span>
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

  function renderCharts() {
    if (!(state.cells || []).length) return;
    if (typeof Chart === "undefined") return; // CDN blocked — page still works
    const models = state.models || [];

    const legGrid = modelWinAvg("cap-legal-8x8", "grid", "legal_rate");
    const legFen = modelWinAvg("cap-legal-8x8", "fen", "legal_rate");
    const order = sortedModels(models, (m) => legGrid[m] ?? null);
    mkChart("chartLegal", {
      type: "bar",
      data: { labels: order, datasets: [
        barDataset("grid", order.map((m) => legGrid[m] ?? null), "rgba(229,72,77,0.75)"),
        barDataset("fen", order.map((m) => legFen[m] ?? null), "rgba(242,169,59,0.7)"),
      ] },
      options: baseOpts,
      plugins: [valueLabels],
    });

    const m1 = modelWinAvg("mate1-lichess", "grid", "compliance_of_legal");
    const m2 = modelWinAvg("mate2-lichess", "grid", "compliance_of_legal");
    const orderT = sortedModels(models, (m) => m1[m] ?? null);
    mkChart("chartTactics", {
      type: "bar",
      data: { labels: orderT, datasets: [
        barDataset("mate-in-1", orderT.map((m) => m1[m] ?? null), "rgba(229,72,77,0.75)"),
        barDataset("mate-in-2", orderT.map((m) => m2[m] ?? null), "rgba(233,230,223,0.55)"),
      ] },
      options: baseOpts,
      plugins: [valueLabels],
    });

    const stTop = modelWinAvg("bestmove-8x8", "grid", "compliance_of_legal");
    const stLegal = modelWinAvg("bestmove-8x8", "grid", "legal_rate");
    const orderS = sortedModels(models, (m) => stTop[m] ?? null);
    mkChart("chartStock", {
      type: "bar",
      data: { labels: orderS, datasets: [
        barDataset("top-1 vs stockfish", orderS.map((m) => stTop[m] ?? null), "rgba(242,169,59,0.75)"),
        barDataset("legal rate", orderS.map((m) => stLegal[m] ?? null), "rgba(90,171,130,0.7)"),
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
      options: { ...baseOpts, scales: { ...baseOpts.scales, x: { ...baseOpts.scales.x, title: { display: true, text: "games played", color: "#6c655a", font: { family: "IBM Plex Mono", size: 9.5 } } } } },
      plugins: [endLabel],
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
