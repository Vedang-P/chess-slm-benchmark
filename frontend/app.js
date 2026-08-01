/* Live monitor for the anti-goal chess benchmark sweep.
   Reads monitor/state.json + history.jsonl from the public live repo. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const fmtPct = (v) => (v === null || v === undefined || v === "" ? "—" : `${(v * 100).toFixed(1)}%`);
  const fmtNum = (v) => (v === null || v === undefined || v === "" ? "—" : String(v));

  let state = null;
  let history = [];
  let timer = null;

  const CHARTS = {
    legal: { border: "#37d6a0", bg: "rgba(55,214,160,0.35)" },
    mate: { border: "#8b7bff", bg: "rgba(139,123,255,0.35)" },
    comply: { win: "#37d6a0", lose: "#ff6b7a" },
  };

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
      if (ageS > CONFIG.REFRESH_S * 4) status("stale", `stale · ${Math.round(ageS / 60)}m ago`);
      else status("live", "live · updated " + timeAgo(state.updated_at));
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

  // ---------------- render ----------------
  function render() {
    renderHero();
    renderKpis();
    renderError();
    renderCharts();
    renderTable();
    $("footRepo").textContent = `repo: ${state.repo || "—"} · mode: ${state.mode || "—"} · run: ${state.started_at || "—"}`;
    $("rawLink").href = CONFIG.STATE_URL;
  }

  function renderHero() {
    const p = state.progress || {};
    const frac = p.fraction || 0;
    $("progressFill").style.width = `${(frac * 100).toFixed(1)}%`;
    $("progressText").textContent =
      `${p.cells_done ?? 0} / ${p.cells_total ?? "?"} cells complete`;
    $("etaText").textContent = state.eta_min != null ? `ETA ~${state.eta_min} min` : "—";
    $("stageTag").textContent = (state.stage || "sweep").toUpperCase();
    $("runMeta").textContent = `${state.mode || "?"} mode · ${(state.models || []).length} models`;
    $("updatedText").textContent = "updated " + timeAgo(state.updated_at);
  }

  function avg(vals) {
    const ok = vals.filter((v) => typeof v === "number");
    return ok.length ? ok.reduce((a, b) => a + b, 0) / ok.length : null;
  }

  function renderKpis() {
    const cells = state.cells || [];
    $("kCells").textContent = fmtNum(state.progress?.cells_done);
    const legalVals = cells.map((c) => c.win?.legal_rate).filter((v) => typeof v === "number");
    const winC = cells.map((c) => c.win?.compliance_of_legal).filter((v) => typeof v === "number");
    const loseC = cells.map((c) => c.lose?.compliance_of_legal).filter((v) => typeof v === "number");
    const divs = cells.map((c) => c.divergence).filter((v) => typeof v === "number");
    $("kLegal").textContent = fmtPct(avg(legalVals));
    $("kComplyWin").textContent = fmtPct(avg(winC));
    $("kComplyLose").textContent = fmtPct(avg(loseC));
    $("kDivergence").textContent = fmtPct(avg(divs));
  }

  function renderError() {
    const has = !!state.last_error;
    $("errorBanner").hidden = !has;
    $("errorText").textContent = has ? state.last_error : "";
  }

  // ---------------- charts ----------------
  const charts = {};
  function mkChart(id, make) {
    if (charts[id]) charts[id].destroy();
    const chart = make($(id).getContext("2d"));
    charts[id] = chart;
    return chart;
  }

  const baseOpts = {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { labels: { color: "#a8a3c2", boxWidth: 10 } } },
    scales: {
      x: { ticks: { color: "#6f6a8f", maxRotation: 45 }, grid: { color: "rgba(42,37,64,0.5)" } },
      y: { beginAtZero: true, max: 1, ticks: { color: "#6f6a8f", callback: (v) => v * 100 + "%" }, grid: { color: "rgba(42,37,64,0.5)" } },
    },
  };

  function cellVals(task, cond, field) {
    return (state.cells || [])
      .filter((c) => c.task === task)
      .map((c) => ({ model: c.model, v: (c[cond] || {})[field] }))
      .filter((x) => typeof x.v === "number");
  }

  function renderCharts() {
    const cells = state.cells || [];
    if (!cells.length) return;

    // 1) legal rate by model (cap-legal-8x8, grid + fen)
    const capLegal = cells.filter((c) => c.task === "cap-legal-8x8");
    const models = [...new Set(cells.map((c) => c.model))];
    mkChart("chartLegal", (ctx) => new Chart(ctx, {
      type: "bar",
      data: {
        labels: models,
        datasets: ["grid", "fen"].map((variant, i) => ({
          label: variant,
          data: models.map((m) => {
            const c = capLegal.find((x) => x.model === m && x.variant === variant);
            return c?.win?.legal_rate ?? null;
          }),
          backgroundColor: i ? "rgba(139,123,255,0.5)" : "rgba(55,214,160,0.5)",
          borderColor: i ? "#8b7bff" : "#37d6a0",
          borderWidth: 1,
        })),
      },
      options: baseOpts,
    }));

    // 2) mate-in-1 solve rate
    const mates = cellVals("mate1-lichess", "win", "compliance_of_legal");
    mkChart("chartMate", (ctx) => new Chart(ctx, {
      type: "bar",
      data: {
        labels: mates.map((x) => x.model),
        datasets: [{ label: "mate found (win)", data: mates.map((x) => x.v),
                     backgroundColor: "rgba(139,123,255,0.5)", borderColor: "#8b7bff", borderWidth: 1 }],
      },
      options: baseOpts,
    }));

    // 3) best-move top-1 accuracy (bestmove-8x8)
    const wins = cellVals("bestmove-8x8", "win", "compliance_of_legal");
    const legalVals = cellVals("bestmove-8x8", "win", "legal_rate");
    mkChart("chartComply", (ctx) => new Chart(ctx, {
      type: "bar",
      data: {
        labels: wins.map((x) => x.model),
        datasets: [
          { label: "top-1 vs Stockfish", data: wins.map((x) => x.v), backgroundColor: "rgba(55,214,160,0.5)", borderColor: "#37d6a0", borderWidth: 1 },
          { label: "legal rate", data: legalVals.map((x) => x.v), backgroundColor: "rgba(139,123,255,0.5)", borderColor: "#8b7bff", borderWidth: 1 },
        ],
      },
      options: baseOpts,
    }));

    // 4) history line
    mkChart("chartHistory", (ctx) => new Chart(ctx, {
      type: "line",
      data: {
        labels: history.map((h) => h.cells_done),
        datasets: [{
          label: "avg legal rate",
          data: history.map((h) => h.legal_avg),
          borderColor: "#8b7bff", backgroundColor: "rgba(139,123,255,0.15)",
          fill: true, tension: 0.3, pointRadius: 2,
        }],
      },
      options: baseOpts,
    }));
  }

  // ---------------- table ----------------
  function renderTable() {
    const body = $("cellsBody");
    const cells = state.cells || [];
    if (!cells.length) {
      body.innerHTML = `<tr><td colspan="9" class="empty">no data yet</td></tr>`;
      return;
    }
    const rows = [];
    for (const c of cells) {
      for (const cond of ["win", "lose"]) {
        const m = c[cond];
        if (!m) continue;
        const parse = m.parse_rate, legal = m.legal_rate, comp = m.compliance_of_legal;
        rows.push(`<tr>
          <td class="mono">${c.model}</td>
          <td>${c.task}</td>
          <td><span class="tag">${c.variant}</span></td>
          <td>${cond}</td>
          <td class="mono">${fmtNum(m.n)}</td>
          <td class="mono ${parse > 0.5 ? "pos" : "neg"}">${fmtPct(parse)}</td>
          <td class="mono ${legal > 0.3 ? "pos" : legal > 0 ? "warn" : "neg"}">${fmtPct(legal)}</td>
          <td class="mono ${comp > 0.3 ? "pos" : comp > 0 ? "warn" : "dim"}">${fmtPct(comp)}</td>
          <td><span class="dot ok">●</span></td>
        </tr>`);
      }
      if (c.divergence !== null && c.divergence !== undefined) {
        rows.push(`<tr class="dim">
          <td class="mono">${c.model}</td><td>${c.task}</td><td><span class="tag">${c.variant}</span></td>
          <td>divergence</td><td colspan="4" class="mono">${fmtPct(c.divergence)}</td>
          <td><span class="dot ok">●</span></td></tr>`);
      }
    }
    body.innerHTML = rows.join("") || `<tr><td colspan="9" class="empty">no data yet</td></tr>`;
  }

  // ---------------- boot ----------------
  $("refreshBtn").addEventListener("click", load);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) load();
  });
  if (CONFIG.REFRESH_S > 0) {
    timer = setInterval(load, CONFIG.REFRESH_S * 1000);
  }
  load();
})();
