// Self-Play Arena dashboard — live deepseek-vs-deepseek games.
// Data flow: /games/state (single throttled GitHub live feed, carries
// per-game thinking tails) + /games/{id} (full game from HF when a game
// finishes or is selected for replay).
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const WORKER_BASE = (CONFIG && CONFIG.WORKER_BASE) || "";
  const INDEX_URL = WORKER_BASE + "/games/state";
  const REFRESH_S = 15;
  const LIVE_REFRESH_S = 3;

  const state = {
    index: null,
    selected: null, // game id
    games: {},      // id -> full game json (from /games/{id})
    viewPly: null,  // null = live
  };

  function fenPieces(fen) {
    const out = {};
    if (!fen) return out;
    const board = fen.split(" ")[0];
    const rows = board.split("/");
    for (let r = 0; r < 8; r++) {
      let c = 0;
      for (const ch of rows[7 - r]) {
        if (/[1-8]/.test(ch)) c += +ch;
        else {
          out[String.fromCharCode(97 + c) + (r + 1)] = ch;
          c++;
        }
      }
    }
    return out;
  }

  function renderBoard(fen, lastUci) {
    const el = $("liveBoard");
    const pieces = fenPieces(fen);
    const fromSq = lastUci ? lastUci.slice(0, 2) : null;
    const toSq = lastUci ? lastUci.slice(2, 4) : null;
    el.style.gridTemplateColumns = "repeat(8, minmax(0, 1fr))";
    el.style.gridTemplateRows = "repeat(8, minmax(0, 1fr))";
    let html = "";
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const sq = String.fromCharCode(97 + c) + (8 - r);
        const piece = pieces[sq];
        const dark = (r + c) % 2 === 1;
        let cls = "board-cell " + (dark ? "dark" : "light");
        if (sq === fromSq || sq === toSq) cls += " last-move";
        const coords = [];
        if (c === 0) coords.push(`<span class="board-coord rank">${8 - r}</span>`);
        if (r === 7) coords.push(`<span class="board-coord file">${String.fromCharCode(97 + c)}</span>`);
        const pieceHtml = piece && window.CHESS_PIECES
          ? `<span class="piece-svg">${window.CHESS_PIECES[(piece === piece.toUpperCase() ? "w" : "b") + piece.toUpperCase()] || ""}</span>`
          : "";
        html += `<div class="${cls}" data-sq="${sq}">${pieceHtml}${coords.join("")}</div>`;
      }
    }
    el.innerHTML = html;
  }

  function renderMoves(game) {
    const el = $("liveReplay");
    const plies = (game && game.plies) || [];
    const liveHistory = (game && game.history) || [];
    const useLive = !plies.length && liveHistory.length;
    const moves = useLive ? liveHistory.map((san, i) => ({
      n: Math.floor(i / 2) + 1, by: i % 2 === 0 ? "w" : "b", san, book: false,
    })) : plies;
    if (!moves.length) {
      el.innerHTML = '<div class="empty">no moves yet</div>';
      return;
    }
    const view = state.viewPly == null ? moves.length - 1 : state.viewPly;
    let html = '<div class="moves-grid">';
    for (let i = 0; i < moves.length; i++) {
      const p = moves[i];
      if (i % 2 === 0) html += `<span class="move-num">${p.n}.</span>`;
      const active = i === view ? ' class="move-active"' : "";
      html += `<span${active} class="move-tag">${p.san}${p.book ? "⁕" : ""}</span>`;
    }
    html += "</div>";
    el.innerHTML = html;
    const act = el.querySelector(".move-active");
    if (act) act.scrollIntoView({ block: "nearest" });
  }

  // Live row from the index feed (has both sides' thinking + fen +
  // history); full game object from /games/{id} (has plies). Merge: board
  // from live fen while running, full replay once finished/selected.
  function renderRow(game) {
    const g = game || {};
    const plies = g.plies || [];
    const liveHistory = g.history || [];
    const lastUci = plies.length ? plies[plies.length - 1].uci : null;
    const liveFen = g.fen || (plies.length ? plies[plies.length - 1].fen : null);
    renderBoard(liveFen, lastUci);
    renderMoves(g);
    $("liveNow").textContent = `${g.id || "—"} · ${g.status || "—"} · ${g.result || "*"} · ${g.plies != null ? g.plies : (g.history || []).length} plies`;
    $("liveMeta").textContent = g.by ? `to move: ${g.by === "w" ? "White" : "Black"}` : "—";
    $("liveCaption").textContent = liveHistory.length
      ? `${liveHistory.join(" ")}`
      : plies.length ? `${plies.map((p) => p.san).join(" ")}` : "waiting for the first move…";
    const think = g.thinking || {};
    const wLive = (g.white && g.white.thinking) || (think.w || "");
    const bLive = (g.black && g.black.thinking) || (think.b || "");
    $("whiteThinking").textContent = wLive || "—";
    $("blackThinking").textContent = bLive || "—";
    $("whiteDot").classList.toggle("blink", !!g.by && g.by === "w" && g.status === "running");
    $("blackDot").classList.toggle("blink", !!g.by && g.by === "b" && g.status === "running");
    $("liveSync").textContent = `updated ${(g.updated_at || "").replace("T", " ").replace("Z", " UTC")}`;
  }

  function renderIndex() {
    const idx = state.index;
    if (!idx) return;
    const games = idx.games || [];
    const running = games.filter((g) => g.status === "running").length;
    const done = games.filter((g) => g.status !== "running" && g.status !== "not_started").length;
    $("entrantCount").textContent = `${games.length} games · ${running} running · ${done} done`;
    $("statusText").textContent = `${running} running / ${games.length}`;
    $("lampDot").classList.toggle("on", running > 0);
    const rows = $("entrantRows");
    rows.innerHTML = "";
    for (const g of games) {
      const row = document.createElement("button");
      row.className = "entrant-row" + (g.id === state.selected ? " active" : "");
      row.innerHTML = `<span class="entrant-id">${g.id}</span>
        <span class="entrant-status">${g.status}</span>
        <span class="entrant-result">${g.result}</span>
        <span class="entrant-plies">${g.plies} plies</span>`;
      row.onclick = () => selectGame(g.id);
      rows.appendChild(row);
    }
  }

  function selectGame(id) {
    state.selected = id;
    state.viewPly = null;
    renderIndex();
    loadFullGame(id);
  }

  async function fetchJson(url) {
    try {
      const r = await fetch(url, { cache: "no-store" });
      if (!r.ok) return null;
      return await r.json();
    } catch (e) {
      return null;
    }
  }

  async function loadFullGame(id) {
    const game = await fetchJson(WORKER_BASE + "/games/" + id);
    if (game) state.games[id] = game;
  }

  async function tick() {
    const idx = await fetchJson(INDEX_URL);
    if (idx && idx.games && idx.games.length) {
      state.index = idx;
      $("notice").hidden = true;
      if (!state.selected) {
        const running = idx.games.find((g) => g.status === "running");
        state.selected = (running || idx.games[idx.games.length - 1]).id;
      }
      renderIndex();
      const live = idx.games.find((g) => g.id === state.selected);
      const full = state.games[state.selected];
      // while running, live feed is fresher; merge thinking + fen into the
      // full game object so the board and panes update every 3s
      if (live) {
        if (full) {
          full.fen = live.fen;
          full.status = live.status;
          full.result = live.result;
          full.by = live.by;
          full.thinking = live.thinking;
          full.history = live.history;
          full.updated_at = live.updated_at;
        }
        renderRow(live);
      } else if (full) {
        renderRow(full);
      }
      // full plies arrive on first /games/{id} load (HF); refresh when finished
      if (!full && live && live.status !== "running") loadFullGame(state.selected);
    } else {
      $("notice").hidden = false;
    }
  }

  setInterval(tick, REFRESH_S * 1000);
  setInterval(tick, LIVE_REFRESH_S * 1000);
  $("refreshBtn").onclick = tick;

  $("navPrev").onclick = () => {
    const plies = state.games[state.selected]?.plies || [];
    const live = state.viewPly == null ? plies.length - 1 : state.viewPly;
    state.viewPly = Math.max(0, live - 1);
    renderRow(state.games[state.selected]);
  };
  $("navNext").onclick = () => {
    const plies = state.games[state.selected]?.plies || [];
    const live = state.viewPly == null ? plies.length - 1 : state.viewPly;
    state.viewPly = live + 1 >= plies.length ? null : live + 1;
    renderRow(state.games[state.selected]);
  };
  $("navSync").onclick = () => {
    state.viewPly = null;
    renderRow(state.games[state.selected]);
  };

  tick();
})();
