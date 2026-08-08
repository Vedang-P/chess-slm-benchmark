/* ChessReasoner-120M — page behaviour.
   Scroll reveals, the animated baseline chart, the interactive architecture
   diagram, and the live board <-> token-span mapping. No dependencies. */
(function () {
  'use strict';

  /* ---------- scroll reveal + section nav ------------------------------- */
  const revealables = document.querySelectorAll('.rv');
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } });
  }, { rootMargin: '0px 0px -8% 0px', threshold: 0.06 });
  revealables.forEach((el) => io.observe(el));

  const navLinks = [...document.querySelectorAll('.nav a')];
  const navIo = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      const link = navLinks.find((a) => a.getAttribute('href') === '#' + e.target.id);
      if (link && e.isIntersecting) {
        navLinks.forEach((a) => a.classList.remove('on'));
        link.classList.add('on');
      }
    });
  }, { rootMargin: '-45% 0px -45% 0px' });
  document.querySelectorAll('section[id]').forEach((s) => navIo.observe(s));

  /* ---------- Fig. 1 bars ------------------------------------------------ */
  const barsRoot = document.getElementById('bars');
  if (barsRoot) {
    const barIo = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        [...barsRoot.querySelectorAll('.bar-row')].forEach((row, i) => {
          const pct = parseFloat(row.dataset.v);
          setTimeout(() => { row.querySelector('.bar-fill').style.width = pct + '%'; }, i * 110);
        });
        barIo.disconnect();
      });
    }, { threshold: 0.25 });
    barIo.observe(barsRoot);
  }

  /* ---------- shared chess data ----------------------------------------- */
  // Ruy Lopez after 3...a6 4.Ba4 Nf6 — a real, recognisable middlegame-bound
  // position rather than the start position, so the empty-square density is
  // representative.
  const FEN = 'r1bqkb1r/1ppp1ppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R';
  const GLYPH = {
    P: '♙', N: '♘', B: '♗', R: '♖', Q: '♕', K: '♔',
    p: '♟', n: '♞', b: '♝', r: '♜', q: '♛', k: '♚'
  };
  const NAME = {
    P: 'white pawn', N: 'white knight', B: 'white bishop', R: 'white rook',
    Q: 'white queen', K: 'white king', p: 'black pawn', n: 'black knight',
    b: 'black bishop', r: 'black rook', q: 'black queen', k: 'black king'
  };

  // occupancy indexed by python-chess square number: a1=0, b1=1 ... h8=63
  function parseFen(fen) {
    const occ = new Array(64).fill(null);
    fen.split('/').forEach((rankStr, i) => {
      const rank = 7 - i;            // FEN starts at rank 8
      let file = 0;
      for (const ch of rankStr) {
        if (/\d/.test(ch)) { file += +ch; continue; }
        occ[rank * 8 + file] = ch;
        file += 1;
      }
    });
    return occ;
  }
  const OCC = parseFen(FEN);
  const sqName = (i) => 'abcdefgh'[i % 8] + (Math.floor(i / 8) + 1);

  /* ---------- Fig. 3 mini board + span cells (inside the SVG) ------------ */
  const NS = 'http://www.w3.org/2000/svg';
  const miniBoard = document.getElementById('mini-board');
  if (miniBoard) {
    for (let r = 0; r < 8; r++) {
      for (let f = 0; f < 8; f++) {
        const sq = (7 - r) * 8 + f;
        const cell = document.createElementNS(NS, 'rect');
        cell.setAttribute('x', f * 8.7); cell.setAttribute('y', r * 8.7);
        cell.setAttribute('width', 8.7); cell.setAttribute('height', 8.7);
        cell.setAttribute('fill', OCC[sq] ? '#e0a03c' : ((r + f) % 2 ? '#10161c' : '#1a222a'));
        cell.setAttribute('fill-opacity', OCC[sq] ? 0.75 : 1);
        miniBoard.appendChild(cell);
      }
    }
  }
  const spanCells = document.getElementById('span-cells');
  if (spanCells) {
    for (let i = 0; i < 36; i++) {
      const cell = document.createElementNS(NS, 'rect');
      cell.setAttribute('x', (i % 18) * 4.9); cell.setAttribute('y', Math.floor(i / 18) * 5.4);
      cell.setAttribute('width', 3.9); cell.setAttribute('height', 4.2);
      const isMarker = i === 0 || i === 35;
      cell.setAttribute('fill', isMarker ? '#52b5c4' : (OCC[i * 2] ? '#e0a03c' : '#1e2831'));
      spanCells.appendChild(cell);
    }
  }

  /* ---------- Fig. 3 interactive rail ----------------------------------- */
  const RAIL = {
    board: { t: 'Raw position', tag: '', tagCls: '',
      p: 'The input is a FEN string and nothing else. Expanding it into a piece list, or pre-applying the candidate moves, would hand the model the exact sub-skill under test — so the input stays raw and the scaffolding moves into the training objective instead.',
      dl: [['Input', 'FEN, unmodified'], ['Preprocessing', 'none'], ['Engine at inference', 'none']] },
    span: { t: 'The 72-token board span', tag: 'fixed length', tagCls: '',
      p: 'One marker, sixty-four square-content tokens in a1→h8 raster order, side to move, four castling slots, en passant, one closing marker. Always exactly 72 tokens. The regularity is the point: square i is always at a computable offset, so file, rank and diagonal neighbours are constant strides.',
      dl: [['Length', '72 tokens, always'], ['Order', 'a1 → h8 raster'], ['Encoding', 'no run-length compression'], ['Empty squares', '57% on average']] },
    stack: { t: '18 decoder blocks', tag: 'the backbone', tagCls: 'keep',
      p: 'Pre-norm, no biases, tied embeddings. Deeper than the usual 120M configuration — GPT-2 small and Pythia-160M are both 12×768. Serializing a search is sequential composition, and depth buys composition steps that width does not.',
      dl: [['Layers', '18'], ['d_model', '768'], ['Attention', 'GQA 12 Q / 4 KV'], ['FFN', 'SwiGLU 2048'], ['Parameters', '113,273,856']] },
    lm: { t: 'LM head', tag: 'kept at inference', tagCls: 'keep',
      p: 'Tied to the input embedding, so it costs no extra parameters. This is the entire inference path: board in, prose out. Everything else on this diagram is training apparatus.',
      dl: [['Shape', '768 → 8192'], ['Weights', 'tied to embedding'], ['Fires at', 'every position'], ['Ships', 'yes']] },
    boardhead: { t: 'Board head', tag: 'training only', tagCls: 'drop',
      p: 'Predicts 64-square occupancy at the end of an analysed variation — not at the FEN, where the board is verbatim in context and the head would be a copy. At the leaf of a line the model must have simulated the moves, so the target exists nowhere in the input. That is what makes it a world model rather than a parser.',
      dl: [['Shape', '768 → 64×13'], ['Fires at', '</LINE>'], ['Parameters', '638,976'], ['Ships', 'no — deleted']] },
    value: { t: 'Value head', tag: 'training only', tagCls: 'drop',
      p: 'A 128-bin categorical distribution over win probability with Gaussian smoothing, not a scalar regression — regressing centipawns is badly conditioned and trains poorly. This head is the internalized half of the design principle: the evaluation that cannot be verbalized.',
      dl: [['Shape', '768 → 128'], ['Target', 'HL-Gauss WDL'], ['Fires at', '</FEN>'], ['Ships', 'no — deleted']] },
    policy: { t: 'Policy head', tag: 'training only', tagCls: 'drop',
      p: 'A candidate prior over all 1968 from-to-promotion moves, derived programmatically so it can never desynchronize from the move encoding. It stops the model spending its verbalized budget on moves no strong player would consider. It also fires before any reasoning exists, which is a risk that has to be measured — see §07.',
      dl: [['Shape', '768 → 1968'], ['Vocabulary', 'derived, not hardcoded'], ['Fires at', '</FEN>'], ['Ships', 'no — deleted']] },
    loss: { t: 'The objective', tag: 'annealed', tagCls: '',
      p: 'Language modelling plus three auxiliary terms. The auxiliary weights are held at full strength for the first half of training and then decayed linearly to zero, so the model is never able to lean on heads it will not have at inference.',
      dl: [['LM loss', 'segment-weighted'], ['Board / prompt / answer', '0.1 / 0.0 / 1.0'], ['Anneal', 'linear over 2nd half'], ['Final λ', '0']] }
  };

  const rail = document.getElementById('arch-rail');
  const nodes = [...document.querySelectorAll('#arch-svg .node')];
  function showRail(key) {
    const d = RAIL[key];
    if (!d || !rail) return;
    nodes.forEach((n) => n.classList.toggle('sel', n.dataset.k === key));
    rail.innerHTML =
      '<div class="tag ' + d.tagCls + '">' + (d.tag || 'component') + '</div>' +
      '<h4>' + d.t + '</h4><p>' + d.p + '</p><dl>' +
      d.dl.map(([k, v]) => '<dt>' + k + '</dt><dd>' + v + '</dd>').join('') + '</dl>';
  }
  nodes.forEach((n) => {
    n.addEventListener('click', () => showRail(n.dataset.k));
    n.addEventListener('mouseenter', () => showRail(n.dataset.k));
  });

  /* ---------- Fig. 4 live board <-> token span -------------------------- */
  const boardEl = document.getElementById('live-board');
  const tokEl = document.getElementById('live-tokens');
  const readout = document.getElementById('tok-readout');

  if (boardEl && tokEl) {
    // board is drawn rank 8 at the top, so visual row r maps to square (7-r)*8+f
    for (let r = 0; r < 8; r++) {
      for (let f = 0; f < 8; f++) {
        const sq = (7 - r) * 8 + f;
        const d = document.createElement('div');
        d.className = 'sq ' + ((r + f) % 2 ? 'dk' : 'lt');
        d.dataset.sq = sq;
        d.textContent = OCC[sq] ? GLYPH[OCC[sq]] : '';
        d.style.color = OCC[sq] && OCC[sq] === OCC[sq].toUpperCase() ? '#e8e3d8' : '#8d8a80';
        const co = document.createElement('span');
        co.className = 'co'; co.textContent = sqName(sq);
        d.appendChild(co);
        boardEl.appendChild(d);
      }
    }

    const chips = [];
    function chip(text, cls, sq, note) {
      const s = document.createElement('span');
      s.className = 'tk' + (cls ? ' ' + cls : '');
      s.textContent = text;
      if (sq != null) s.dataset.sq = sq;
      s.dataset.note = note;
      tokEl.appendChild(s);
      chips.push(s);
      return s;
    }

    chip('<FEN>', 'mark', null, 'Opens the board span. Token 1 of 72.');
    for (let i = 0; i < 64; i++) {
      const p = OCC[i];
      chip(p ? '<' + p + '>' : '<.>', null, i,
        (p ? NAME[p] : 'empty') + ' on ' + sqName(i) + ' — span offset ' + (i + 1) + ' of 72');
    }
    chip('<stm:b>', 'flag', null, 'Side to move. Black, after 4.Ba4 Nf6 is answered.');
    ['<yes>', '<yes>', '<yes>', '<yes>'].forEach((t, i) => chip(t, 'flag', null,
      'Castling slot ' + (i + 1) + ' of 4 — ' +
      ['White king-side', 'White queen-side', 'Black king-side', 'Black queen-side'][i] +
      '. Positional, so each slot only needs yes/no; the letters K/Q/k/q would collide with the piece tokens.'));
    chip('<no>', 'flag', null, 'En-passant square, or <no> when there is none.');
    chip('</FEN>', 'mark', null, 'Closes the span. Token 72 — and the anchor the value and policy heads read from.');

    let pinned = false;
    function highlight(sq, note) {
      boardEl.querySelectorAll('.sq').forEach((s) => s.classList.toggle('on', +s.dataset.sq === sq));
      chips.forEach((c) => c.classList.toggle('on', c.dataset.sq != null && +c.dataset.sq === sq));
      if (note) readout.textContent = note;
    }
    function clear() {
      if (pinned) return;
      boardEl.querySelectorAll('.sq.on').forEach((s) => s.classList.remove('on'));
      chips.forEach((c) => c.classList.remove('on'));
      readout.textContent = 'Hover a square or a token.';
    }

    boardEl.addEventListener('mouseover', (e) => {
      const t = e.target.closest('.sq'); if (!t || pinned) return;
      const sq = +t.dataset.sq;
      const p = OCC[sq];
      highlight(sq, (p ? NAME[p] : 'empty') + ' on ' + sqName(sq) +
        ' → token <' + (p || '.') + '> at span offset ' + (sq + 1) + ' of 72');
    });
    tokEl.addEventListener('mouseover', (e) => {
      const t = e.target.closest('.tk'); if (!t || pinned) return;
      if (t.dataset.sq != null) highlight(+t.dataset.sq, t.dataset.note);
      else { clear(); readout.textContent = t.dataset.note; }
    });
    [boardEl, tokEl].forEach((el) => el.addEventListener('mouseleave', clear));
    [boardEl, tokEl].forEach((el) => el.addEventListener('click', (e) => {
      if (!e.target.closest('.sq') && !e.target.closest('.tk')) return;
      pinned = !pinned;
    }));
  }
})();
