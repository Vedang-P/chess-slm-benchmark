import { Chess } from "chess.js";

const HF = "https://huggingface.co";
const REPO = "vedangfake/chess-slm-benchmark";
const RAW = "https://raw.githubusercontent.com/Vedang-P/chess-slm-benchmark/main";
const RUN = "ccgavn-5m-seed0";
const REFERENCE = [
  { name: "Ruoss 9M (teacher)", mate: 98.72, puzzles: 86.13 },
  { name: "legacy GAVN 5.3M", mate: 85.35, puzzles: 43.83 },
];
const STAGES = [
  { key: "train320", label: "CC-GAVN 320k training" },
  { key: "eval320", label: "320k frozen eval" },
  { key: "corpus1b", label: "1B corpus labeling" },
  { key: "train1b", label: "1B continuation (→1.62M steps)" },
  { key: "eval1b", label: "1B frozen eval" },
];

async function hfTree(env, path = "") {
  const out = [];
  let url = `${HF}/api/datasets/${REPO}/tree/main${path ? `/${path}` : ""}?recursive=1`;
  for (let page = 0; page < 12 && url; page++) {
    const r = await fetch(url, { headers: { Authorization: `Bearer ${env.HF_TOKEN}` } });
    if (!r.ok) break;
    const items = await r.json();
    if (Array.isArray(items)) out.push(...items);
    const link = r.headers.get("link") || "";
    const m = link.match(/<([^>]+)>;\s*rel="next"/);
    url = m ? m[1] : null;
  }
  return out;
}

async function hfFile(env, path) {
  const r = await fetch(`${HF}/datasets/${REPO}/resolve/main/${path}`, {
    headers: { Authorization: `Bearer ${env.HF_TOKEN}` },
  });
  if (!r.ok) return null;
  return await r.text();
}

async function hfJson(env, path) {
  const t = await hfFile(env, path);
  if (t === null) return null;
  try {
    return JSON.parse(t);
  } catch {
    return null;
  }
}

function parseGames(pgnText, limit = 6) {
  const games = [];
  const chunks = pgnText.split(/\n\s*\n(?=\[Event )/);
  for (const chunk of chunks) {
    if (games.length >= limit) break;
    const headers = {};
    for (const line of chunk.split("\n")) {
      const m = line.match(/^\[(\w+)\s+"(.*)"\]\s*$/);
      if (m) headers[m[1]] = m[2];
    }
    if (!headers.Event) continue;
    try {
      const replay = new Chess();
      replay.loadPgn(chunk);
      const history = replay.history({ verbose: true });
      const walk = new Chess();
      const moves = [];
      for (const mv of history) {
        walk.move(mv.san);
        moves.push({ san: mv.san, fen: walk.fen() });
      }
      if (moves.length) {
        games.push({
          event: headers.Event,
          white: headers.White || "?",
          black: headers.Black || "?",
          result: headers.Result || "*",
          moves,
        });
      }
    } catch {
      continue;
    }
  }
  return games;
}

function computeStages(snap) {
  const curve = snap.curve || [];
  const lastStep = curve.length ? curve[curve.length - 1].step : 0;
  const stages = {};
  const ckpt320 = lastStep >= 320000;
  stages.train320 = ckpt320 ? 100 : Math.min(100, (lastStep / 320000) * 100);
  stages.eval320 = snap.eval320_done ? 100 : (ckpt320 ? 50 : 0);
  const corpus = snap.corpus || {};
  stages.corpus1b = Math.min(100, ((corpus.labeled_rows || 0) / 920000000) * 100);
  stages.train1b = lastStep >= 1620000 ? 100
    : ckpt320 ? Math.max(0, Math.min(100, ((lastStep - 320000) / 1300000) * 100)) : 0;
  stages.eval1b = snap.eval1b_done ? 100 : 0;
  return stages;
}

async function refresh(env) {
  const prev = (await env.SNAPSHOT.get("snapshot", "json")) || {};
  const snap = {
    updated_at: new Date().toISOString(),
    curve: prev.curve || [],
    evals: prev.evals || {},
    runs: prev.runs || {},
    games: prev.games || [],
    kernels: prev.kernels || [],
    quota: prev.quota || {},
    stages: {},
    corpus: {},
    errors: [],
  };

  let files = [];
  try {
    files = await hfTree(env);
  } catch (e) {
    snap.errors.push(`hf tree: ${e}`);
  }
  const paths = new Set(files.map((f) => f.path));

  // ---- training curve (checkpoint metrics; fetch only new steps) ----
  const known = new Set(snap.curve.map((p) => p.step));
  const metrics = files
    .map((f) => f.path)
    .filter((p) => p && /^ccgavn-5m-seed0\/checkpoint-\d+\/metrics\.json$/.test(p))
    .map((p) => ({ p, step: parseInt(p.match(/checkpoint-(\d+)/)[1], 10) }))
    .filter((x) => !known.has(x.step))
    .sort((a, b) => a.step - b.step);
  for (const { p, step } of metrics.slice(0, 10)) {
    const m = await hfJson(env, p);
    if (m && typeof m.train_loss === "number") {
      snap.curve.push({ step, train: m.train_loss, dev: m.dev_loss ?? null });
    }
  }
  snap.curve.sort((a, b) => a.step - b.step);
  if (snap.curve.length > 5000) snap.curve = snap.curve.slice(-5000);

  // ---- frozen eval summaries ----
  const evalPaths = [...paths].filter((p) => /^eval-results\/.*\/eval-summary\.json$/.test(p));
  for (const p of evalPaths) {
    if (snap.evals[p]) continue;
    const s = await hfJson(env, p);
    if (s) snap.evals[p] = { ...s, fetched_at: new Date().toISOString() };
  }

  // ---- run status files (small, refetch each time) ----
  for (const p of [...paths].filter((x) => /^[^/]+\/run-status\.txt$/.test(x)).slice(0, 4)) {
    const t = await hfFile(env, p);
    if (t !== null) snap.runs[p] = t.slice(0, 400);
  }

  // ---- corpus progress (rows per shard from the repo manifest) ----
  let rowsMap = {};
  try {
    const r = await fetch(`${RAW}/kernels/build-2b/shard_rows.json`);
    if (r.ok) rowsMap = await r.json();
  } catch { /* fall back to shard count only */ }
  const newShards = Object.keys(rowsMap);
  const doneShards = newShards.filter(
    (s) => paths.has(`chessbench-full-build/shard-${s}/teacher_logp.npy`));
  let labeled = doneShards.reduce((a, s) => a + (rowsMap[s] || 0), 0);
  if (!newShards.length) {
    labeled = files.filter((f) => /^chessbench-full-build\/shard-[0-9A-Z]+\/teacher_logp\.npy$/.test(f.path)).length * 18000000;
  }
  snap.corpus = {
    target_rows: 920000000,
    labeled_rows: labeled,
    shards_planned: newShards.length || 108,
    shards_done: doneShards.length,
    original_rows: 94277038,
  };

  // ---- stages ----
  snap.stages = computeStages(snap);
    [...paths].some((p) => /eval-results\/.*1b.*\/eval-summary\.json$/.test(p)) ? 100 : 0;

  // ---- games for the viewer (ladder PGNs + any future ones) ----
  if (!snap.games.length) {
    const pgnPaths = [...paths].filter((p) => /^elo-results\/.*\/games-\d+\.pgn$/.test(p)).slice(0, 1);
    const all = [];
    for (const p of pgnPaths) {
      const t = await hfFile(env, p);
      if (t) all.push(...parseGames(t, 6 - all.length));
    }
    snap.games = all;
  }

  snap.reference = REFERENCE;
  await env.SNAPSHOT.put("snapshot", JSON.stringify(snap));
  return snap;
}

export default {
  async scheduled(event, env, ctx) {
    ctx.waitUntil(refresh(env));
  },

  async fetch(req, env) {
    const url = new URL(req.url);

    if (url.pathname === "/api/snapshot") {
      const snap = (await env.SNAPSHOT.get("snapshot", "json")) || {};
      snap.stages = computeStages(snap);
      snap.updated_at = snap.updated_at || snap.ingested_at || new Date().toISOString();
      return Response.json(snap, { headers: { "cache-control": "no-store" } });
    }

    if (url.pathname === "/api/ingest" && req.method === "POST") {
      if (req.headers.get("x-ingest-key") !== env.INGEST_KEY) {
        return new Response("forbidden", { status: 403 });
      }
      const body = await req.json().catch(() => null);
      if (!body) return new Response("bad json", { status: 400 });
      const snap = (await env.SNAPSHOT.get("snapshot", "json")) || {};
      for (const k of ["curve", "evals", "corpus", "games", "runs", "kernels", "quota", "live"]) {
        if (k in body) snap[k] = body[k];
      }
      snap.ingested_at = new Date().toISOString();
      snap.updated_at = snap.ingested_at;
      await env.SNAPSHOT.put("snapshot", JSON.stringify(snap));
      return Response.json({ ok: true });
    }

    if (url.pathname === "/api/refresh" && url.searchParams.get("key") === env.INGEST_KEY) {
      const snap = await refresh(env);
      return Response.json({ ok: true, stages: snap.stages, corpus: snap.corpus });
    }

    return new Response(HTML, {
      headers: { "content-type": "text/html;charset=utf-8", "cache-control": "no-store" },
    });
  },
};

const HTML = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chess SLM — mission control</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#0b0e14;--panel:#131826;--panel2:#1a2133;--line:#232c42;--text:#e8ecf5;--mut:#8b96ad;--accent:#5eead4;--accent2:#818cf8;--warn:#fbbf24;--bad:#f87171;--good:#34d399}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:16px}
header{display:flex;flex-wrap:wrap;align-items:baseline;gap:10px;margin:8px 0 18px}
h1{font-size:20px;margin:0;letter-spacing:.3px}
.sub{color:var(--mut);font-size:12px}
h2{font-size:14px;text-transform:uppercase;letter-spacing:1.2px;color:var(--mut);margin:26px 0 10px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px}
.grid{display:grid;gap:14px}
.cols2{grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}
.stages{display:grid;gap:10px}
.stage{display:grid;grid-template-columns:26px 1fr auto;align-items:center;gap:10px}
.dot{width:14px;height:14px;border-radius:50%;background:#2a3350;border:1px solid var(--line)}
.dot.done{background:var(--good);border-color:transparent}
.dot.active{background:var(--warn);border-color:transparent;box-shadow:0 0 0 4px rgba(251,191,36,.15)}
.bar{height:8px;background:#1f2740;border-radius:99px;overflow:hidden;margin-top:6px}
.bar>i{display:block;height:100%;background:linear-gradient(90deg,var(--accent2),var(--accent));width:0%}
.stage .pct{color:var(--mut);font-size:12px;min-width:42px;text-align:right}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line)}
th{color:var(--mut);font-weight:600}
.k{color:var(--mut);font-size:12px}
.num{font-variant-numeric:tabular-nums}
.tag{display:inline-block;padding:2px 8px;border-radius:99px;font-size:11px;border:1px solid var(--line);color:var(--mut)}
.tag.run{color:var(--good);border-color:rgba(52,211,153,.4)}
.tag.dead{color:var(--bad);border-color:rgba(248,113,113,.4)}
canvas{width:100%!important;max-height:320px}
.gamewrap{display:grid;gap:14px;grid-template-columns:minmax(240px,320px) 1fr}
@media(max-width:700px){.gamewrap{grid-template-columns:1fr}}
.board{display:grid;grid-template-columns:repeat(8,1fr);aspect-ratio:1/1;border-radius:10px;overflow:hidden;border:1px solid var(--line)}
.sq{display:flex;align-items:center;justify-content:center;font-size:min(5.2vw,30px)}
.sq.light{background:#cdd6e6}.sq.dark{background:#7c8aa5}
.movelist{max-height:320px;overflow:auto;font-variant-numeric:tabular-nums;font-size:13px}
.movelist span{padding:2px 5px;border-radius:6px;margin-right:4px;cursor:pointer;display:inline-block}
.movelist span.hl{background:rgba(94,234,212,.18);color:var(--accent)}
.controls{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
button{background:var(--panel2);border:1px solid var(--line);color:var(--text);padding:7px 12px;border-radius:9px;font-size:13px}
button:active{transform:translateY(1px)}
select{background:var(--panel2);color:var(--text);border:1px solid var(--line);border-radius:9px;padding:7px}
.err{color:var(--warn);font-size:12px}
.foot{color:var(--mut);font-size:12px;margin:26px 0 40px}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>Chess SLM — mission control</h1>
  <span class="sub" id="updated">loading…</span>
  <span class="sub" style="margin-left:auto"><span class="tag" id="live"></span></span>
</header>

<div class="card">
  <h2 style="margin-top:0">Stages</h2>
  <div class="stages" id="stages"></div>
</div>

<div class="grid cols2" style="margin-top:14px">
  <div class="card">
    <h2 style="margin-top:0">Loss curve (CC-GAVN seed 0)</h2>
    <canvas id="losses" height="220"></canvas>
  </div>
  <div class="card">
    <h2 style="margin-top:0">Frozen evals</h2>
    <canvas id="evals" height="220"></canvas>
    <table id="evaltable" style="margin-top:10px"></table>
  </div>
</div>

<div class="grid cols2" style="margin-top:14px">
  <div class="card">
    <h2 style="margin-top:0">1B corpus</h2>
    <div id="corpus"></div>
  </div>
  <div class="card">
    <h2 style="margin-top:0">Runs</h2>
    <table id="runs"></table>
  </div>
</div>

<div class="card" style="margin-top:14px">
  <h2 style="margin-top:0">Eval games — CC-GAVN vs Stockfish</h2>
  <div class="gamewrap">
    <div>
      <select id="gamesel"></select>
      <div class="board" id="board" style="margin-top:10px"></div>
      <div class="controls">
        <button id="prev">‹ prev</button>
        <button id="next">next ›</button>
        <button id="play">▶ autoplay</button>
        <button id="reset">⟲ start</button>
      </div>
      <div class="k" id="gameresult" style="margin-top:8px"></div>
    </div>
    <div>
      <div class="movelist" id="movelist"></div>
    </div>
  </div>
</div>

<div class="foot" id="errors"></div>
</div>

<script>
const PIECES={p:'♟',n:'♞',b:'♝',r:'♜',q:'♛',k:'♚',P:'♙',N:'♘',B:'♗',R:'♖',Q:'♕',K:'♔'};
let snap={}, curveChart, evalChart, game=null, ply=0, timer=null;

function boardHTML(fen){
  const rows=(fen||'8/8/8/8/8/8/8/8').split(' ')[0].split('/');
  let h='';
  rows.forEach((row,r)=>{
    let f=0;
    for(const ch of row){
      if(/\\d/.test(ch)){for(let i=0;i<+ch;i++){h+=sq(r,f,'');f++;}}
      else{h+=sq(r,f,PIECES[ch]||'');f++;}
    }
  });
  return h;
}
function sq(r,f,p){const dark=(r+f)%2===1;return '<div class="sq '+(dark?'dark':'light')+'">'+p+'</div>';}

function renderStages(s){
  const names={train320:'CC-GAVN 320k training',eval320:'320k frozen eval',corpus1b:'1B corpus labeling',train1b:'1B continuation (→ 1.62M steps)',eval1b:'1B frozen eval'};
  document.getElementById('stages').innerHTML=Object.keys(names).map(k=>{
    const p=Math.round(s[k]||0);
    const cls=p>=100?'done':(p>0?'active':'');
    return '<div><div class="stage"><span class="dot '+cls+'"></span><span>'+names[k]+'</span><span class="pct">'+p+'%</span></div><div class="bar"><i style="width:'+p+'%"></i></div></div>';
  }).join('');
}

function renderLosses(curve){
  const pts=curve.filter(p=>typeof p.train==='number');
  const el=document.getElementById('losses');
  if(!el||!window.Chart){return;}
  if(curveChart) curveChart.destroy();
  curveChart=new Chart(el,{type:'line',data:{labels:pts.map(p=>p.step),datasets:[
    {label:'train',data:pts.map(p=>p.train),borderColor:'#818cf8',pointRadius:0,borderWidth:2},
    {label:'dev',data:pts.map(p=>p.dev),borderColor:'#5eead4',pointRadius:0,borderWidth:2,spanGaps:true}]},
    options:{animation:false,plugins:{legend:{labels:{color:'#8b96ad'}}},scales:{
      x:{ticks:{color:'#8b96ad',maxTicksLimit:6},grid:{color:'#1f2740'}},
      y:{ticks:{color:'#8b96ad'},grid:{color:'#1f2740'}}}}});
}

function renderEvals(evals,reference){
  const rows=[];
  for(const [path,s] of Object.entries(evals||{})){
    const mate=s.mate&&s.mate[0]?s.mate[0].pct:null;
    const puz=s.puzzles&&s.puzzles[0]?s.puzzles[0].pct:null;
    const label=path.replace(/^eval-results\\//,'').replace(/\\/eval-summary\\.json$/,'');
    rows.push({label,mate,puz});
  }
  rows.sort((a,b)=>(b.mate||0)-(a.mate||0));
  for(const r of (reference||[])) rows.push({label:r.name,mate:r.mate,puz:r.puzzles,ref:true});
  const el=document.getElementById('evals');
  if(el&&window.Chart){
    if(evalChart) evalChart.destroy();
    evalChart=new Chart(el,{type:'bar',data:{labels:rows.map(r=>r.label),datasets:[
      {label:'MATE %',data:rows.map(r=>r.mate),backgroundColor:'#818cf8'},
      {label:'Puzzles %',data:rows.map(r=>r.puz),backgroundColor:'#5eead4'}]},
      options:{animation:false,plugins:{legend:{labels:{color:'#8b96ad'}}},scales:{
        x:{ticks:{color:'#8b96ad',maxRotation:50,minRotation:0,font:{size:10}},grid:{display:false}},
        y:{ticks:{color:'#8b96ad'},grid:{color:'#1f2740'},suggestedMax:100}}}}});
  }
  document.getElementById('evaltable').innerHTML='<tr><th>eval</th><th>MATE</th><th>puzzles</th></tr>'+
    rows.map(r=>'<tr><td>'+r.label+'</td><td class="num">'+(r.mate==null?'—':r.mate.toFixed(2)+'%')+'</td><td class="num">'+(r.puz==null?'—':r.puz.toFixed(2)+'%')+'</td></tr>').join('');
}

function renderCorpus(c){
  const pct=Math.min(100,c.target_rows?c.labeled_rows/c.target_rows*100:0);
  document.getElementById('corpus').innerHTML=
    '<div class="k">labeled new rows</div><div style="font-size:22px" class="num">'+(c.labeled_rows/1e6).toFixed(0)+'M <span class="k">/ 920M</span></div>'+
    '<div class="bar" style="margin-top:10px"><i style="width:'+pct.toFixed(1)+'%"></i></div>'+
    '<div class="k" style="margin-top:8px">shards: '+c.shards_done+' / '+c.shards_planned+' · existing corpus '+((c.original_rows||0)/1e6).toFixed(0)+'M rows</div>';
}

function renderRuns(snap){
  const t=document.getElementById('runs');
  let html='<tr><th>run</th><th>status</th></tr>';
  for(const [p,v] of Object.entries(snap.runs||{})){
    const s=String(v).trim().split('\\n')[0].slice(0,80);
    const cls=/DONE/i.test(s)?'run':(/error|Traceback/i.test(s)?'dead':'');
    html+='<tr><td>'+p.split('/')[0]+'</td><td><span class="tag '+cls+'">'+s+'</span></td></tr>';
  }
  for(const k of (snap.kernels||[])){
    const cls=/RUNNING|QUEUED/i.test(k.status||'')?'run':'';
    html+='<tr><td>'+k.account+'/'+k.kernel+'</td><td><span class="tag '+cls+'">'+(k.status||'?')+'</span></td></tr>';
  }
  for(const [a,q] of Object.entries(snap.quota||{})){
    html+='<tr><td class="k">'+a+' quota</td><td class="k num">'+q+'h GPU left</td></tr>';
  }
  t.innerHTML=html;
}

function selectGame(i){
  if(!snap.games||!snap.games.length){document.getElementById('board').innerHTML='';document.getElementById('movelist').innerHTML='<span class="k">no games archived yet</span>';return;}
  game=snap.games[i]; ply=0; draw();
  document.getElementById('gameresult').textContent=game.white+' vs '+game.black+' — '+game.result+' ('+game.moves.length+' plies)';
}
function draw(){
  if(!game) return;
  const start='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
  const fen=ply===0?start:game.moves[ply-1].fen;
  document.getElementById('board').innerHTML=boardHTML(fen);
  document.getElementById('movelist').innerHTML=game.moves.map((m,i)=>{
    const n=Math.floor(i/2)+1; const pre=(i%2===0)?n+'. ':''; 
    return '<span class="'+(i===ply-1?'hl':'')+'" onclick="jump('+(i+1)+')">'+pre+m.san+'</span>';
  }).join('');
}
function jump(p){ply=Math.max(0,Math.min(game.moves.length,p));draw();}
window.jump=jump;

async function load(){
  const r=await fetch('/api/snapshot',{cache:'no-store'});
  snap=await r.json();
  document.getElementById('updated').textContent='updated '+new Date(snap.updated_at||Date.now()).toLocaleString();
  document.getElementById('live').textContent=(snap.ingested_at?'ingest '+new Date(snap.ingested_at).toLocaleTimeString():'no ingest yet');
  renderStages(snap.stages||{});
  renderLosses(snap.curve||[]);
  renderEvals(snap.evals||{},snap.reference||[]);
  renderCorpus(snap.corpus||{});
  renderRuns(snap);
  const sel=document.getElementById('gamesel');
  sel.innerHTML=(snap.games||[]).map((g,i)=>'<option value="'+i+'">'+g.white+' vs '+g.black+' ('+g.result+')</option>').join('');
  sel.onchange=e=>selectGame(+e.target.value);
  if(snap.games&&snap.games.length&&!game) selectGame(0);
  document.getElementById('errors').textContent=(snap.errors||[]).join(' · ');
}
document.getElementById('prev').onclick=()=>jump(ply-1);
document.getElementById('next').onclick=()=>jump(ply+1);
document.getElementById('reset').onclick=()=>jump(0);
document.getElementById('play').onclick=()=>{
  if(timer){clearInterval(timer);timer=null;return;}
  timer=setInterval(()=>{if(!game||ply>=game.moves.length){clearInterval(timer);timer=null;return;}jump(ply+1);},900);
};
const _c=new Chess?null:null;
load();setInterval(load,60000);
</script>
<script>/* chess.js not needed client-side; FENs are precomputed. */
window.Chess=window.Chess||function(){};</script>
</body>
</html>`;
