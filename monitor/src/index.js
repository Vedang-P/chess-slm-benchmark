const GITHUB = "https://api.github.com/repos/Vedang-P/chess-slm-benchmark/dispatches";
const BASELINE = { mate: 92.75, puzzles: 76.20, params: 4_762_088 };
const TEACHER = { mate: 98.72, puzzles: 86.13 };
const RL_RUN = "ccgavn-rl-root-v1-seed0";

async function dispatchTick(env) {
  if (!env.GH_TOKEN) return;
  const previous = Number((await env.SNAPSHOT.get("last_tick")) || 0);
  if (Date.now() - previous < 8 * 60_000) return;
  const response = await fetch(GITHUB, {
    method: "POST",
    headers: { Authorization: `Bearer ${env.GH_TOKEN}`,
               Accept: "application/vnd.github+json", "User-Agent": "ccgavn-rl-monitor" },
    body: JSON.stringify({ event_type: "tick" }),
  });
  if (!response.ok && response.status !== 204) {
    throw new Error(`GitHub dispatch HTTP ${response.status}`);
  }
  await env.SNAPSHOT.put("last_tick", String(Date.now()));
}

const SECURITY = {
  "x-content-type-options": "nosniff",
  "referrer-policy": "no-referrer",
  "x-frame-options": "DENY",
  "content-security-policy": "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
};

export default {
  async scheduled(_event, env, ctx) {
    ctx.waitUntil(dispatchTick(env));
  },

  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/api/snapshot" && request.method === "GET") {
      const snapshot = (await env.SNAPSHOT.get("rl_snapshot", "json")) || {};
      return Response.json(snapshot, { headers: { ...SECURITY, "cache-control": "no-store" } });
    }
    if (url.pathname === "/api/ingest" && request.method === "POST") {
      if (!env.INGEST_KEY || request.headers.get("x-ingest-key") !== env.INGEST_KEY) {
        return new Response("forbidden", { status: 403, headers: SECURITY });
      }
      const body = await request.json().catch(() => null);
      if (!body || body.run !== RL_RUN || !Array.isArray(body.labels) ||
          !Array.isArray(body.accounts) || !body.training) {
        return new Response("invalid RL snapshot", { status: 400, headers: SECURITY });
      }
      await env.SNAPSHOT.put("rl_snapshot", JSON.stringify(body));
      return Response.json({ ok: true }, { headers: SECURITY });
    }
    if (url.pathname.startsWith("/api/")) {
      return new Response("not found", { status: 404, headers: SECURITY });
    }
    return new Response(HTML, { headers: { ...SECURITY,
      "content-type": "text/html; charset=utf-8", "cache-control": "no-cache" } });
  },
};

const HTML = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#0e1110">
<title>CC-GAVN · RL monitor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{color-scheme:dark;--bg:#0e1110;--surface:#171c19;--line:#303a32;--text:#ecf3e8;--muted:#aebcaf;--soft:#778779;--accent:#b7ea79;--bad:#f0a99c}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--text);font:16px/1.5 "DM Sans",sans-serif}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}a:focus-visible{outline:2px solid var(--accent);outline-offset:4px;border-radius:3px}
.shell{max-width:1120px;margin:auto;padding:0 28px}.top{border-bottom:1px solid var(--line)}.top .shell{min-height:74px;display:flex;align-items:center;justify-content:space-between;gap:24px}
.brand{font-weight:700;letter-spacing:-.03em;font-size:1.18rem;color:var(--text)}.toplinks{display:flex;gap:24px;font-size:.88rem;font-weight:600}
main{padding-top:58px;padding-bottom:70px}.hero{display:flex;align-items:end;justify-content:space-between;gap:28px;margin-bottom:54px}.hero h1{font-size:clamp(2.5rem,6vw,5rem);line-height:1.02;letter-spacing:-.035em;margin:0;max-width:720px}
.hero p{color:var(--muted);max-width:46ch;margin:18px 0 0}.status{border:1px solid var(--line);border-radius:14px;background:var(--surface);min-width:208px;padding:17px 20px}
.status small{display:block;color:var(--muted);font-size:.78rem}.status strong{display:block;color:var(--accent);font-size:1.05rem;text-transform:capitalize;margin-top:4px}
.section{margin-top:48px}.sectionhead{display:flex;justify-content:space-between;align-items:baseline;gap:16px;margin-bottom:15px}.sectionhead h2{font-size:1.45rem;letter-spacing:-.025em;margin:0}.sectionhead span{color:var(--muted);font-size:.86rem}
.overview{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.stat{padding:22px;background:var(--surface);border:1px solid var(--line);border-radius:14px;min-height:156px}
.stat .label{color:var(--muted);font-size:.86rem}.stat .value{font:600 clamp(2rem,4vw,3rem)/1.1 "IBM Plex Mono",monospace;letter-spacing:-.035em;margin:13px 0 7px}.stat .note{font-size:.8rem;color:var(--muted)}
.progress{height:8px;border-radius:5px;background:#2a342b;overflow:hidden;margin:18px 0 0}.progress i{display:block;height:100%;background:var(--accent);width:100%;transform:scaleX(0);transform-origin:left;transition:transform .5s ease-out}
.phases{border-top:1px solid var(--line)}.phase{display:grid;grid-template-columns:48px minmax(160px,1fr) minmax(120px,2fr) 130px;gap:18px;align-items:center;padding:21px 0;border-bottom:1px solid var(--line)}
.phase .n{font:500 .82rem "IBM Plex Mono",monospace;color:var(--soft)}.phase .name{font-weight:700}.phase .detail{color:var(--muted);font-size:.89rem}.phase .state{text-align:right;font:500 .8rem "IBM Plex Mono",monospace;color:var(--muted)}.phase .state.live,.phase .state.done{color:var(--accent)}.phase .state.fail{color:var(--bad)}
.scoreline{display:grid;grid-template-columns:1.3fr repeat(3,1fr);gap:12px;padding:13px 0;border-bottom:1px solid var(--line);align-items:baseline}.scoreline.header{color:var(--muted);font-size:.8rem;padding-top:0}.scoreline strong{font-weight:600}.mono{font-family:"IBM Plex Mono",monospace}.goal{color:var(--accent)}.subnote{margin:14px 0 0;color:var(--muted);font-size:.85rem;max-width:74ch}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:14px}.accounts{width:100%;border-collapse:collapse;min-width:600px}.accounts th,.accounts td{text-align:left;padding:13px 18px;border-bottom:1px solid var(--line)}.accounts tr:last-child td{border-bottom:0}.accounts th{color:var(--muted);font-size:.78rem;font-weight:600}.accounts td{font-size:.87rem}.accounts td:first-child{font-family:"IBM Plex Mono",monospace}.accounts .bar{display:inline-block;vertical-align:middle;width:100px;height:5px;background:#2a342b;border-radius:3px;margin-right:8px}.accounts .bar i{display:block;background:var(--accent);height:100%;border-radius:3px}
.error{border:1px solid #9e5a4e;background:#33221e;color:#ffd9d2;border-radius:12px;padding:14px 18px;margin-top:24px}.links{display:flex;flex-wrap:wrap;gap:20px;font-size:.86rem;margin-top:32px}.foot{border-top:1px solid var(--line);padding:24px 0;color:var(--muted);font-size:.8rem}
@media(max-width:760px){.shell{padding:0 18px}.top .shell{min-height:66px}.toplinks{gap:13px}.toplinks a:last-child{display:none}main{padding-top:38px}.hero{display:block;margin-bottom:36px}.status{display:inline-block;margin-top:27px;min-width:170px}.overview{grid-template-columns:1fr}.stat{min-height:0}.phase{grid-template-columns:28px 1fr 86px;gap:10px}.phase .detail{grid-column:2/4;grid-row:2}.scoreline{grid-template-columns:1.2fr 1fr 1fr}.scoreline>*:last-child{display:none}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}.progress i{transition:none}}
</style>
</head>
<body>
<header class="top"><div class="shell"><a class="brand" href="/">CC-GAVN</a><nav class="toplinks" aria-label="Project links"><a href="#pipeline">Pipeline</a><a href="#results">Results</a><a href="https://github.com/Vedang-P/chess-slm-benchmark" target="_blank" rel="noopener noreferrer">GitHub ↗</a></nav></div></header>
<main class="shell">
<div class="hero"><div><h1>Repaired model.<br>Reinforcement learning.</h1><p>One pass per legal move. Five Kaggle accounts, exact Stockfish verifier labels, and a frozen chess evaluation at the end.</p></div><div class="status"><small>Current phase</small><strong id="phase-label">Connecting</strong></div></div>
<section class="overview" aria-label="Current progress">
  <div class="stat"><div class="label">Verifier labels</div><div class="value" id="label-count">—</div><div class="note" id="label-note">of 10,000 exact depth 14 positions</div><div class="progress" aria-hidden="true"><i id="label-bar"></i></div></div>
  <div class="stat"><div class="label">RL updates</div><div class="value" id="train-step">—</div><div class="note" id="train-note">of 20,000 steps · T4 pool</div><div class="progress" aria-hidden="true"><i id="train-bar"></i></div></div>
  <div class="stat"><div class="label">Puzzle result</div><div class="value" id="puzzle-value">Pending</div><div class="note" id="puzzle-note">Target: beat 86.13% on 10,000 puzzles</div></div>
</section>
<section class="section" id="pipeline"><div class="sectionhead"><h2>Run timeline</h2><span id="checkpoint">New RL run</span></div><div class="phases" id="phases"></div></section>
<section class="section" id="results"><div class="sectionhead"><h2>Frozen protocol</h2><span>4,000 MATE rows · 10,000 puzzles</span></div><div class="scoreline header"><span>Model</span><span>MATE</span><span>Puzzles</span><span>Parameters</span></div><div class="scoreline"><strong>Current baseline</strong><span class="mono">92.75%</span><span class="mono">76.20%</span><span class="mono">4.76M</span></div><div class="scoreline"><strong>9M teacher</strong><span class="mono">98.72%</span><span class="mono">86.13%</span><span class="mono">9M</span></div><div class="scoreline"><strong>RL checkpoint</strong><span class="mono" id="rl-mate">Pending</span><span class="mono goal" id="rl-puzzle">Pending</span><span class="mono" id="rl-params">3.90M</span></div><p class="subnote" id="result-note">RL numbers appear only after the complete checkpoint scoped evaluation has all 4,000 MATE rows and 10,000 puzzles.</p></section>
<section class="section"><div class="sectionhead"><h2>Account pool</h2><span>Verifier jobs use CPU; training rotates onto available T4 quota</span></div><div class="tablewrap"><table class="accounts"><thead><tr><th>Account</th><th>Labels</th><th>Verifier</th><th>T4 hours left</th><th>RL worker</th></tr></thead><tbody id="account-rows"><tr><td colspan="5">Waiting for first update</td></tr></tbody></table></div></section>
<div class="error" id="problem" hidden></div><div class="links" id="artifact-links"></div>
</main><footer class="foot"><div class="shell" id="freshness">Waiting for pipeline snapshot · refreshes every minute</div></footer>
<script>
const goal=86.13;
const byId=id=>document.getElementById(id);
const esc=value=>String(value??"").replace(/[&<>\"']/g,ch=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch]));
const num=value=>Number(value||0).toLocaleString();
const pct=(n,d)=>d?Math.max(0,Math.min(100,100*n/d)):0;
function stage(name,detail,state,index){return '<div class="phase"><span class="n">'+index+'</span><span class="name">'+esc(name)+'</span><span class="detail">'+esc(detail)+'</span><span class="state '+esc(state)+'">'+esc(state)+'</span></div>';}
function render(s){
  const labels=Array.isArray(s.labels)?s.labels:[];
  const accounts=Array.isArray(s.accounts)?s.accounts:[];
  const training=s.training||{};
  const evaluation=s.evaluation||null;
  const labelCount=Number(s.label_count||0), labelTarget=Number(s.label_target||10000), step=Number(training.step||0), target=Number(training.target||20000);
  const labelDone=labelCount>=labelTarget, trainDone=step>=target, evalDone=!!evaluation;
  const phase=evalDone?'Complete':trainDone?'Evaluation':step>0?'RL training':'Verifier labels';
  byId('phase-label').textContent=phase;
  byId('label-count').textContent=num(labelCount);byId('label-note').textContent='of '+num(labelTarget)+' positions · depth '+(s.label_depth||14);byId('label-bar').style.transform='scaleX('+(pct(labelCount,labelTarget)/100)+')';
  byId('train-step').textContent=num(step);byId('train-note').textContent='of '+num(target)+' RL updates · T4 pool';byId('train-bar').style.transform='scaleX('+(pct(step,target)/100)+')';
  byId('checkpoint').textContent=training.checkpoint||'Awaiting first checkpoint';
  let mate=null,puzzle=null;
  if(evalDone){mate=Number(evaluation.mate.correct)/Number(evaluation.mate.total)*100;puzzle=Number(evaluation.puzzles.solved)/Number(evaluation.puzzles.total)*100;}
  byId('puzzle-value').textContent=evalDone?puzzle.toFixed(2)+'%':'Pending';
  byId('puzzle-note').textContent=evalDone?(puzzle>goal?'Above teacher by '+(puzzle-goal).toFixed(2)+' points':'Gap to teacher: '+(goal-puzzle).toFixed(2)+' points'):'Target: beat 86.13% on 10,000 puzzles';
  byId('rl-mate').textContent=evalDone?mate.toFixed(2)+'%':'Pending';byId('rl-puzzle').textContent=evalDone?puzzle.toFixed(2)+'%':'Pending';
  byId('rl-params').textContent=training.parameter_count?(Number(training.parameter_count)/1e6).toFixed(2)+'M':'3.90M';
  byId('result-note').textContent=evalDone?'Complete frozen evaluation: '+num(evaluation.mate.correct)+'/4,000 MATE, '+num(evaluation.puzzles.solved)+'/10,000 puzzles. Historical benchmark exposure applies.':'RL numbers appear only after the complete checkpoint scoped evaluation has all 4,000 MATE rows and 10,000 puzzles.';
  byId('phases').innerHTML=stage('Verifier labels',num(labelCount)+' / '+num(labelTarget)+' all legal move roots',labelDone?'done':labelCount?'live':'waiting','01')+
    stage('RL post training',num(step)+' / '+num(target)+' updates',trainDone?'done':step?'live':'waiting','02')+
    stage('Frozen evaluation',evalDone?'4,000 MATE rows and 10,000 puzzles complete':trainDone?'Scoring final checkpoint':'Waiting for final checkpoint',evalDone?'done':trainDone?'live':'waiting','03');
  byId('account-rows').innerHTML=labels.length?labels.map((item,i)=>{const pool=accounts[i]||{};return '<tr><td>'+esc(item.account)+'</td><td><span class="bar"><i style="width:'+pct(item.count,item.target)+'%"></i></span>'+num(item.count)+' / '+num(item.target)+'</td><td>'+esc(item.kernel)+'</td><td>'+esc(pool.gpu_hours==null?'—':Number(pool.gpu_hours).toFixed(1)+'h')+'</td><td>'+esc(pool.learner||'—')+'</td></tr>';}).join(''):'<tr><td colspan="5">Waiting for first update</td></tr>';
  const errors=[];
  for(const item of labels)if(/FAILED|ERROR/.test(item.status||''))errors.push(item.account+': '+item.status);
  if(/FAILED|ERROR/.test(training.status||''))errors.push('RL: '+training.status.split('\\n').at(-1));
  if(/FAILED|ERROR|INCOMPLETE/.test(s.evaluation_status||''))errors.push('Evaluation: '+s.evaluation_status.split('\\n').at(-1));
  byId('problem').hidden=!errors.length;byId('problem').textContent=errors.join(' · ');
  const links=s.links||{};byId('artifact-links').innerHTML=[['GitHub Actions',links.github],['Verifier labels',links.labels],['RL checkpoints',links.checkpoints],['Evaluation',links.evaluation]].filter(x=>x[1]).map(x=>'<a href="'+esc(x[1])+'" target="_blank" rel="noopener noreferrer">'+esc(x[0])+' ↗</a>').join('');
  byId('freshness').textContent=s.updated_at?'Updated '+new Date(s.updated_at).toLocaleString()+' · refreshes every minute':'Waiting for pipeline snapshot · refreshes every minute';
}
async function load(){try{const response=await fetch('/api/snapshot',{cache:'no-store'});if(!response.ok)throw new Error('Snapshot unavailable');render(await response.json());}catch(error){byId('phase-label').textContent='Update delayed';byId('freshness').textContent='Monitor update delayed: '+error.message;}}
load();setInterval(load,60_000);
</script>
</body></html>`;
