export const HTML = String.raw`<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#081211">
<title>CC-GAVN · continuation monitor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{color-scheme:dark;--bg:#081211;--surface:#101c1a;--surface2:#14221f;--line:#29413b;--text:#e9f3ed;--muted:#9ab0a7;--dim:#708a80;--green:#bcf188;--teal:#75dfc3;--amber:#edc47e;--red:#f0938d}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 'DM Sans',sans-serif}
a{color:var(--green);text-decoration:none}a:hover{text-decoration:underline}a:focus-visible{outline:2px solid var(--green);outline-offset:3px}
.mono{font-family:'IBM Plex Mono',monospace;font-variant-numeric:tabular-nums}.wrap{max-width:1440px;margin:auto;padding:0 28px}
.top{border-bottom:1px solid var(--line);background:#0b1715;position:sticky;top:0;z-index:10}.topin{min-height:65px;display:flex;align-items:center;gap:24px}
.brand{font-weight:700;letter-spacing:-.035em;font-size:17px;color:var(--text);white-space:nowrap}.brand b{color:var(--green)}
.nav{display:flex;gap:22px;font-size:12px;margin-left:20px}.nav a{color:var(--muted)}.nav a:hover{color:var(--text)}
.right{margin-left:auto;display:flex;gap:13px;align-items:center;color:var(--muted);font:11px 'IBM Plex Mono',monospace;white-space:nowrap}
.badge{border:1px solid var(--line);border-radius:100px;padding:5px 10px;color:var(--green);text-transform:uppercase;letter-spacing:.07em}.badge:before{content:'';display:inline-block;width:7px;height:7px;margin-right:7px;border-radius:50%;background:currentColor}.badge.warn{color:var(--amber)}.badge.bad{color:var(--red)}
main{padding:32px 0 70px}.eyebrow{font:600 11px 'IBM Plex Mono',monospace;letter-spacing:.13em;text-transform:uppercase;color:var(--teal)}
h1,h2,h3,p{margin-top:0}h1{font-size:clamp(34px,4.3vw,60px);line-height:1.04;letter-spacing:-.06em;margin:11px 0 14px}h2{font-size:22px;letter-spacing:-.035em;margin:0}h3{font-size:15px;margin:0}
.hero{display:grid;grid-template-columns:1.45fr .55fr;gap:14px;margin-bottom:14px}.panel{border:1px solid var(--line);border-radius:12px;background:var(--surface)}
.heroMain{min-height:238px;padding:30px;background:radial-gradient(ellipse at 88% 12%,#234438 0%,transparent 45%),var(--surface);display:flex;flex-direction:column}.heroMain p{max-width:70ch;color:var(--muted);margin:0}
.heroMeta{display:flex;flex-wrap:wrap;gap:8px;margin-top:auto;padding-top:27px}.tag{border:1px solid #3b5548;border-radius:5px;padding:5px 8px;font:11px 'IBM Plex Mono',monospace;color:#c7ddcb}
.heroSide{padding:26px;display:flex;flex-direction:column}.cardLabel{font:11px 'IBM Plex Mono',monospace;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
.big{font:500 clamp(35px,4vw,54px)/1 'IBM Plex Mono',monospace;letter-spacing:-.06em;color:var(--green);margin:19px 0 8px}.big small{font-size:.34em;color:var(--muted)}
.small{font-size:12px;color:var(--muted)}.track{height:7px;border-radius:10px;background:#2c403a;overflow:hidden}.track i{height:100%;display:block;background:var(--green);width:0;border-radius:10px;transition:width .4s}.heroSide .track{margin:17px 0 9px}
.split{display:flex;justify-content:space-between;gap:10px;font:11px 'IBM Plex Mono',monospace;color:var(--dim)}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.stat{padding:16px 18px;min-height:113px}.stat strong{display:block;margin-top:15px;font:500 25px 'IBM Plex Mono',monospace;letter-spacing:-.05em}.stat span:last-child{font-size:11px;color:var(--muted)}
.section{margin-top:37px}.sectionHead{display:flex;justify-content:space-between;align-items:end;gap:12px;margin-bottom:13px}.sectionHead p{color:var(--muted);font-size:12px;margin:5px 0 0}.sectionHead .mono{font-size:11px;color:var(--dim)}
.chartGrid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:13px}.chart{padding:18px 20px 16px;min-height:330px}.chartTop{display:flex;justify-content:space-between;gap:10px}.chartTop strong{font:500 15px 'IBM Plex Mono',monospace;color:var(--green)}.chartSub{font-size:11px;color:var(--muted);margin:4px 0 8px}.legend{display:flex;flex-wrap:wrap;gap:4px 12px;margin:0 0 8px;font:10px 'IBM Plex Mono',monospace;color:var(--muted)}.legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}.chartBox{height:212px}.chartBox svg{width:100%;height:100%;display:block}.empty{height:100%;display:grid;place-content:center;text-align:center;color:var(--muted);font-size:12px}.empty b{color:var(--text);font-size:14px;display:block;margin-bottom:4px}
.gridline{stroke:#2a403a;stroke-width:1}.plotline{fill:none;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}.plotlabel{font:10px 'IBM Plex Mono',monospace;fill:var(--dim)}
.cols{display:grid;grid-template-columns:1.25fr .75fr;gap:13px}.tablePanel{overflow:auto}table{width:100%;border-collapse:collapse;min-width:620px}th,td{text-align:left;padding:12px 15px;border-bottom:1px solid #23372f}th{font:10px 'IBM Plex Mono',monospace;letter-spacing:.07em;text-transform:uppercase;color:var(--dim);background:var(--surface2)}td{font-size:12px}tr:last-child td{border-bottom:0}td:first-child{font-family:'IBM Plex Mono',monospace}
.events{padding:18px 19px;max-height:410px;overflow:auto}.event{border-bottom:1px solid #23372f;padding:10px 0 10px 14px;position:relative}.event:last-child{border:0}.event:before{content:'';position:absolute;left:0;top:18px;width:6px;height:6px;border-radius:50%;background:var(--teal)}.event.good:before{background:var(--green)}.event.bad:before{background:var(--red)}.event b{font-size:12px}.event time{font:10px 'IBM Plex Mono',monospace;color:var(--dim);float:right}.event p{font-size:11px;color:var(--muted);margin:4px 0 0}
.contract{display:grid;grid-template-columns:repeat(3,1fr);gap:13px}.contract .panel{padding:20px}.contract h3{margin-bottom:10px}.contract p,.contract li{font-size:12px;color:var(--muted)}.contract ul{padding-left:17px;margin-bottom:0}.links{display:flex;flex-wrap:wrap;gap:8px;margin-top:17px}.links a{border:1px solid #3c594a;border-radius:6px;padding:7px 10px;font:11px 'IBM Plex Mono',monospace}
.warning{border-left:3px solid var(--amber);background:#29271f;color:#e7d3a7;padding:12px 15px;margin-top:13px;font-size:12px;border-radius:0 6px 6px 0}.warning.bad{border-color:var(--red);background:#322321;color:#f2c0b9}.warning[hidden]{display:none}
.foot{border-top:1px solid var(--line);padding:21px 0;color:var(--dim);font-size:11px}.foot .wrap{display:flex;justify-content:space-between;gap:10px}
@media(max-width:1100px){.hero{grid-template-columns:1fr 350px}.cols{grid-template-columns:1fr}.events{max-height:260px}}
@media(max-width:760px){.wrap{padding:0 17px}.topin{flex-wrap:wrap;gap:7px;padding-top:11px;padding-bottom:10px}.nav{order:3;margin:4px 0 0;width:100%;overflow:auto}.right{gap:7px}.right #fresh{display:none}main{padding-top:20px}.hero{grid-template-columns:1fr}.heroMain{min-height:215px;padding:22px}.heroSide{padding:22px}.stats,.chartGrid,.contract{grid-template-columns:1fr 1fr}.stat strong{font-size:22px}}
@media(max-width:520px){.stats,.chartGrid,.contract{grid-template-columns:1fr}.stat{min-height:93px}.stat strong{margin-top:8px}.sectionHead{align-items:start}.sectionHead .mono{text-align:right}.chart{padding:17px 12px}.heroMain{min-height:230px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}.track i{transition:none}}
</style>
</head>
<body>
<header class="top"><div class="wrap topin"><a class="brand" href="#top"><b>◆</b> CC-GAVN <span style="color:var(--dim)">/ live research</span></a><nav class="nav"><a href="#validation">Validation</a><a href="#official">Official eval</a><a href="#pool">Pool</a><a href="#contract">Contract</a></nav><div class="right"><span id="fresh">Connecting</span><span id="badge" class="badge warn">Connecting</span></div></div></header>
<main id="top" class="wrap">
  <div class="hero"><section class="panel heroMain"><span class="eyebrow">CONSERVATIVE CONTINUATION · EXPERIMENT 02</span><h1>Restarting from 76.2%<span style="color:var(--green)">.</span></h1><p id="summary">Connecting to the live Kaggle and Hugging Face pipeline.</p><div class="heroMeta"><span class="tag">3.90M parameters</span><span class="tag">constant low LR · EMA weights</span><span class="tag">broad replay + tactical roots</span><span class="tag">5 Kaggle accounts · 11.5 h sessions</span></div></section><section class="panel heroSide"><span class="cardLabel">Main run</span><div class="big"><span id="main-step">0</span><small> steps</small></div><span class="small" id="main-caption">Awaiting the main run</span><div class="track"><i id="main-bar"></i></div><div class="split"><span id="main-status">idle</span><span id="phase-name">waiting</span></div></section></div>
  <div class="stats"><div class="panel stat"><span class="cardLabel">Best official puzzles</span><strong id="best-official">76.20%</strong><span id="best-official-note">verified starting checkpoint · gate 86.13%</span></div><div class="panel stat"><span class="cardLabel">Validation puzzles</span><strong id="val-now">—</strong><span id="val-note">held-out, rating-matched · EMA weights</span></div><div class="panel stat"><span class="cardLabel">Tactical top-1 (dev)</span><strong id="hit-now">—</strong><span>best Stockfish move picked, dense roots</span></div><div class="panel stat"><span class="cardLabel">GPU hours left</span><strong id="quota">—</strong><span>across five accounts this week</span></div></div>
  <div id="alert" class="warning" hidden></div>

  <section id="validation" class="section"><div class="sectionHead"><div><h2>Held-out validation</h2><p>4,000 Lichess puzzles disjoint from the official 10,000 and from all training data. Model selection uses only this set.</p></div><span class="mono" id="val-baseline">baseline —</span></div><div class="chartGrid"><div class="panel chart"><div class="chartTop"><h3>Validation puzzle accuracy</h3><strong id="val-best">—</strong></div><p class="chartSub">Full-sequence solve rate of the EMA weights · dashed line is the 76.2% starting model</p><div class="legend" id="legend-val"></div><div id="chart-val" class="chartBox"></div></div><div class="panel chart"><div class="chartTop"><h3>Tactical ranking</h3><strong id="hit-best">—</strong></div><p class="chartSub">Share of held-out puzzle roots where the argmax move ties the best Stockfish value</p><div class="legend" id="legend-hit"></div><div id="chart-hit" class="chartBox"></div></div></div><div class="panel tablePanel" style="margin-top:13px"><table><thead><tr><th>Run</th><th>Recipe</th><th>Step</th><th>Validation</th><th>vs start</th><th>Tactical top-1</th><th>Official</th><th>Status</th></tr></thead><tbody id="runs"><tr><td colspan="8">Awaiting the first checkpoint.</td></tr></tbody></table></div></section>

  <section id="official" class="section"><div class="sectionHead"><div><h2>Official evaluation</h2><p>All 10,000 puzzles and 4,000 MATE rows. Logged at milestones; never used to pick checkpoints.</p></div><span class="mono">start 76.20% · 9M reference 86.13%</span></div><div class="cols"><div class="panel chart"><div class="chartTop"><h3>Puzzle accuracy</h3><strong id="official-best">76.20%</strong></div><p class="chartSub">Dashed lines: starting checkpoint and the 9M reference (the success gate)</p><div class="legend" id="legend-official"></div><div id="chart-official" class="chartBox"></div></div><div class="panel chart"><div class="chartTop"><h3>Training telemetry</h3><strong id="loss-now">—</strong></div><p class="chartSub">Broad-corpus dev distillation loss · rising means forgetting</p><div class="legend" id="legend-loss"></div><div id="chart-loss" class="chartBox"></div></div></div></section>

  <section id="pool" class="section"><div class="sectionHead"><div><h2>Five-account pool</h2><p>Kaggle learner status and remaining weekly GPU quota. Sessions run one after another, each ending at about 11.5 hours.</p></div><span class="mono" id="supervisor">supervisor</span></div><div class="cols"><div class="panel tablePanel"><table><thead><tr><th>Account</th><th>GPU hours left</th><th>Learner kernel</th></tr></thead><tbody id="accounts"><tr><td colspan="3">Connecting to Kaggle status.</td></tr></tbody></table></div><div class="panel events"><h3>Live activity</h3><div id="events"><p class="small" style="margin-top:14px">Waiting for the first monitor observation.</p></div></div></div></section>

  <section id="contract" class="section"><div class="sectionHead"><div><h2>Run contract</h2><p>Fixed before the run; results are reported whether or not they clear the gate.</p></div></div><div class="contract"><article class="panel"><h3>What changed from the failed continuation</h3><ul><li>Constant 2e-5 learning rate (the old restart used 1.3e-4) with an EMA of the weights.</li><li>Every batch mixes the original corpus with whole tactical roots, so tactics cannot erase general play.</li><li>A listwise loss over each root's legal moves targets the argmax the puzzle metric scores.</li></ul></article><article class="panel"><h3>Success gate and milestones</h3><p>Headline gate: more than 86.13% on all 10,000 official puzzles. Pre-registered checkpoints: about 80% by 23 Nov and 83% by 14 Dec. The sealed test puzzles are scored once, at the end, next to the 9M reference.</p></article><article class="panel"><h3>Safeguards</h3><p>Validation runs every few thousand steps. If it falls more than 3 points below the starting model twice, the run halts itself instead of burning quota. Every session uploads weights, EMA, optimizer and metrics to Hugging Face before its cutoff.</p></article></div><div class="links" id="links"></div></section>
</main>
<footer class="foot"><div class="wrap"><span>CC-GAVN · live evidence, checkpoint-scoped</span><span>Refreshes every minute · source of truth: Kaggle + Hugging Face</span></div></footer>
<script>
const $=id=>document.getElementById(id);
const n=(v,d=0)=>Number.isFinite(Number(v))?Number(v):d;
const fmt=v=>Math.round(n(v)).toLocaleString();
const esc=s=>String(s==null?'':s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const age=s=>{const d=Date.now()-Date.parse(s||'');return Number.isFinite(d)?(d<60000?'just now':d<3600000?Math.floor(d/60000)+'m ago':Math.floor(d/3600000)+'h ago'):'unknown';};
const COLORS={main:'#bcf188',a0:'#9ab0a7',a1:'#75dfc3',a2:'#edc47e',a3:'#f0938d',a4:'#b69cf0'};
const short=run=>String(run).replace('ccgavn-cont1-','');
function empty(id,title,detail){$(id).innerHTML='<div class="empty"><b>'+esc(title)+'</b>'+esc(detail)+'</div>';}
function legend(id,series){$(id).innerHTML=series.filter(s=>s.points.length).map(s=>'<span><i style="background:'+s.color+'"></i>'+esc(s.name)+'</span>').join('');}
function plot(id,series,limits,markers){
  const rows=series.flatMap(s=>s.points||[]).filter(p=>Number.isFinite(p[0])&&Number.isFinite(p[1]));
  if(!rows.length){empty(id,'Awaiting measurements','This chart fills after a verified checkpoint.');return;}
  const W=620,H=206,L=46,R=14,T=12,B=26;
  let xmin=limits&&limits.xmin!=null?limits.xmin:Math.min(...rows.map(p=>p[0])),xmax=limits&&limits.xmax!=null?limits.xmax:Math.max(...rows.map(p=>p[0]));
  let ymin=limits&&limits.ymin!=null?limits.ymin:Math.min(...rows.map(p=>p[1])),ymax=limits&&limits.ymax!=null?limits.ymax:Math.max(...rows.map(p=>p[1]));
  if(xmax<=xmin)xmax=xmin+1;if(ymax<=ymin)ymax=ymin+1;
  const x=v=>L+(v-xmin)*(W-L-R)/(xmax-xmin),y=v=>H-B-(v-ymin)*(H-T-B)/(ymax-ymin);
  let svg='<svg viewBox="0 0 '+W+' '+H+'" role="img" aria-label="chart">';
  for(let i=0;i<4;i++){const value=ymin+(ymax-ymin)*i/3,yy=y(value);svg+='<line class="gridline" x1="'+L+'" y1="'+yy+'" x2="'+(W-R)+'" y2="'+yy+'"/><text class="plotlabel" x="2" y="'+(yy+3)+'">'+(Math.abs(value)>=1000?Math.round(value).toLocaleString():value.toFixed(value<1?3:1))+'</text>';}
  for(const m of markers||[]){const yy=y(m.value);if(yy>=T&&yy<=H-B)svg+='<line x1="'+L+'" y1="'+yy+'" x2="'+(W-R)+'" y2="'+yy+'" stroke="'+m.color+'" stroke-dasharray="4 4" stroke-width="1"/><text class="plotlabel" x="'+(W-R-60)+'" y="'+(yy-4)+'">'+esc(m.label)+'</text>';}
  for(const s of series){const pts=(s.points||[]).filter(p=>Number.isFinite(p[0])&&Number.isFinite(p[1]));if(!pts.length)continue;svg+='<polyline class="plotline" stroke="'+s.color+'" points="'+pts.map(p=>x(p[0])+','+y(p[1])).join(' ')+'"/>';for(const p of pts)svg+='<circle cx="'+x(p[0])+'" cy="'+y(p[1])+'" r="2.6" fill="'+s.color+'"/>';}
  svg+='<text class="plotlabel" x="'+L+'" y="'+(H-5)+'">'+fmt(xmin)+'</text><text class="plotlabel" text-anchor="end" x="'+(W-R)+'" y="'+(H-5)+'">'+fmt(xmax)+'</text></svg>';
  $(id).innerHTML=svg;
}
function seriesOf(hist,field,scale){return Object.keys(hist).sort().map(run=>({name:short(run),color:COLORS[short(run)]||'#ccc',points:(hist[run]||[]).filter(p=>p[field]!=null).map(p=>[p.step,p[field]*(scale||1)])}));}
function render(s){
  const runs=s.runs||{},hist=(s.history&&s.history.metrics)||{},ref=s.reference||{},main=runs[s.main_run]||null;
  const phase=s.phase||'unknown';$('badge').textContent=phase;$('badge').className='badge'+(['blocked','supervisor-error'].includes(phase)?' bad':Date.now()-Date.parse(s.updated_at)>25*60000?' warn':'');
  $('fresh').textContent='Updated '+age(s.updated_at);$('phase-name').textContent=phase;
  const names=Object.keys(runs),arms=names.filter(r=>r!==s.main_run);
  $('summary').textContent=main?'The main run is continuing from the verified 76.20% checkpoint with a conservative recipe.':arms.length?'Sprint arms are comparing one-change-at-a-time variants of the conservative recipe on held-out validation puzzles. The main run starts from the best arm.':'The supervisor is preparing the first sessions.';
  const step=main?n(main.step):0;$('main-step').textContent=fmt(step);$('main-caption').textContent=main?'Latest verified checkpoint on Hugging Face':'Awaiting the main run';
  $('main-status').textContent=main?(main.status||'running').split('\n')[0].slice(0,38):'idle';
  const progress=Math.min(100,100*step/300000);$('main-bar').style.width=progress+'%';
  let bestOfficial={pct:ref.base_puzzles?ref.base_puzzles/100:76.2,note:'verified starting checkpoint · gate '+n(ref.gate_pct,86.13).toFixed(2)+'%'};
  for(const [run,r] of Object.entries(runs))for(const e of r.evaluations||[]){const pct=100*n(e.puzzles&&e.puzzles.solved)/10000;if(pct>bestOfficial.pct)bestOfficial={pct,note:short(run)+' · step '+fmt(e.step)+' · gate '+n(ref.gate_pct,86.13).toFixed(2)+'%'};}
  $('best-official').textContent=bestOfficial.pct.toFixed(2)+'%';$('best-official-note').textContent=bestOfficial.note;$('official-best').textContent=bestOfficial.pct.toFixed(2)+'%';
  const base=(main&&main.baseline)||(arms.map(r=>runs[r].baseline).find(Boolean))||null;
  if(base&&base.val_pct!=null)$('val-baseline').textContent='start '+n(base.val_pct).toFixed(2)+'% on '+fmt(base.val_total)+' puzzles';
  const latestMetric=r=>{const h=hist[r]||[];for(let i=h.length-1;i>=0;i--)if(h[i].val_pct!=null)return h[i];return null;};
  const focus=main?s.main_run:arms[0];const lm=focus?latestMetric(focus):null;
  $('val-now').textContent=lm?n(lm.val_pct).toFixed(2)+'%':'—';$('val-note').textContent=lm?short(focus)+' · step '+fmt(lm.step)+' · EMA weights':'held-out, rating-matched · EMA weights';
  const hitNow=(()=>{const h=hist[focus]||[];for(let i=h.length-1;i>=0;i--)if(h[i].tac_hit!=null)return h[i];return null;})();
  $('hit-now').textContent=hitNow?(100*n(hitNow.tac_hit)).toFixed(1)+'%':'—';
  const totalQuota=(s.accounts||[]).reduce((a,x)=>a+(x.gpu_hours==null?0:n(x.gpu_hours)),0);$('quota').textContent=totalQuota.toFixed(1)+' h';
  const vs=seriesOf(hist,'val_pct'),hs=seriesOf(hist,'tac_hit',100),ls=seriesOf(hist,'dev_dist');
  const allVal=vs.flatMap(x=>x.points.map(p=>p[1]));const vb=base&&base.val_pct!=null?n(base.val_pct):null;
  plot('chart-val',vs,{ymin:Math.min(...allVal,vb==null?100:vb)-1.5,ymax:Math.max(...allVal,vb==null?0:vb)+1.5},vb==null?[]:[{value:vb,label:'start',color:'#edc47e'}]);legend('legend-val',vs);
  plot('chart-hit',hs,{},[]);legend('legend-hit',hs);
  const bv=Math.max(...allVal,-1);$('val-best').textContent=bv>=0?bv.toFixed(2)+'%':'—';const bh=Math.max(...hs.flatMap(x=>x.points.map(p=>p[1])),-1);$('hit-best').textContent=bh>=0?bh.toFixed(1)+'%':'—';
  plot('chart-loss',ls,{},[]);legend('legend-loss',ls);const lastLoss=ls.map(x=>x.points.at(-1)).filter(Boolean).at(-1);$('loss-now').textContent=lastLoss?lastLoss[1].toFixed(4):'—';
  const op=names.sort().map(run=>({name:short(run),color:COLORS[short(run)]||'#ccc',points:(runs[run].evaluations||[]).map(e=>[e.step,100*n(e.puzzles&&e.puzzles.solved)/10000])}));
  plot('chart-official',op,{ymin:Math.min(70,...op.flatMap(x=>x.points.map(p=>p[1])))-1,ymax:Math.max(88,...op.flatMap(x=>x.points.map(p=>p[1])))+1},[{value:76.2,label:'start',color:'#9ab0a7'},{value:n(ref.gate_pct,86.13),label:'9M',color:'#edc47e'}]);legend('legend-official',op);
  const row=r=>{const x=runs[r],m=latestMetric(r),h=(hist[r]||[]).filter(p=>p.tac_hit!=null).at(-1),e=(x.evaluations||[]).at(-1),delta=m&&base&&base.val_pct!=null?n(m.val_pct)-n(base.val_pct):null,rc=x.recipe||{};
    const recipe=(rc.tactical_groups?rc.tactical_groups+' roots':'broad only')+(rc.lambda_list?' + list':'')+' · lr '+(rc.lr==null?'—':Number(rc.lr).toExponential(0));
    return '<tr><td>'+esc(short(r))+'</td><td>'+esc(recipe)+'</td><td>'+fmt(x.step)+'</td><td>'+(m?n(m.val_pct).toFixed(2)+'%':'—')+'</td><td>'+(delta==null?'—':(delta>=0?'+':'')+delta.toFixed(2)+' pp')+'</td><td>'+(h?(100*n(h.tac_hit)).toFixed(1)+'%':'—')+'</td><td>'+(e?(100*n(e.puzzles.solved)/10000).toFixed(2)+'%':'—')+'</td><td>'+esc((x.status||'running').split('\n')[0].slice(0,34))+'</td></tr>';};
  $('runs').innerHTML=names.length?names.sort().map(row).join(''):'<tr><td colspan="8">Awaiting the first checkpoint.</td></tr>';
  $('accounts').innerHTML=(s.accounts||[]).map(x=>'<tr><td>'+esc(x.account)+'</td><td>'+(x.gpu_hours==null?'unknown':n(x.gpu_hours).toFixed(2)+' h')+'</td><td>'+esc(x.learner)+'</td></tr>').join('')||'<tr><td colspan="3">Awaiting Kaggle status.</td></tr>';
  $('supervisor').textContent=(s.supervisor&&s.supervisor.reason)?String(s.supervisor.reason).slice(0,70):'supervisor ok';
  $('events').innerHTML=(s.events||[]).length?(s.events||[]).slice(-16).reverse().map(x=>'<div class="event '+esc(x.tone)+'"><b>'+esc(x.title)+'</b><time>'+esc(age(x.at))+'</time><p>'+esc(x.detail)+'</p></div>').join(''):'<p class="small" style="margin-top:14px">No activity recorded yet.</p>';
  const L=s.links||{};$('links').innerHTML=Object.entries(L).map(([k,v])=>'<a href="'+esc(v)+'">'+esc(k)+' ↗</a>').join('');
  const warnings=s.provider_warnings||[],attention=['blocked','supervisor-error'].includes(phase),alert=$('alert');alert.hidden=!(attention||warnings.length);
  if(!alert.hidden){alert.textContent=(s.supervisor&&s.supervisor.reason)||warnings.join(' · ')||'Supervisor attention needed';alert.className='warning'+(attention?' bad':'');}
}
async function refresh(){try{const res=await fetch('/api/snapshot',{cache:'no-store'});if(!res.ok)throw Error('HTTP '+res.status);const s=await res.json();if(!s.updated_at){$('fresh').textContent='Awaiting first snapshot';return;}render(s);}catch(e){$('badge').textContent='Connection error';$('badge').className='badge bad';$('fresh').textContent=String(e.message||e);}}
refresh();setInterval(refresh,60000);
</script>
</body>
</html>`;
