export const HTML = String.raw`<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#0b1112">
<title>CC-GAVN · live research monitor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{color-scheme:dark;--bg:#0b1112;--panel:#111a1b;--panel2:#152021;--line:#293738;--line2:#1f2b2c;--text:#e9f1ed;--muted:#9aadaa;--dim:#718986;--accent:#b8ef83;--teal:#79d9c8;--amber:#f0c47a;--red:#f0a69a}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 "DM Sans",sans-serif}
button{font:inherit}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
a:focus-visible,button:focus-visible{outline:2px solid var(--accent);outline-offset:3px}
.mono{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}
.wrap{max-width:1460px;padding:0 32px;margin:0 auto}
.top{height:68px;border-bottom:1px solid var(--line);background:#0d1415;position:sticky;top:0;z-index:20}
.topinner{height:100%;display:flex;align-items:center;gap:36px}
.brand{display:flex;align-items:center;gap:12px;color:var(--text);font-weight:700;letter-spacing:-.03em;white-space:nowrap}
.mark{display:grid;grid-template-columns:repeat(2,7px);grid-template-rows:repeat(2,7px);gap:2px;transform:rotate(-12deg)}
.mark i{background:var(--accent);border-radius:2px}.mark i:nth-child(2),.mark i:nth-child(3){opacity:.4}
.brand span:last-child{font-size:17px}
.topnav{display:flex;gap:7px;height:100%;align-items:stretch}
.topnav button{border:0;background:none;color:var(--muted);padding:0 16px;cursor:pointer;position:relative;font-size:13px;font-weight:700}
.topnav button[aria-selected=true]{color:var(--text)}
.topnav button[aria-selected=true]:after{content:"";position:absolute;left:16px;right:16px;bottom:0;height:2px;background:var(--accent)}
.topright{display:flex;align-items:center;gap:18px;margin-left:auto;white-space:nowrap;font-size:12px;color:var(--muted)}
.pill{display:inline-flex;align-items:center;gap:8px;border:1px solid var(--line);border-radius:999px;padding:5px 10px;background:var(--panel);font:500 11px "IBM Plex Mono",monospace;text-transform:uppercase;letter-spacing:.06em}
.pill .dot{height:7px;width:7px;border-radius:50%;background:var(--accent);box-shadow:0 0 10px #b8ef8377}
.pill.stale .dot{background:var(--amber);box-shadow:none}
.pill.bad .dot{background:var(--red);box-shadow:none}
main{padding:36px 0 96px}
.tabpage[hidden]{display:none}
.tabpage{scroll-margin-top:86px}
.eyebrow{font:500 11px "IBM Plex Mono",monospace;text-transform:uppercase;letter-spacing:.12em;color:var(--teal)}
h1,h2,h3,p{margin-top:0}
h1{font-size:clamp(34px,4.4vw,64px);letter-spacing:-.055em;line-height:1.02;margin:16px 0 14px;font-weight:700}
h2{font-size:23px;letter-spacing:-.035em;line-height:1.2;margin:0}
h3{font-size:16px;letter-spacing:-.02em;margin:0}
.hero{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(300px,.7fr);gap:24px;align-items:stretch;margin-bottom:16px}
.heroleft{padding:30px 32px;border:1px solid var(--line);background:radial-gradient(ellipse at 83% 0%,#244034 0%,transparent 44%),var(--panel);border-radius:14px;min-height:270px;display:flex;flex-direction:column}
.hero p{color:var(--muted);max-width:60ch;margin:0}
.herofoot{display:flex;flex-wrap:wrap;gap:8px;margin-top:auto;padding-top:28px}
.tag{border:1px solid #3b5146;border-radius:5px;padding:5px 8px;color:#c5d7c4;font:500 11px "IBM Plex Mono",monospace}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px}
.heroaside{padding:26px 27px;display:flex;flex-direction:column}
.cardlabel{color:var(--muted);font:500 11px "IBM Plex Mono",monospace;text-transform:uppercase;letter-spacing:.08em}
.bigvalue{font:500 clamp(42px,5vw,68px)/1 "IBM Plex Mono",monospace;letter-spacing:-.06em;margin:20px 0 5px;color:var(--accent)}
.bigvalue small{font-size:.38em;letter-spacing:0;color:var(--muted)}
.muted{color:var(--muted)}
.tiny{font-size:12px}
.track{height:7px;background:#273333;border-radius:99px;overflow:hidden}
.track i{display:block;height:100%;width:0;background:var(--accent);border-radius:99px;transition:width .4s}
.heroaside .track{margin:16px 0}
.herometa{display:flex;justify-content:space-between;gap:14px;font:500 12px "IBM Plex Mono",monospace;color:var(--muted)}
.facts{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:42px}
.fact{padding:16px 20px;min-height:106px}
.fact strong{font:500 24px "IBM Plex Mono",monospace;letter-spacing:-.04em;display:block;margin-top:12px}
.fact .unit{font-size:13px;color:var(--muted);margin-left:5px}
.section{margin-top:38px}
.sectionhead{display:flex;align-items:end;justify-content:space-between;gap:15px;margin-bottom:14px}
.sectionhead p{margin:5px 0 0;color:var(--muted);font-size:13px}
.sectionhead .aside{color:var(--dim);font:500 11px "IBM Plex Mono",monospace;white-space:nowrap}
.maincols{display:grid;grid-template-columns:minmax(0,1.75fr) minmax(290px,.85fr);gap:14px}
.chartcard{padding:20px 21px 15px;min-height:340px}
.charthead{display:flex;align-items:start;justify-content:space-between;gap:12px}
.charthead .reading{font:500 18px "IBM Plex Mono",monospace;color:var(--accent)}
.chartsub{font-size:12px;color:var(--muted);margin-top:5px}
.legend{display:flex;flex-wrap:wrap;gap:13px;margin:12px 0 0;font:11px "IBM Plex Mono",monospace;color:var(--muted)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.legend i{display:inline-block;width:8px;height:8px;border-radius:2px;background:var(--accent)}
.chartbox{height:225px;margin-top:10px;position:relative}
.chartbox svg{width:100%;height:100%;display:block}
.chartbox .empty{height:100%;display:grid;place-content:center;text-align:center;color:var(--muted);font-size:13px}
.chartbox .empty strong{display:block;color:var(--text);font-size:15px;margin-bottom:5px}
.chartbox text{font:10px "IBM Plex Mono",monospace;fill:var(--dim)}
.chartbox .grid{stroke:#2c3a3a;stroke-width:1}
.chartbox .line{fill:none;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
.chartbox .point{stroke:var(--panel);stroke-width:2}
.activity{min-height:340px;display:flex;flex-direction:column}
.activityhead{padding:20px 20px 13px;border-bottom:1px solid var(--line2);display:flex;justify-content:space-between;align-items:center}
.activityhead span{color:var(--muted);font:11px "IBM Plex Mono",monospace}
.feed{overflow:auto;max-height:285px;padding:2px 20px 6px;flex:1}
.event{padding:14px 0 14px 16px;border-bottom:1px solid var(--line2);position:relative}
.event:last-child{border-bottom:0}
.event:before{content:"";position:absolute;left:0;top:21px;width:6px;height:6px;border-radius:50%;background:var(--teal)}
.event.good:before{background:var(--accent)}.event.bad:before{background:var(--red)}
.eventtop{display:flex;justify-content:space-between;gap:8px}
.eventtop strong{font-size:12px}
.eventtop time{font:10px "IBM Plex Mono",monospace;color:var(--dim);white-space:nowrap}
.event p{font-size:11px;color:var(--muted);margin:5px 0 0;line-height:1.45}
.emptyfeed{padding:24px 0;color:var(--muted);font-size:12px}
.stages{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.stage{padding:19px 19px 17px}
.stage .topline{display:flex;align-items:center;justify-content:space-between;color:var(--muted);font:11px "IBM Plex Mono",monospace}
.stage .state{color:var(--dim);text-transform:uppercase}
.stage .state.running,.stage .state.done{color:var(--accent)}
.stage .state.error{color:var(--red)}
.stage h3{margin:18px 0 5px}
.stage p{color:var(--muted);font-size:12px;min-height:34px;margin:0 0 15px}
.stage .track{height:5px}
.stage .foot{font:11px "IBM Plex Mono",monospace;color:var(--dim);margin-top:10px}
.tablecard{overflow-x:auto}
table{border-collapse:collapse;width:100%;min-width:680px}
th,td{text-align:left;padding:12px 18px;border-bottom:1px solid var(--line2)}
tr:last-child td{border-bottom:0}
th{font:500 10px "IBM Plex Mono",monospace;text-transform:uppercase;letter-spacing:.07em;color:var(--dim);background:#152021}
td{font-size:12px}
td:first-child{font:500 12px "IBM Plex Mono",monospace}
tr:hover td{background:#172223}
.cellbar{width:110px;height:5px;background:#2d3a39;border-radius:8px;display:inline-block;vertical-align:middle;margin-right:10px}
.cellbar i{height:100%;display:block;background:var(--teal);border-radius:8px}
.statusword{font:500 10px "IBM Plex Mono",monospace;text-transform:uppercase;letter-spacing:.03em}
.statusword.running,.statusword.done{color:var(--accent)}
.statusword.queued{color:var(--amber)}.statusword.error{color:var(--red)}
.scoregrid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.score{padding:17px 19px}
.score .who{font:500 10px "IBM Plex Mono",monospace;color:var(--muted);text-transform:uppercase;letter-spacing:.07em}
.score .scoreval{font:500 27px "IBM Plex Mono",monospace;margin:15px 0 6px;letter-spacing:-.04em}
.score .scoreval.goal{color:var(--accent)}
.score p{margin:0;color:var(--muted);font-size:11px}
.notice{border-left:2px solid var(--amber);background:#29251d;color:#e9d4ad;padding:12px 16px;border-radius:0 7px 7px 0;font-size:12px;margin-top:16px}
.notice[hidden]{display:none}
.toolbar{display:flex;align-items:center;gap:12px;justify-content:space-between;flex-wrap:wrap;margin:20px 0 14px}
.segmented{display:inline-flex;padding:3px;border:1px solid var(--line);border-radius:7px;background:var(--panel)}
.segmented button{color:var(--muted);border:0;background:none;padding:6px 13px;font:500 11px "IBM Plex Mono",monospace;border-radius:4px;cursor:pointer}
.segmented button[aria-pressed=true]{background:#263636;color:var(--text)}
.metricgrid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:13px}
.metricgrid .chartcard{min-height:325px}
.metricgrid .chartbox{height:215px}
.metricgrid .chartcard:first-child,.metricgrid .chartcard:nth-child(2){grid-column:auto}
.methodintro{max-width:74ch;color:var(--muted);margin:12px 0 28px}
.methodcols{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.methodcard{padding:22px}
.methodcard h3{margin-bottom:12px}
.definition{display:grid;grid-template-columns:160px 1fr;gap:10px;padding:10px 0;border-top:1px solid var(--line2);font-size:12px}
.definition span{color:var(--muted)}
.definition strong{font:500 12px "IBM Plex Mono",monospace;overflow-wrap:anywhere}
.linklist{display:flex;gap:9px;flex-wrap:wrap;margin-top:16px}
.linklist a{padding:8px 11px;border:1px solid #42574a;border-radius:6px;font:500 11px "IBM Plex Mono",monospace}
.bottom{border-top:1px solid var(--line);padding:22px 0 32px;color:var(--dim);font-size:11px}
.bottom .wrap{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap}
@media(max-width:1050px){.maincols{grid-template-columns:1fr}.activity{min-height:0}.feed{max-height:250px}.facts{grid-template-columns:repeat(2,1fr)}}
@media(max-width:740px){.wrap{padding:0 18px}.top{height:auto}.topinner{height:auto;min-height:62px;gap:8px;flex-wrap:wrap}.brand span:last-child{font-size:15px}.topright{gap:7px;margin-left:auto}.topright .fresh{display:none}.topnav{order:3;width:100%;height:39px;overflow-x:auto}.topnav button{padding:0 12px;flex:none}.topnav button:first-child{padding-left:0}.topnav button[aria-selected=true]:after{left:0;right:12px}.tabpage{scroll-margin-top:115px}main{padding-top:22px}.hero{grid-template-columns:1fr}.heroleft{padding:25px;min-height:235px}.heroaside{padding:22px}.facts{margin-bottom:30px}.stages{grid-template-columns:1fr}.stage p{min-height:0}.scoregrid{grid-template-columns:1fr}.metricgrid{grid-template-columns:1fr}.methodcols{grid-template-columns:1fr}.sectionhead{align-items:start}.sectionhead .aside{white-space:normal;text-align:right}.definition{grid-template-columns:120px 1fr}.chartcard{padding:18px 13px 10px}.chartbox{height:215px}}
@media(max-width:430px){.facts{gap:8px}.fact{padding:13px}.fact strong{font-size:20px}.heroleft{padding:21px}.topright .pill{font-size:9px;padding:5px 7px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}.track i{transition:none}}
</style>
</head>
<body>
<header class="top"><div class="wrap topinner">
  <a class="brand" href="#overview"><span class="mark" aria-hidden="true"><i></i><i></i><i></i><i></i></span><span>CC-GAVN <span class="muted">/ research monitor</span></span></a>
  <nav class="topnav" role="tablist" aria-label="Monitor views">
    <button id="tab-overview" role="tab" aria-controls="overview" data-tab="overview" aria-selected="true" tabindex="0">Overview</button>
    <button id="tab-metrics" role="tab" aria-controls="metrics" data-tab="metrics" aria-selected="false" tabindex="-1">Metrics</button>
    <button id="tab-method" role="tab" aria-controls="method" data-tab="method" aria-selected="false" tabindex="-1">Run contract</button>
  </nav>
  <div class="topright"><span class="fresh mono" id="top-fresh">Connecting</span><span class="pill" id="live-pill"><span class="dot"></span><span id="live-text">Connecting</span></span></div>
</div></header>
<main class="wrap">
<section class="tabpage" id="overview" role="tabpanel" aria-labelledby="tab-overview">
  <div class="hero">
    <div class="heroleft">
      <div class="eyebrow">EXPERIMENT 01 / ROOT-POLICY REINFORCEMENT LEARNING</div>
      <h1>Live run monitor<span style="color:var(--accent)">.</span></h1>
      <p>Five Kaggle accounts generate exact verifier labels. A compact chess model then trains with group-relative rewards before one frozen evaluation.</p>
      <div class="herofoot"><span class="tag">ONE PASS / LEGAL MOVE</span><span class="tag">T4 GPU POOL</span><span class="tag">HF RESUMABLE</span><span class="tag" id="phase-tag">PHASE · CONNECTING</span></div>
    </div>
    <div class="heroaside card">
      <span class="cardlabel">Verifier dataset collected</span>
      <div class="bigvalue" id="hero-percent">—<small>%</small></div>
      <div class="herometa"><span id="hero-labels">— / 10,000 roots</span><span id="hero-depth">DEPTH 14</span></div>
      <div class="track"><i id="hero-progress"></i></div>
      <div class="herometa"><span id="hero-workers">— active verifiers</span><span id="hero-phase">Labels phase</span></div>
    </div>
  </div>
  <div class="facts">
    <div class="card fact"><span class="cardlabel">Exact labels</span><strong id="fact-labels">—</strong><span class="tiny muted">of 10,000 positions</span></div>
    <div class="card fact"><span class="cardlabel">RL checkpoint</span><strong id="fact-step">—</strong><span class="tiny muted">of 20,000 updates</span></div>
    <div class="card fact"><span class="cardlabel">Active accounts</span><strong id="fact-accounts">—</strong><span class="tiny muted">of five in pool</span></div>
    <div class="card fact"><span class="cardlabel">Frozen puzzle score</span><strong id="fact-score">Pending</strong><span class="tiny muted">10,000 puzzles · target &gt; 86.13%</span></div>
  </div>
  <div class="sectionhead"><div><h2>Run activity</h2><p>Observed pipeline states and persisted checkpoint uploads</p></div><span class="aside" id="history-count">History starts with next publish</span></div>
  <div class="maincols">
    <div class="card chartcard">
      <div class="charthead"><div><h3>Verifier labels over time</h3><div class="chartsub">Persisted exact roots across five account shards</div></div><strong class="reading" id="label-reading">—</strong></div>
      <div class="legend" id="label-legend"></div>
      <div class="chartbox" id="label-chart"></div>
    </div>
    <div class="card activity">
      <div class="activityhead"><h3>Activity feed</h3><span>NEWEST FIRST</span></div>
      <div class="feed" id="activity-feed"><div class="emptyfeed">Waiting for the next supervisor publish.</div></div>
    </div>
  </div>
  <section class="section"><div class="sectionhead"><div><h2>Pipeline</h2><p>Completion is based on uploaded state, not queued jobs</p></div><span class="aside" id="snapshot-time">—</span></div>
    <div class="stages" id="stages"></div>
  </section>
  <section class="section"><div class="sectionhead"><div><h2>Five-account pool</h2><p>Verifier state, persisted work, remaining T4 quota, and RL worker</p></div><span class="aside">SOURCE · KAGGLE + HF</span></div>
    <div class="card tablecard"><table><thead><tr><th>Account</th><th>Labels</th><th>Verifier kernel</th><th>Upload status</th><th>T4 remaining</th><th>RL kernel</th></tr></thead><tbody id="account-rows"><tr><td colspan="6">Waiting for first snapshot</td></tr></tbody></table></div>
  </section>
  <section class="section"><div class="sectionhead"><div><h2>Frozen benchmark</h2><p>Only the complete checkpoint-scoped evaluation can populate RL scores</p></div><span class="aside">4,000 MATE · 10,000 PUZZLES</span></div>
    <div class="scoregrid">
      <div class="card score"><div class="who">Current baseline / 4.76M</div><div class="scoreval">76.20%</div><p>Puzzles · MATE 92.75%</p></div>
      <div class="card score"><div class="who">9M teacher / target to beat</div><div class="scoreval">86.13%</div><p>Puzzles · MATE 98.72%</p></div>
      <div class="card score"><div class="who">Root-policy RL / final</div><div class="scoreval goal" id="score-rl">Pending</div><p id="score-rl-detail">Waiting for frozen evaluation</p></div>
    </div>
  </section>
  <div class="notice" id="problem" hidden></div>
</section>
<section class="tabpage" id="metrics" role="tabpanel" aria-labelledby="tab-metrics" hidden>
  <div class="eyebrow">TRAINING TELEMETRY / COMPLETE HF CHECKPOINTS</div>
  <h1>Metrics<span style="color:var(--accent)">.</span></h1>
  <p class="methodintro">Each point comes from a persisted checkpoint. Training losses and the fixed dev probe are shown separately from the final frozen benchmark.</p>
  <div class="facts">
    <div class="card fact"><span class="cardlabel">Latest step</span><strong id="metric-step">—</strong><span class="tiny muted">of 20,000</span></div>
    <div class="card fact"><span class="cardlabel">Dev top-1</span><strong id="metric-top1">—</strong><span class="tiny muted">checkpoint probe only</span></div>
    <div class="card fact"><span class="cardlabel">Dev mean regret</span><strong id="metric-regret">—</strong><span class="tiny muted">lower is better</span></div>
    <div class="card fact"><span class="cardlabel">Recorded points</span><strong id="metric-points">—</strong><span class="tiny muted">complete checkpoints</span></div>
  </div>
  <div class="toolbar"><div><h2>Checkpoint traces</h2><div class="chartsub">No interpolation or projected result points</div></div><div class="segmented" role="group" aria-label="Chart horizontal axis"><button data-axis="step" aria-pressed="true">Step</button><button data-axis="time" aria-pressed="false">Observed time</button></div></div>
  <div class="metricgrid" id="metric-grid"></div>
  <div class="notice">Dev top-1 is a training probe. It is not puzzle accuracy. Frozen puzzle and MATE scores appear only after the full evaluation completes.</div>
</section>
<section class="tabpage" id="method" role="tabpanel" aria-labelledby="tab-method" hidden>
  <div class="eyebrow">PREDECLARED EXPERIMENT CONTRACT</div>
  <h1>Run contract<span style="color:var(--accent)">.</span></h1>
  <p class="methodintro">This page documents the live run and the evidence behind each dashboard value. The five accounts share the label workload; training resumes from complete Hugging Face checkpoints.</p>
  <div class="methodcols">
    <div class="card methodcard"><h3>Verifier and learner</h3><div id="method-verifier"></div></div>
    <div class="card methodcard"><h3>Training configuration</h3><div id="method-train"></div></div>
    <div class="card methodcard"><h3>Frozen success gate</h3>
      <div class="definition"><span>Puzzles</span><strong>&gt; 86.13% / 10,000</strong></div>
      <div class="definition"><span>MATE</span><strong>4,000 positions</strong></div>
      <div class="definition"><span>Teacher</span><strong>9M parameters</strong></div>
      <div class="definition"><span>Current baseline</span><strong>76.20% puzzle / 92.75% MATE</strong></div>
      <div class="definition"><span>Claim rule</span><strong>Complete checkpoint-scoped evaluation only</strong></div>
    </div>
    <div class="card methodcard"><h3>Sources and artifacts</h3><p class="tiny muted">Snapshots are collected by GitHub Actions from Kaggle and Hugging Face, then published to this Worker. The page refreshes every 30 seconds; the supervisor is scheduled every five minutes. Checkpoint uploads arrive on their training cadence.</p><div class="linklist" id="artifact-links"></div></div>
  </div>
</section>
</main>
<footer class="bottom"><div class="wrap"><span>CC-GAVN · ROOT-POLICY RL MONITOR</span><span id="footer-fresh">Connecting to pipeline</span></div></footer>
<script>
const byId = id => document.getElementById(id);
const esc = value => String(value ?? "").replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch]));
const number = value => Number(value || 0).toLocaleString();
const percent = (n,d) => d ? Math.max(0,Math.min(100,100*n/d)) : 0;
const time = value => { const d=new Date(value); return isNaN(d) ? "—" : d.toLocaleString(); };
const shortTime = value => { const d=new Date(value); return isNaN(d) ? "—" : d.toLocaleTimeString([], {hour:"2-digit",minute:"2-digit"}); };
const age = value => { const n=Date.now()-new Date(value).getTime(); if(!Number.isFinite(n))return "unknown"; if(n<60000)return "just now"; if(n<3600000)return Math.floor(n/60000)+"m ago"; if(n<86400000)return Math.floor(n/3600000)+"h ago"; return Math.floor(n/86400000)+"d ago"; };
let current = {};
let axis = "step";
const metricSpec = [
  {key:"dev_top1", title:"Dev top-1", sub:"Fixed development probe · higher is better", format:v=>(100*v).toFixed(1)+"%", color:"#b8ef83", domain:[0,1]},
  {key:"dev_regret", title:"Dev mean regret", sub:"Fixed development probe · lower is better", format:v=>v.toFixed(3), color:"#79d9c8"},
  {key:"loss", title:"Total loss", sub:"Checkpoint training loss", format:v=>v.toFixed(4), color:"#b8ef83"},
  {key:"pg", title:"Policy objective", sub:"Group-relative policy gradient component", format:v=>v.toFixed(4), color:"#79d9c8"},
  {key:"kl", title:"KL term", sub:"Reference policy regularization", format:v=>v.toFixed(4), color:"#f0c47a"},
  {key:"value", title:"Value term", sub:"Value-head training component", format:v=>v.toFixed(4), color:"#a7b8f2"},
  {key:"grad_norm", title:"Gradient norm", sub:"After backward pass, before optimizer step", format:v=>v.toFixed(3), color:"#e5a6c9"}
];
function setTab(tab) {
  if(!["overview","metrics","method"].includes(tab))tab="overview";
  document.querySelectorAll(".tabpage").forEach(node=>node.hidden=node.id!==tab);
  document.querySelectorAll("[data-tab]").forEach(node=>{
    const selected=node.dataset.tab===tab;
    node.setAttribute("aria-selected",String(selected));
    node.tabIndex=selected?0:-1;
  });
  if(location.hash!=="#"+tab)history.replaceState(null,"","#"+tab);
  if(tab==="metrics")renderMetrics(current);
}
document.querySelectorAll("[data-tab]").forEach(node=>node.addEventListener("click",()=>{setTab(node.dataset.tab);window.scrollTo(0,0); }));
byId("tab-overview").parentElement.addEventListener("keydown",event=>{
  const tabs=[...document.querySelectorAll("[data-tab]")];
  const currentIndex=tabs.indexOf(document.activeElement);
  if(currentIndex<0)return;
  let next=currentIndex;
  if(event.key==="ArrowRight")next=(currentIndex+1)%tabs.length;
  else if(event.key==="ArrowLeft")next=(currentIndex+tabs.length-1)%tabs.length;
  else if(event.key==="Home")next=0;
  else if(event.key==="End")next=tabs.length-1;
  else return;
  event.preventDefault();setTab(tabs[next].dataset.tab);tabs[next].focus();window.scrollTo(0,0);
});
window.addEventListener("hashchange",()=>setTab(location.hash.slice(1)));
document.querySelectorAll("[data-axis]").forEach(node=>node.addEventListener("click",()=>{
  axis=node.dataset.axis;
  document.querySelectorAll("[data-axis]").forEach(button=>button.setAttribute("aria-pressed",String(button.dataset.axis===axis)));
  renderMetrics(current);
}));
function define(label,value){return '<div class="definition"><span>'+esc(label)+'</span><strong>'+esc(value==null||value===""?"Pending":value)+'</strong></div>';}
function statusClass(value){const x=String(value||"").toUpperCase();if(/FAILED|ERROR|INCOMPLETE/.test(x))return "error";if(/DONE|COMPLETE|SUCCEEDED/.test(x))return "done";if(/RUNNING/.test(x))return "running";if(/QUEUED|PENDING/.test(x))return "queued";return "waiting";}
function statusSpan(value){return '<span class="statusword '+statusClass(value)+'">'+esc(value||"—")+'</span>';}
function stage(index,title,detail,value,target,state,foot){
 return '<div class="card stage"><div class="topline"><span>0'+index+' / 03</span><span class="state '+esc(state)+'">'+esc(state)+'</span></div><h3>'+esc(title)+'</h3><p>'+esc(detail)+'</p><div class="track"><i style="width:'+percent(value,target)+'%"></i></div><div class="foot">'+esc(foot)+'</div></div>';
}
function svgChart(container, points, series, opts={}) {
  const rows=(points||[]).filter(p=>Number.isFinite(Number(p.x))&&series.some(s=>Number.isFinite(Number(p[s.key]))&&p[s.key]!==null));
  if(!rows.length){container.innerHTML='<div class="empty"><div><strong>'+esc(opts.emptyTitle||"Awaiting first checkpoint")+'</strong>'+esc(opts.emptyText||"Chart begins when a complete data point is published.")+'</div></div>';return;}
  const W=700,H=220,L=47,R=15,T=14,B=31;
  let xmin=Math.min(...rows.map(p=>Number(p.x))),xmax=Math.max(...rows.map(p=>Number(p.x)));
  if(xmin===xmax){xmin-=1;xmax+=1;}
  const values=rows.flatMap(p=>series.map(s=>p[s.key]).filter(v=>v!==null&&Number.isFinite(Number(v))).map(Number));
  let ymin=opts.ymin!=null?opts.ymin:Math.min(...values),ymax=opts.ymax!=null?opts.ymax:Math.max(...values);
  if(ymin===ymax){const pad=Math.max(1,Math.abs(ymin)*.1);ymin-=pad;ymax+=pad;}
  if(opts.ymin==null&&opts.ymax==null){const pad=(ymax-ymin)*.12;ymin-=pad;ymax+=pad;}
  const sx=x=>L+(Number(x)-xmin)/(xmax-xmin)*(W-L-R);
  const sy=y=>T+(ymax-Number(y))/(ymax-ymin)*(H-T-B);
  const fmtY=opts.formatY|| (v=>Math.abs(v)>=1000?Math.round(v).toLocaleString():Number(v.toFixed(2)).toString());
  let out='<svg viewBox="0 0 '+W+' '+H+'" role="img" aria-label="'+esc(opts.aria||"Metric history")+'" preserveAspectRatio="none">';
  for(let i=0;i<=4;i++){const y=T+(H-T-B)*i/4;out+='<line class="grid" x1="'+L+'" x2="'+(W-R)+'" y1="'+y+'" y2="'+y+'"/><text x="'+(L-7)+'" y="'+(y+3)+'" text-anchor="end">'+esc(fmtY(ymax-(ymax-ymin)*i/4))+'</text>';}
  const fmtX=opts.formatX|| (v=>Math.round(v).toLocaleString());
  for(let i=0;i<=4;i++){const x=L+(W-L-R)*i/4;out+='<text x="'+x+'" y="'+(H-7)+'" text-anchor="'+(i===0?"start":i===4?"end":"middle")+'">'+esc(fmtX(xmin+(xmax-xmin)*i/4))+'</text>';}
  for(const s of series){
    const pts=rows.filter(p=>p[s.key]!==null&&Number.isFinite(Number(p[s.key])));
    if(pts.length>1)out+='<polyline class="line" points="'+pts.map(p=>sx(p.x).toFixed(1)+','+sy(p[s.key]).toFixed(1)).join(" ")+'" stroke="'+s.color+'"/>';
    for(const p of pts){const label=(opts.pointLabel?opts.pointLabel(p):fmtX(p.x))+' · '+s.name+': '+(s.format?s.format(Number(p[s.key])):fmtY(Number(p[s.key])));
      out+='<circle class="point" cx="'+sx(p.x).toFixed(1)+'" cy="'+sy(p[s.key]).toFixed(1)+'" r="4" fill="'+s.color+'"><title>'+esc(label)+'</title></circle>';
    }
  }
  out+='</svg>';container.innerHTML=out;
}
function renderLabelChart(s){
 const observations=s.history?.observations||[];
 byId("history-count").textContent=observations.length+" observed snapshots · "+(s.history?.metrics||[]).length+" checkpoint points";
 byId("label-reading").textContent=number(s.label_count||0)+" / "+number(s.label_target||10000);
 const colors=["#b8ef83","#79d9c8","#f0c47a","#a7b8f2","#e5a6c9"];
 const series=[{key:"total",name:"All accounts",color:"#ffffff",format:v=>number(v)}];
 for(let i=0;i<5;i++)series.push({key:"a"+i,name:"Account "+(i+1),color:colors[i],format:v=>number(v)});
 byId("label-legend").innerHTML=series.map(item=>'<span><i style="background:'+item.color+'"></i>'+esc(item.name)+'</span>').join("");
 const rows=observations.map(p=>({x:new Date(p.at).getTime(),at:p.at,total:Number(p.label_count||0),
  a0:p.labels?.[0]??0,a1:p.labels?.[1]??0,a2:p.labels?.[2]??0,a3:p.labels?.[3]??0,a4:p.labels?.[4]??0}));
 svgChart(byId("label-chart"),rows,series,{ymin:0,ymax:Number(s.label_target||10000),
   formatY:v=>number(Math.round(v)),formatX:v=>shortTime(v),pointLabel:p=>time(p.at),
   emptyTitle:"Timeline collecting observations",emptyText:"The first point appears after the next supervisor publish.",
   aria:"Persisted verifier labels by account over observed time"});
}
function renderMetrics(s){
 const points=s.history?.metrics||[];
 const grid=byId("metric-grid");
 grid.innerHTML=metricSpec.map(item=>'<div class="card chartcard"><div class="charthead"><div><h3>'+esc(item.title)+'</h3><div class="chartsub">'+esc(item.sub)+'</div></div><strong class="reading" id="reading-'+item.key+'">—</strong></div><div class="chartbox" id="chart-'+item.key+'"></div></div>').join("");
 for(const item of metricSpec){
  const rows=points.filter(p=>p[item.key]!==null&&Number.isFinite(Number(p[item.key]))).map(p=>({x:axis==="step"?p.step:new Date(p.observed_at).getTime(),step:p.step,at:p.observed_at,[item.key]:p[item.key]}));
  const last=rows.at(-1);byId("reading-"+item.key).textContent=last?item.format(Number(last[item.key])):"—";
  svgChart(byId("chart-"+item.key),rows,[{key:item.key,name:item.title,color:item.color,format:item.format}],{
    ymin:item.domain?.[0],ymax:item.domain?.[1],formatY:item.format,
    formatX:axis==="step"?v=>number(Math.round(v)):v=>shortTime(v),
    pointLabel:p=>"Step "+number(p.step)+" · first observed "+time(p.at),
    emptyTitle:"Awaiting first checkpoint",emptyText:"The first complete HF checkpoint will start this trace.",
    aria:item.title+" by "+(axis==="step"?"RL step":"first observed time")
  });
 }
 byId("metric-step").textContent=number(s.training?.step||0);
 const latest=points.at(-1);
 byId("metric-top1").textContent=latest?.dev_top1==null?"—":(latest.dev_top1*100).toFixed(1)+"%";
 byId("metric-regret").textContent=latest?.dev_regret==null?"—":Number(latest.dev_regret).toFixed(3);
 byId("metric-points").textContent=number(points.length);
}
function render(s){
 current=s;
 const labels=Array.isArray(s.labels)?s.labels:[];
 const accounts=Array.isArray(s.accounts)?s.accounts:[];
 const training=s.training||{};
 const step=Number(training.step||0),target=Number(training.target||20000);
 const count=Number(s.label_count||0),labelTarget=Number(s.label_target||10000);
 const evalComplete=!!s.evaluation;
 const active=labels.filter(item=>statusClass(item.kernel)==="running").length;
 const rlRunning=accounts.filter(item=>statusClass(item.learner)==="running").length;
 const phase=evalComplete?"Evaluation complete":step>=target?"Frozen evaluation":step>0||rlRunning?"RL training":"Verifier labels";
 const freshness=s.updated_at?age(s.updated_at):"No snapshot";
 const stale=!s.updated_at||Date.now()-new Date(s.updated_at).getTime()>30*60000;
 const pill=byId("live-pill");pill.className="pill"+(stale?" stale":"");
 byId("live-text").textContent=stale?"Update delayed":"Live monitor";
 byId("top-fresh").textContent=s.updated_at?"Updated "+freshness:"Connecting";
 byId("footer-fresh").textContent=s.updated_at?"Last pipeline observation "+time(s.updated_at)+" · page polls every 30 seconds":"Waiting for first pipeline snapshot";
 byId("phase-tag").textContent="PHASE · "+phase.toUpperCase();
 byId("hero-percent").innerHTML=percent(count,labelTarget).toFixed(1)+'<small>%</small>';
 byId("hero-labels").textContent=number(count)+" / "+number(labelTarget)+" roots";
 byId("hero-depth").textContent="DEPTH "+(s.label_depth||14);
 byId("hero-progress").style.width=percent(count,labelTarget)+"%";
 byId("hero-workers").textContent=active+" active verifier"+(active===1?"":"s");
 byId("hero-phase").textContent=phase;
 byId("fact-labels").textContent=number(count);
 byId("fact-step").textContent=number(step);
 byId("fact-accounts").textContent=number(new Set([...labels.filter(x=>statusClass(x.kernel)==="running").map(x=>x.account),...accounts.filter(x=>statusClass(x.learner)==="running").map(x=>x.account)]).size);
 byId("fact-score").textContent=evalComplete?(100*s.evaluation.puzzles.solved/s.evaluation.puzzles.total).toFixed(2)+"%":"Pending";
 byId("snapshot-time").textContent=s.updated_at?"OBSERVED "+time(s.updated_at):"AWAITING SNAPSHOT";
 renderLabelChart(s);
 const events=(s.events||[]).slice().reverse();
 byId("activity-feed").innerHTML=events.length?events.map(item=>'<div class="event '+esc(item.tone)+'"><div class="eventtop"><strong>'+esc(item.title)+'</strong><time title="'+esc(time(item.at))+'">'+esc(age(item.at))+'</time></div><p>'+esc(item.detail)+'</p></div>').join(""):'<div class="emptyfeed">Activity appears after the next supervisor publish.</div>';
 const labelDone=count>=labelTarget;
 byId("stages").innerHTML=
 stage(1,"Exact verifier labels",number(count)+" / "+number(labelTarget)+" depth "+(s.label_depth||14)+" all-legal roots",count,labelTarget,labelDone?"done":active?"running":"waiting",labels.length+" shards · "+active+" running")+
 stage(2,"Root-policy RL",number(step)+" / "+number(target)+" complete checkpoint steps",step,target,step>=target?"done":rlRunning?"running":"waiting",training.checkpoint||"Awaiting first checkpoint")+
 stage(3,"Frozen evaluation",evalComplete?"4,000 MATE and 10,000 puzzles complete":"Starts after final checkpoint",evalComplete?1:0,1,evalComplete?"done":step>=target?"running":"waiting",evalComplete?"Results archived":"No final score yet");
 byId("account-rows").innerHTML=labels.length?labels.map((item,i)=>{const pool=accounts.find(x=>x.account===item.account)||accounts[i]||{};
  return '<tr><td>'+esc(item.account)+'</td><td><span class="cellbar"><i style="width:'+percent(item.count,item.target)+'%"></i></span><span class="mono">'+number(item.count)+' / '+number(item.target)+'</span></td><td>'+statusSpan(item.kernel)+'</td><td>'+esc(item.status||"—")+'</td><td class="mono">'+esc(pool.gpu_hours==null?"—":Number(pool.gpu_hours).toFixed(1)+"h")+'</td><td>'+statusSpan(pool.learner)+'</td></tr>';
 }).join(""):'<tr><td colspan="6">Waiting for first snapshot</td></tr>';
 const score=evalComplete?100*s.evaluation.puzzles.solved/s.evaluation.puzzles.total:null;
 const mate=evalComplete?100*s.evaluation.mate.correct/s.evaluation.mate.total:null;
 byId("score-rl").textContent=evalComplete?score.toFixed(2)+"%":"Pending";
 byId("score-rl-detail").textContent=evalComplete?"MATE "+mate.toFixed(2)+"% · "+(score>86.13?"+"+(score-86.13).toFixed(2):"-"+(86.13-score).toFixed(2))+" pp vs teacher":"Waiting for frozen evaluation";
 const errors=[];
 labels.forEach(item=>{if(statusClass(item.status)==="error")errors.push(item.account+": "+item.status);});
 if(statusClass(training.status)==="error")errors.push("RL: "+String(training.status).split(String.fromCharCode(10)).at(-1));
 if(statusClass(s.evaluation_status)==="error")errors.push("Evaluation: "+String(s.evaluation_status).split(String.fromCharCode(10)).at(-1));
 byId("problem").hidden=!errors.length;byId("problem").textContent=errors.join(" · ");
 const c=training.config||{};
 byId("method-verifier").innerHTML=define("Run",s.run||"ccgavn-rl-root-v1-seed0")+define("Positions",number(labelTarget)+" exact roots")+define("Verifier depth",s.label_depth||14)+define("Account shards",labels.length||5)+define("Learner size",training.parameter_count?number(training.parameter_count)+" parameters":"Awaiting checkpoint")+define("Source checkpoint",c.source_checkpoint);
 byId("method-train").innerHTML=define("Algorithm",c.algorithm||"Awaiting checkpoint")+define("Steps",number(target))+define("Batch roots",c.batch_positions)+define("Group size",c.group_size)+define("PPO epochs",c.ppo_epochs)+define("Learning rate",c.lr)+define("KL coefficient",c.kl_coef)+define("Seed",c.seed)+define("Source commit",c.source_commit);
 const links=s.links||{};
 byId("artifact-links").innerHTML=[["GitHub workflow",links.github],["Verifier labels",links.labels],["RL checkpoints",links.checkpoints],["Evaluation",links.evaluation]].filter(x=>x[1]).map(x=>'<a target="_blank" rel="noopener noreferrer" href="'+esc(x[1])+'">'+esc(x[0])+' ↗</a>').join("");
 if(!byId("metrics").hidden)renderMetrics(s);
}
async function load(){
 try{const response=await fetch("/api/snapshot",{cache:"no-store"});if(!response.ok)throw Error("HTTP "+response.status);render(await response.json());}
 catch(error){byId("live-pill").className="pill bad";byId("live-text").textContent="Connection error";byId("top-fresh").textContent="Snapshot unavailable";byId("footer-fresh").textContent="Monitor request failed: "+error.message;}
}
if("scrollRestoration" in history)history.scrollRestoration="manual";
setTab(location.hash.slice(1));
window.scrollTo(0,0);
window.addEventListener("load",()=>window.scrollTo(0,0));
load();
setInterval(load,30000);
setInterval(()=>{if(current.updated_at){byId("top-fresh").textContent="Updated "+age(current.updated_at);document.querySelectorAll(".event time").forEach((node,i)=>{const events=(current.events||[]).slice().reverse();if(events[i])node.textContent=age(events[i].at);});}},15000);
</script>
</body></html>`;
