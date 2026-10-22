// Renders the Worker HTML, executes the served inline script in a stub DOM,
// and asserts every dashboard section is populated. Catches scope bugs like
// identifiers used in the browser script but declared only in the Worker
// module (2026-10-06: FINAL_STEP / evalNumbers blanked the page).
//
// Usage: node monitor/check_render.mjs
import { readFileSync, writeFileSync, mkdtempSync } from "fs";
import { tmpdir } from "os";
import { join, dirname } from "path";
import { fileURLToPath, pathToFileURL } from "url";
import vm from "vm";

const here = dirname(fileURLToPath(import.meta.url));
let src = readFileSync(join(here, "src", "index.js"), "utf8");
src = src.replace('import { Chess } from "chess.js";', "const Chess = class {};");
src = src.replace("export default {", "const __worker = {");
const tmp = join(mkdtempSync(join(tmpdir(), "monitor-render-")), "render.mjs");
writeFileSync(tmp, src + "\nexport { HTML };\n");
const { HTML } = await import(pathToFileURL(tmp).href);

const script = HTML.match(/<script>([\s\S]*?)<\/script>/)[1];
const now = Date.now();
const snapshot = {
  curve: [
    { step: 1715000, run: "ccgavn-5m-seed0", train: 3.6, dev: 3.5, shard: "00000", t: new Date(now - 3600e3).toISOString() },
    { step: 1725000, run: "ccgavn-5m-seed0-v2", train: 3.5, dev: 3.4, shard: "02084", t: new Date(now - 60e3).toISOString() },
    { step: 3896658, run: "ccgavn-5m-seed0-v2", train: 3.2, dev: 3.1, shard: "10200", t: new Date(now - 30e3).toISOString() },
  ],
  evals: {
    "eval-results/ccgavn-5m-seed0-1620k-frozen/eval-summary.json": {
      checkpoint: "ccgavn-5m-seed0/checkpoint-1620000", returncode: 0,
      mate: [{ correct: 3668, total: 4000, pct: 91.7 }],
      puzzles: [{ correct: 7431, total: 10000, pct: 74.31 }],
    },
    "eval-results/ccgavn-5m-seed0-v2-1800k-preview/eval-summary.json": {
      checkpoint: "ccgavn-5m-seed0-v2/checkpoint-1800000", returncode: 0,
      mate: { correct: 3600, total: 4000 }, puzzles: { solved: 7200, total: 10000 },
    },
    "eval-results/ccgavn-5m-seed0-v2-3896k-final/eval-summary.json": {
      checkpoint: "ccgavn-5m-seed0-v2/checkpoint-3896658", returncode: 0, complete: true,
      mate: { correct: 3710, total: 4000 }, puzzles: { solved: 7620, total: 10000 },
    },
  },
  reference: [
    { name: "Ruoss 9M (teacher)", mate: 98.72, puzzles: 86.13, ref: true },
  ],
  kernels: [
    { account: "acct-1", kernel: "ccgavn-2b", status: "RUNNING" },
    { account: "acct-3", kernel: "build-2b-slice", status: "COMPLETE" },
    { account: "acct-1", kernel: "ccgavn-refutation-r0", status: "RUNNING" },
  ],
  quota: { "acct-1": "28.9", "acct-3": "28.7" },
  quota_reset: "2026-10-10T00:00:00Z",
  runs: {
    "ccgavn-5m-seed0/run-status.txt": "DONE — 1.62M steps",
    "ccgavn-5m-seed0-v2/run-status.txt": "DONE — final eval archived",
    "refutation-r0/run-status.txt": "IN PROGRESS: planner training",
    "refutation-data-r0/run-status.txt": "DONE — 1944 rows",
    "ccgavn-refutation-r0-seed0-v2/run-status.txt": "IN PROGRESS: post-training",
  },
  refutation: {
    phase: "training", source_ready: true, label_done: 1944, label_total: 1944, label_pct: 100,
    accepted: 1900, rejected: 44, label_rate: 0.35, eta_seconds: 3600,
    train_step: 4200, train_pct: 33.6, train_loss: 2.31, train_dev: 0.42, eval: null,
    top_status: "IN PROGRESS: planner training", data_status: "DONE — 1944 rows",
    train_status: "IN PROGRESS: post-training", kernel_status: "RUNNING",
    updated_at: new Date(now - 45e3).toISOString(),
  },
  notices: [{ t: now, text: "corrected-v2 training armed — steps 1.715M → 3.896M" }],
  corpus: {
    target_rows: 1920690429, labeled_rows: 1786000000, shards_planned: 108,
    shards_done: 100, original_rows: 94277038,
    done_tags: ["00000"], all_tags: ["00000", "02084"], rows_by_tag: { "00000": 16764154, "02084": 24000000 },
  },
  build_accounts: {},
  stages: { train320: 100, eval320: 100, corpus1b: 93, train1b: 100, eval1b: 100, train2b: 5, eval2b: 0 },
  updated_at: new Date(now).toISOString(),
};

const elements = new Map();
function el(id) {
  return {
    id, innerHTML: "", textContent: "", value: "", style: {}, dataset: {},
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    appendChild() {}, querySelector() { return null; }, querySelectorAll() { return []; },
    getContext() { return {}; },
    getBoundingClientRect() { return { width: 0, height: 0, top: 0, left: 0, bottom: 0, right: 0 }; },
    addEventListener() {}, removeEventListener() {}, setAttribute() {}, removeAttribute() {},
    focus() {}, blur() {}, scrollIntoView() {}, children: [], parentElement: null, closest() { return null; },
    insertBefore() {}, remove() {},
  };
}
const document = {
  getElementById: (id) => { if (!elements.has(id)) elements.set(id, el(id)); return elements.get(id); },
  querySelector: () => null, querySelectorAll: () => [], createElement: (t) => el(t),
  addEventListener() {}, body: el("body"), documentElement: el("html"), head: el("head"),
};
const sandbox = {
  document, console, Math, JSON, Date, Number, String, Array, Object, Set, Map, RegExp,
  isFinite, parseInt, parseFloat, Intl,
  fetch: async () => ({ ok: true, status: 200, json: async () => snapshot, text: async () => JSON.stringify(snapshot) }),
  setInterval: () => 0, clearInterval() {},
  setTimeout: (fn) => { fn(); return 0; }, clearTimeout() {},
  localStorage: { getItem: () => null, setItem() {}, removeItem() {} },
  location: { hash: "#overview", href: "" },
  addEventListener() {}, matchMedia: () => ({ matches: false, addEventListener() {} }),
  requestAnimationFrame: () => 0,
};
sandbox.window = sandbox; sandbox.globalThis = sandbox; sandbox.self = sandbox;

const runnable = script.replace(/load\(\);setInterval\(load,60000\);/, "globalThis.__loadPromise = load();") +
  "\nglobalThis.__load = load;";
await vm.runInNewContext(runnable, sandbox, { filename: "monitor-inline.js" });
await sandbox.__loadPromise;

const required = ["headline", "live-hero", "live-progress", "live-kernel", "live-eval", "live-spec",
  "live-log", "res-hero", "res-table", "res-base", "res-ref", "res-notes", "tl-ref",
  "runs", "kernels", "quota", "stages", "notices", "evaltable", "ckpttable", "corpus", "tl-hero"];
const empty = required.filter((id) => !(elements.get(id)?.innerHTML || "").length);
if (empty.length) {
  console.error(`FAIL: empty dashboard sections: ${empty.join(", ")}`);
  process.exit(1);
}
const htmlFor = (id) => elements.get(id)?.innerHTML || "";
const trainingHero = htmlFor("live-hero");
snapshot.refutation = Object.assign({}, snapshot.refutation, {
  phase: "source", source_ready: false, label_done: 0, label_pct: 0, train_step: 0, train_pct: 0,
  source: { stage: "scanning_games", pct: 42.7, bytes_done: 12680000000,
            bytes_total: 29700000000, games: 814000, updated_at: new Date(now).toISOString() },
});
await sandbox.__load();
const sourceHero = htmlFor("live-hero");
const sourceProgress = htmlFor("live-progress");
const checks = [
  [htmlFor("headline").includes("92.75"), "headline has final MATE"],
  [htmlFor("res-table").includes("CC-GAVN 5M"), "results table has model rows"],
  [trainingHero.includes("planner training"), "live page shows phase"],
  [htmlFor("tl-ref").includes("1,944"), "timeline shows pilot contract"],
  [!HTML.includes("Run has not started"), "stale preflight copy is gone"],
  [sourceHero.includes("Preparing the deterministic pilot source"), "source phase has its own headline"],
  [sourceProgress.includes("scanning Lichess games") && sourceProgress.includes("43%"),
   "source progress shows stage and percent"],
  [!sourceProgress.includes("0 / 1,944"), "source phase hides empty label count"],
  [required.every((id) => !/undefined|NaN/.test(htmlFor(id))), "no undefined/NaN in sections"],
];
const failed = checks.filter(([ok]) => !ok).map(([, name]) => name);
if (failed.length) {
  console.error(`FAIL: ${failed.join(", ")}`);
  process.exit(1);
}
console.log(`render ok: ${required.length} sections populated, ${checks.length} content checks`);
