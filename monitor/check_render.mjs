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
  },
  reference: [
    { name: "Ruoss 9M (teacher)", mate: 98.72, puzzles: 86.13, ref: true },
  ],
  kernels: [
    { account: "acct-1", kernel: "ccgavn-2b", status: "RUNNING" },
    { account: "acct-3", kernel: "build-2b-slice", status: "COMPLETE" },
  ],
  quota: { "acct-1": "28.9", "acct-3": "28.7" },
  quota_reset: "2026-10-10T00:00:00Z",
  runs: {
    "ccgavn-5m-seed0/run-status.txt": "DONE — 1.62M steps",
    "ccgavn-5m-seed0-v2/run-status.txt": "INCOMPLETE — eval cleanup failed",
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

const runnable = script.replace(/load\(\);setInterval\(load,60000\);/, "globalThis.__loadPromise = load();");
await vm.runInNewContext(runnable, sandbox, { filename: "monitor-inline.js" });
await sandbox.__loadPromise;

const required = ["runs", "kernels", "quota", "stages", "notices", "evaltable", "ckpttable", "corpus", "tl-hero"];
const empty = required.filter((id) => !(elements.get(id)?.innerHTML || "").length);
if (empty.length) {
  console.error(`FAIL: empty dashboard sections: ${empty.join(", ")}`);
  process.exit(1);
}
console.log(`render ok: ${required.length} sections populated`);
