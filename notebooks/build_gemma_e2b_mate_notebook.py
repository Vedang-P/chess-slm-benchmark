"""Generate kaggle_mate_gemma_e2b.ipynb — local Gemma 4 E2B, thinking ON,
on the EXACT 100 MATE positions the DeepSeek thinking arm ran.

    python notebooks/build_gemma_e2b_mate_notebook.py

Flow: clone -> deps -> gate -> 5-position probe (real pipeline, thinking on)
-> inspect (parsing/extraction/token accounting MUST look right) -> 100-run
(--resume skips the probed 5) -> summary vs the saved DeepSeek numbers.

Every arm runs `scripts/run_mate_eval.py --model gemma4-e2b --local-thinking`
with the same forced answer prompt and the same 32768 total budget as the
DeepSeek thinking-final arm (results/mate-selection-thinking100-final/,
94/100, saved locally 2026-08-04). Thinking is extracted from the
<|channel>thought ... <channel|> block, reasoning text and reasoning tokens
are recorded, and the samples/summary/report upload to the HF archive
(vedangfake/chess-bench-results) exactly like the DeepSeek runs.
"""
from __future__ import annotations

import json
from pathlib import Path

from build_notebook import _code, _md, _notebook

NB_DIR = Path(__file__).resolve().parent

# The EXACT 100 position ids the DeepSeek thinking-final arm scored,
# extracted verbatim from results/mate-selection-thinking100-final/
# deepseek-v4-flash_mate-selection-test_strategy.samples.jsonl (2026-08-04).
# The probe uses the first 5; the 100-run resumes past them.
MATE100_IDS = [
    "mate-sel-00279", "mate-sel-00477", "mate-sel-00542", "mate-sel-00563",
    "mate-sel-00666", "mate-sel-00747", "mate-sel-00832", "mate-sel-00839",
    "mate-sel-00852", "mate-sel-00883", "mate-sel-01008", "mate-sel-01020",
    "mate-sel-01032", "mate-sel-01058", "mate-sel-01119", "mate-sel-01126",
    "mate-sel-01128", "mate-sel-01131", "mate-sel-01137", "mate-sel-01140",
    "mate-sel-01143", "mate-sel-01151", "mate-sel-01152", "mate-sel-01157",
    "mate-sel-01158", "mate-sel-01160", "mate-sel-01164", "mate-sel-01169",
    "mate-sel-01170", "mate-sel-01172", "mate-sel-01180", "mate-sel-01184",
    "mate-sel-01185", "mate-sel-01188", "mate-sel-01189", "mate-sel-01191",
    "mate-sel-01192", "mate-sel-01196", "mate-sel-01199", "mate-sel-01201",
    "mate-sel-01204", "mate-sel-01205", "mate-sel-01207", "mate-sel-01208",
    "mate-sel-01209", "mate-sel-01210", "mate-sel-01211", "mate-sel-01212",
    "mate-sel-01213", "mate-sel-01214", "mate-sel-01215", "mate-sel-01216",
    "mate-sel-01217", "mate-sel-01218", "mate-sel-01219", "mate-sel-01221",
    "mate-sel-01222", "mate-sel-01223", "mate-sel-01224", "mate-sel-01225",
    "mate-sel-01226", "mate-sel-01227", "mate-sel-01228", "mate-sel-01229",
    "mate-sel-01230", "mate-sel-01231", "mate-sel-01232", "mate-sel-01233",
    "mate-sel-01234", "mate-sel-01235", "mate-sel-01236", "mate-sel-01237",
    "mate-sel-01238", "mate-sel-01239", "mate-sel-01240", "mate-sel-01241",
    "mate-sel-01242", "mate-sel-01243", "mate-sel-01244", "mate-sel-01245",
    "mate-sel-01246", "mate-sel-01247", "mate-sel-01248", "mate-sel-01249",
    "mate-sel-01250", "mate-sel-01251", "mate-sel-01252", "mate-sel-01253",
    "mate-sel-01254", "mate-sel-01255", "mate-sel-01256", "mate-sel-01257",
    "mate-sel-01258", "mate-sel-01259", "mate-sel-01260", "mate-sel-01261",
    "mate-sel-01262", "mate-sel-01263",
]
PROBE_IDS = MATE100_IDS[:5]
IDS_ARG = ",".join(MATE100_IDS)
PROBE_ARG = ",".join(PROBE_IDS)

CLONE_CELL = r'''
import os, shutil, subprocess, sys
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "chess-slm-benchmark"
if REPO.exists():
    shutil.rmtree(REPO)

def find_token():
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):
        if os.environ.get(name):
            return os.environ[name]
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("GITHUB_TOKEN")
    except Exception:
        return None

token = find_token()
url = "https://github.com/Vedang-P/chess-slm-benchmark.git"
if token:
    url = url.replace("https://", f"https://x-access-token:{token}@")
res = subprocess.run(["git", "clone", "--quiet", url, str(REPO)],
                     capture_output=True, text=True)
if res.returncode != 0:
    raise RuntimeError("clone failed (token not attached?): " + res.stderr[-300:])
os.chdir(REPO)
print("cwd:", Path.cwd())
'''.strip()

DEPS_CELL = r'''
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-U",
                "-r", "requirements.txt"], check=True)
import transformers, torch
if int(transformers.__version__.split(".")[0]) < 5:
    raise RuntimeError(f"transformers {transformers.__version__} too old "
                       "for Gemma 4 (needs >= 5.13)")
print("transformers", transformers.__version__, "| cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), "| vram GB:", torch.cuda.get_device_properties(0).total_memory / 1e9)
'''.strip()

GATE_CELL = r'''
status = subprocess.run([sys.executable, "scripts/test_engine.py", "--quick"],
                        capture_output=True, text=True)
if status.returncode != 0:
    print(status.stdout[-2000:]); print(status.stderr[-2000:])
    raise RuntimeError("test_engine failed")
print("ALL TESTS PASSED")
'''.strip()

RUN_CMD = r'''
cmd = [sys.executable, "scripts/run_mate_eval.py",
       "--model", "gemma4-e2b",
       "--ids", "{IDS_ARG}",
       "--force-answer-prompt",
       "--max_new_tokens", "32768",
       "--local-thinking",
       "--output_dir", "results/mate-selection-e2b-100",
       "--live-push",
       "--resume",
       {EXTRA}]
'''.strip()

PROBE_CELL = f'''
import json, os, time
from pathlib import Path

# one run id for probe + full run: the archive gets ONE cell, the final
# upload overwrites the 5-position partial (upload_cell re-uploads the
# whole current samples file)
os.environ["BENCH_RUN_ID"] = "mate-selection-e2b-100-" + time.strftime("%Y%m%d")

{RUN_CMD.format(IDS_ARG=PROBE_ARG, EXTRA='"--verbose"')}
t0 = time.time()
res = subprocess.run(cmd)
print(f"probe exit rc={{res.returncode}} after {{(time.time()-t0)/60:.1f}}min")
'''.strip()

INSPECT_CELL = r'''
import json, collections
from pathlib import Path

p = Path("results/mate-selection-e2b-100/deepseek-v4-flash_mate-selection-test_strategy.samples.jsonl")
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
print(f"probe samples on disk: {len(rows)}")
for s in rows:
    tu = s.get("token_usage") or {}
    print(f"{s['position_id']}: {s['status']:11s} label={s.get('label')} "
          f"move={s.get('move')} correct={s.get('compliance')} "
          f"answer_chars={s.get('answer_chars')} reasoning_chars={s.get('reasoning_chars')} "
          f"tokens={{in:{tu.get('input_tokens')} out:{tu.get('output_tokens')} "
          f"reason:{tu.get('reasoning_tokens')}}}")
print()
s = rows[0]
print("--- sample 0 reasoning (first 400 chars) ---")
print((s.get("reasoning") or "")[:400])
print("--- sample 0 output ---")
print(repr((s.get("output") or "")[:200]))
parsed = [r for r in rows if r["status"] in ("correct", "wrong")]
if len(parsed) == 0:
    raise RuntimeError(
        "0/5 positions produced a parseable MoveA/MoveB choice. Parsing or "
        "extraction is broken — fix before the 100-position run. Check the "
        "reasoning/output dump above; do NOT proceed.")
print(f"\nPROBE OK: {len(parsed)}/5 parsed — extraction looks sane.")
'''.strip()

FULL_CELL = f'''
import json, os, time

{RUN_CMD.format(IDS_ARG=IDS_ARG, EXTRA='')}
t0 = time.time()
res = subprocess.run(cmd)
print(f"full run exit rc={{res.returncode}} after {{(time.time()-t0)/3600:.2f}}h")
'''.strip()

SUMMARY_CELL = r'''
import json
from pathlib import Path

p = Path("results/mate-selection-e2b-100/deepseek-v4-flash_mate-selection-test_strategy.samples.jsonl")
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
scored = [r for r in rows if r["status"] != "api_error"]
parsed = [r for r in scored if r["status"] in ("correct", "wrong")]
n = len(scored)
acc = {
    "n": n,
    "n_attempted": len(rows),
    "api_error": sum(r["status"] == "api_error" for r in rows),
    "parse_rate": round(len(parsed) / n, 4) if n else None,
    "accuracy_strict": round(sum(bool(r["compliance"]) for r in scored) / n, 4) if n else None,
    "accuracy_of_parsed": round(sum(bool(r["compliance"]) for r in parsed) / len(parsed), 4) if parsed else None,
    "correct": sum(bool(r["compliance"]) for r in scored),
    "wrong": sum(r["status"] == "wrong" for r in scored),
    "no_answer": sum(r["status"] == "no_answer" for r in scored),
    "parse_error": sum(r["status"] == "parse_error" for r in scored),
}
print("GEMMA 4 E2B (local, thinking ON, forced prompt, 32768 budget):")
print(json.dumps(acc, indent=1))

# DeepSeek reference numbers, saved locally 2026-08-04 (NOT fetched — the
# repo does not track results/):
DEEPSEEK_THINKING_FINAL = {"n": 100, "accuracy_strict": 0.94, "parse_rate": 1.0}
DEEPSEEK_DIRECT_1000 = {"n": 1000, "accuracy_strict": 0.486}
print()
print(f"deepseek-v4-flash thinking-final (same 100 positions): "
      f"{DEEPSEEK_THINKING_FINAL['accuracy_strict']:.0%} strict "
      f"(n={DEEPSEEK_THINKING_FINAL['n']})")
print(f"deepseek-v4-flash direct  (1000 positions):              "
      f"{DEEPSEEK_DIRECT_1000['accuracy_strict']:.0%} strict "
      f"(n={DEEPSEEK_DIRECT_1000['n']})")
'''.strip()


def build() -> list:
    cells = [
        _md("# MATE Move-Selection: Local Gemma 4 E2B, Thinking ON (100 positions)\n\n"
            "Gemma 4 E2B (4-bit, T4, **no API**) on the **exact same 100 positions** the\n"
            "DeepSeek thinking-final arm scored (`results/mate-selection-thinking100-final/`,\n"
            "94/100). Same forced answer prompt, same 32768 total budget. Thinking runs\n"
            "locally via the `<|channel>thought` channel; the thinking text and reasoning\n"
            "tokens are recorded, and samples/summary/report upload to Hugging Face\n"
            "(vedangfake/chess-bench-results) exactly like the DeepSeek runs.\n\n"
            "**Flow: probe 5 positions first, inspect the parsing/extraction, then run the\n"
            "full 100.** `--resume` means the full run skips the probed 5.\n\n"
            "Secrets needed: `GITHUB_TOKEN` (clone + live push), `HF_WRITE_TOKEN` (results\n"
            "archive). `google/gemma-4-E2B-it` is NOT gated, so no HF token is needed to\n"
            "load the model. Attach, SAVE, then Kernel -> Restart & Run All."),
        _md("## 1. Get the repo (fresh clone)"),
        _code(CLONE_CELL),
        _md("## 2. Dependencies (transformers >= 5.13 for Gemma 4)"),
        _code(DEPS_CELL),
        _md("## 3. Engine/dataset gate"),
        _code(GATE_CELL),
        _md("## 4. Probe: 5 positions through the REAL pipeline\n\n"
            "Runs `run_mate_eval.py --local-thinking` on the first 5 of the 100 ids.\n"
            "This is the parsing/extraction test: does the thought channel get split into\n"
            "`reasoning` + `content`? Are `reasoning_tokens` recorded? Does the answer\n"
            "parser land on MoveA/MoveB? Model load is ~1-3 min; 5 positions a few\n"
            "minutes. The probe writes into the SAME output dir as the full run under a\n"
            "shared run id, so the archive ends up with one complete cell."),
        _code(PROBE_CELL),
        _md("## 5. Inspect the probe before proceeding\n\n"
            "Verify, for each of the 5 samples: `status` is correct/wrong (NOT\n"
            "no_answer/parse_error), `reasoning_chars > 0` (thinking was captured),\n"
            "`token_usage.reasoning_tokens > 0` (thinking is accounted like the gateway\n"
            "arm). If 0/5 parse, this cell raises and the 100-run must NOT be started."),
        _code(INSPECT_CELL),
        _md("## 6. The full 100-position run\n\n"
            "Same command, all 100 ids, `--resume` skips the probed 5. Thinking E2B on a\n"
            "T4 is slow (each position can take minutes); expect roughly 3-8h total.\n"
            "`--live-push` streams progress to the dashboard; the HF archive gets the\n"
            "complete cell when the run finishes. If the session dies, re-run this\n"
            "notebook — the probe/run cells skip what is already scored."),
        _code(FULL_CELL),
        _md("## 7. Results vs DeepSeek"),
        _code(SUMMARY_CELL),
        _md("## Notes\n"
            "- **Confound vs DeepSeek thinking-final:** that arm had a 16384-token\n"
            "  *thinking* budget within its 32768 total; local gemma has one undivided\n"
            "  budget (the thinking block and answer share `max_new_tokens`). The prompt,\n"
            "  position set, and total budget are identical.\n"
            "- **Honesty contract:** no retries, no fallbacks. The answer is the model's\n"
            "  own text after channel parsing; a budget-cut generation records its\n"
            "  partial thinking as reasoning and an empty answer (no_answer, reason:\n"
            "  truncated).\n"
            "- **Extraction:** `processor.parse_response` (the checkpoint's shipped\n"
            "  response_template) with a channel-marker fallback in src/models.py.\n"
            "- Results live in `results/mate-selection-e2b-100/` and the HF archive; the\n"
            "  repo itself does not track results/."),
    ]
    return cells


if __name__ == "__main__":
    out = NB_DIR / "kaggle_mate_gemma_e2b.ipynb"
    out.write_text(json.dumps(_notebook(build()), indent=1))
    print(f"wrote {out}")
