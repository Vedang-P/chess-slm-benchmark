"""Generate kaggle_mate_resume.ipynb — finish the 14 positions that were cut
off mid-reasoning in the original 100-position thinking run (run_id
2026-08-03T22:12:40Z), with no token-budget hint and a much larger hard cap.

    python notebooks/build_mate_resume_notebook.py

Scope, precisely: of the original 100, 81 are correct and 5 are wrong --
both are final, scored, publishable answers and this notebook must not touch
them. 14 hit the max_new_tokens ceiling while still mid-reasoning (verified:
12/14 have reasoning_tokens == max_new_tokens exactly; the other 2 have the
same non-"stop" finish signal and comparably long reasoning, just missing
exact usage telemetry from the provider) and never got to write an answer.
Those 14, and only those 14, are re-run here.

Config change from the original run:
  --thinking-budget REMOVED. It is a hint the gateway is documented (README)
  to ignore on many requests, so it added no real ceiling below max_new_tokens
  -- keeping it implied a control that was not actually there.
  --max_new_tokens 32768 -> 131072. The goal is an unbounded-in-practice
  budget: let the model reach an answer, right or wrong, rather than cutting
  it off mid-thought. A live probe already confirms this works: position
  mate-sel-00543 (one of the 14) truncated at 32768 and answered CORRECTLY at
  a 65536 cap using only 10,789 tokens -- it needed room to finish, not an
  unlimited budget.
--force-answer-prompt is KEPT, matching the other 86, so the merged 100 share
one identical prompt and differ only in how much budget was available.
"""
from __future__ import annotations

import json
from pathlib import Path

from build_notebook import _code, _md, _notebook

NB_DIR = Path(__file__).resolve().parent

# The definitive 14 -- verified against results/mate-selection-thinking100/
# (the merged 100 with the 2 api_error positions already re-run and resolved
# correct). The seed cell asserts the downloaded set matches this exactly
# before writing anything, so a mismatch fails loudly instead of silently
# processing the wrong positions.
SOURCE_RUN_ID = "2026-08-04T10:52:23Z"
EXPECTED_MISSING = sorted([
    "mate-sel-00543", "mate-sel-00640", "mate-sel-01167", "mate-sel-02586",
    "mate-sel-02999", "mate-sel-03955", "mate-sel-04111", "mate-sel-04124",
    "mate-sel-04600", "mate-sel-04709", "mate-sel-05255", "mate-sel-06659",
    "mate-sel-09605", "mate-sel-10387",
])
OUT_DIR = "results/mate-selection-thinking100-unbounded"
RUN_NAME = "deepseek-v4-flash_mate-selection-test_strategy"
NEW_MAX_NEW_TOKENS = 131072

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
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-U", "-r", "requirements.txt"], check=True)
import torch, transformers
print("transformers", transformers.__version__, "| cuda", torch.cuda.is_available())
'''.strip()

GATE_CELL = r'''
status = subprocess.run([sys.executable, "scripts/test_engine.py"], capture_output=True, text=True)
if status.returncode != 0:
    print(status.stdout[-2000:]); print(status.stderr[-2000:])
    raise RuntimeError("test_engine failed")
print("ALL TESTS PASSED")
'''.strip()

SEED_CELL = rf'''
import json
from pathlib import Path
from huggingface_hub import hf_hub_download

SOURCE_RUN_ID = {SOURCE_RUN_ID!r}
EXPECTED_MISSING = {EXPECTED_MISSING!r}
OUT_DIR = Path({OUT_DIR!r})
RUN_NAME = {RUN_NAME!r}

src = hf_hub_download(
    repo_id="vedangfake/chess-bench-results", repo_type="dataset",
    filename=f"runs/{{SOURCE_RUN_ID}}/{{RUN_NAME}}.samples.jsonl",
)
rows = [json.loads(l) for l in Path(src).read_text().splitlines() if l.strip()]
assert len(rows) == 100, f"expected 100 rows in the source run, got {{len(rows)}}"

missing = sorted(r["position_id"] for r in rows if r["status"] == "no_answer")
assert missing == EXPECTED_MISSING, (
    "the source run's no_answer set does not match the expected 14 -- "
    f"stopping rather than guessing.\\n  expected: {{EXPECTED_MISSING}}\\n  got:      {{missing}}"
)

keep = [r for r in rows if r["status"] != "no_answer"]
assert len(keep) == 86, len(keep)

OUT_DIR.mkdir(parents=True, exist_ok=True)
dest = OUT_DIR / f"{{RUN_NAME}}.samples.jsonl"
with dest.open("w") as f:
    for r in keep:
        f.write(json.dumps(r) + "\\n")

print(f"seeded {{len(keep)}} already-final rows (81 correct + 5 wrong) -> {{dest}}")
print(f"will attempt exactly these {{len(missing)}} positions: {{missing}}")
'''.strip()

RUN_CELL = rf'''
import time
from pathlib import Path

out = Path({OUT_DIR!r})
out.mkdir(parents=True, exist_ok=True)
cmd = [sys.executable, "scripts/run_mate_eval.py",
       "--model", "deepseek-v4-flash",
       "--n", "100",
       "--max_new_tokens", "{NEW_MAX_NEW_TOKENS}",
       "--force-answer-prompt",
       "--output_dir", {OUT_DIR!r},
       "--live-push",
       "--verbose",
       "--resume"]
t0 = time.time()
res = subprocess.run(cmd)
print(f"run exited rc={{res.returncode}} after {{(time.time()-t0)/3600:.2f}}h")
'''.strip()

SUMMARY_CELL = rf'''
import json, glob, collections

rows = []
for f in glob.glob(str(Path({OUT_DIR!r}) / "*.samples.jsonl")):
    for line in open(f):
        if line.strip(): rows.append(json.loads(line))
n = len(rows)
by_status = collections.Counter(r["status"] for r in rows)
correct = sum(bool(r["compliance"]) for r in rows if r["status"] != "api_error")
print(f"total rows: {{n}} (expect 100)")
print("status breakdown:", dict(by_status))
print(f"accuracy (of {{n - by_status.get('api_error', 0)}} scored): "
      f"{{correct}}/{{n - by_status.get('api_error', 0)}}")

resolved = [r for r in rows if r["position_id"] in {EXPECTED_MISSING!r}]
print(f"\\nthe {{len(resolved)}} re-run positions:")
for r in sorted(resolved, key=lambda r: r["position_id"]):
    u = r.get("token_usage") or {{}}
    print(f"  {{r['position_id']:>14}}  status={{r['status']:<12}}  "
          f"label={{r.get('label')}}  output_tok={{u.get('output_tokens')}}  "
          f"finished={{r.get('finished')}}")

if by_status.get("no_answer", 0) == 0 and by_status.get("api_error", 0) == 0:
    print("\\nCLEAN 100: every position has a conclusive, scored answer.")
else:
    print(f"\\nNOT YET CLEAN: {{by_status.get('no_answer', 0)}} still no_answer, "
          f"{{by_status.get('api_error', 0)}} still api_error.")
'''.strip()


def build() -> list:
    cells = [
        _md("# MATE Move-Selection: Resume the 14 Truncated Positions\n\n"
            f"Finishes the 14 of 100 positions from run `{SOURCE_RUN_ID}` that hit "
            "the max_new_tokens ceiling mid-reasoning and never produced an answer. "
            "The other 86 (81 correct + 5 wrong) are untouched and already final -- "
            "this notebook seeds them from Hugging Face and only asks the model for "
            "the missing 14.\n\n"
            "**Config change from the original run:** `--thinking-budget` is dropped "
            "(the gateway is documented to ignore it on many requests -- it implied "
            "a control that was not really there). `--max_new_tokens` goes from "
            f"32768 to {NEW_MAX_NEW_TOKENS} so the model is not cut off before it "
            "reaches a conclusive answer, right or wrong. `--force-answer-prompt` is "
            "kept for prompt parity with the other 86.\n\n"
            "Secrets needed: `GITHUB_TOKEN` (clone + live push), `HF_TOKEN` (results "
            "upload), `OPENCODE_API_KEY` (deepseek gateway).\n\n"
            "Honesty contract: no fallbacks, no retries on content. Empty answers are "
            "recorded as no_answer with a reason; gateway failures are api_error, "
            "never scored as a model answer."),
        _md("## 1. Get the repo"),
        _code(CLONE_CELL),
        _md("## 2. Dependencies"),
        _code(DEPS_CELL),
        _md("## 3. Engine/dataset gate"),
        _code(GATE_CELL),
        _md("## 4. Seed the 86 already-final answers\n\n"
            "Downloads the completed 100-position run from the public HF dataset, "
            "verifies its no_answer set is exactly the expected 14, and writes only "
            "the other 86 locally so `--resume` skips them."),
        _code(SEED_CELL),
        _md("## 5. Run the missing 14 (unbounded budget)\n\n"
            "`--resume` skips the 86 seeded rows and attempts only the 14 missing "
            "positions. `--live-push` streams progress and live thinking to the "
            "dashboard; `--verbose` publishes state after every position (there are "
            "only 14, so the default every-25 cadence would barely update)."),
        _code(RUN_CELL),
        _md("## 6. Results summary"),
        _code(SUMMARY_CELL),
        _md("## Notes\n"
            "- Scope: ONLY the 14 positions that never produced an answer. The 81 "
            "correct and 5 wrong positions from the original run are not "
            "re-generated -- they are copied forward unchanged.\n"
            "- No artificial cutoff: thinking-budget hint removed, max_new_tokens "
            f"raised to {NEW_MAX_NEW_TOKENS}. If a position still does not answer at "
            "this cap, that is a real finding (the model's reasoning did not "
            "converge), not a data-collection artifact.\n"
            "- Results auto-upload to HF (vedangfake/chess-bench-results, a fresh "
            "run_id) and the live dashboard (chess-bench-live.pages.dev)."),
    ]
    return cells


if __name__ == "__main__":
    out = NB_DIR / "kaggle_mate_resume.ipynb"
    out.write_text(json.dumps(_notebook(build()), indent=1))
    print(f"wrote {out}")
