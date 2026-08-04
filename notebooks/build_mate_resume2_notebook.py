"""Generate kaggle_mate_resume2.ipynb — finish the last 5 positions that came
back no_answer even at the 131072-token cap (run_id 2026-08-04T12:03:24Z),
using the silent-stream retry fix in src/models.py.

    python notebooks/build_mate_resume2_notebook.py

Scope, precisely: of that 100, 95 are final (90 correct, 5 wrong) and must
not be touched. 5 came back no_answer a second time -- but NOT because they
ran out of budget: measured directly, 4 of them had stream_events=0 (the
gateway connection opened and stayed open 92-147s delivering literally zero
tokens before closing, no finish_reason either way) and the 5th streamed
87,808 real reasoning characters before cutting off mid-word at 295s. Only
the 5th looks like genuine truncation; the other 4 are a stalled transport
wearing a no_answer costume -- confirmed by re-running mate-sel-00543 in
isolation, with nothing else competing for the gateway: it converged
correctly in 30.7s using 4,161 tokens.

Config is otherwise UNCHANGED from the prior resume (same 131072 cap, same
--force-answer-prompt, thinking still enabled, no thinking-budget hint) --
the only change is the retry fix itself, which lives in the freshly cloned
src/models.py, not in this notebook's flags.
"""
from __future__ import annotations

import json
from pathlib import Path

from build_notebook import _code, _md, _notebook

NB_DIR = Path(__file__).resolve().parent

SOURCE_RUN_ID = "2026-08-04T12:03:24Z"
EXPECTED_MISSING = sorted([
    "mate-sel-00543", "mate-sel-01167", "mate-sel-02586",
    "mate-sel-02999", "mate-sel-04111",
])
OUT_DIR = "results/mate-selection-thinking100-final"
RUN_NAME = "deepseek-v4-flash_mate-selection-test_strategy"
MAX_NEW_TOKENS = 131072

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
    "the source run's no_answer set does not match the expected 5 -- "
    f"stopping rather than guessing.\n  expected: {{EXPECTED_MISSING}}\n  got:      {{missing}}"
)

keep = [r for r in rows if r["status"] != "no_answer"]
assert len(keep) == 95, len(keep)

OUT_DIR.mkdir(parents=True, exist_ok=True)
dest = OUT_DIR / f"{{RUN_NAME}}.samples.jsonl"
with dest.open("w") as f:
    for r in keep:
        f.write(json.dumps(r) + "\n")

print(f"seeded {{len(keep)}} already-final rows (90 correct + 5 wrong) -> {{dest}}")
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
       "--max_new_tokens", "{MAX_NEW_TOKENS}",
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
print(f"\nthe {{len(resolved)}} re-run positions:")
for r in sorted(resolved, key=lambda r: r["position_id"]):
    u = r.get("token_usage") or {{}}
    print(f"  {{r['position_id']:>14}}  status={{r['status']:<12}}  "
          f"label={{r.get('label')}}  attempts={{r.get('attempts')}}  "
          f"output_tok={{u.get('output_tokens')}}  finished={{r.get('finished')}}")

if by_status.get("no_answer", 0) == 0 and by_status.get("api_error", 0) == 0:
    print("\nCLEAN 100: every position has a conclusive, scored answer.")
else:
    print(f"\nNOT YET CLEAN: {{by_status.get('no_answer', 0)}} still no_answer, "
          f"{{by_status.get('api_error', 0)}} still api_error.")
'''.strip()


def build() -> list:
    cells = [
        _md("# MATE Move-Selection: Resume the Last 5 (Silent-Stream Retry Fix)\n\n"
            f"Finishes the 5 of 100 positions from run `{SOURCE_RUN_ID}` still marked "
            "no_answer -- 4 of which measured stream_events=0 (a stalled gateway "
            "connection, not the model), now covered by the silent-stream retry "
            "in `src/models.py`. The other 95 (90 correct + 5 wrong) are untouched "
            "and already final.\n\n"
            "Secrets needed: `GITHUB_TOKEN`, `HF_TOKEN`, `OPENCODE_API_KEY`.\n\n"
            "Honesty contract unchanged: retries only cover a stream that opened "
            "and delivered zero tokens with no finish signal -- any real content "
            "or explicit finish_reason (including a genuine length cutoff) is "
            "accepted immediately and recorded as-is, never retried away."),
        _md("## 1. Get the repo"),
        _code(CLONE_CELL),
        _md("## 2. Dependencies"),
        _code(DEPS_CELL),
        _md("## 3. Engine/dataset gate (exercises the new retry regression test)"),
        _code(GATE_CELL),
        _md("## 4. Seed the 95 already-final answers"),
        _code(SEED_CELL),
        _md("## 5. Run the missing 5"),
        _code(RUN_CELL),
        _md("## 6. Results summary"),
        _code(SUMMARY_CELL),
        _md("## Notes\n"
            "- Scope: ONLY the 5 positions still lacking a conclusive answer. "
            "The other 95 are copied forward unchanged.\n"
            "- If any of the 4 silent-stream cases are STILL no_answer after "
            f"up to {2 + 1} attempts each, that's a real finding (a persistently "
            "unavailable gateway for that request), not a retry-count bug.\n"
            "- Results auto-upload to HF (vedangfake/chess-bench-results, a "
            "fresh run_id) and the live dashboard."),
    ]
    return cells


if __name__ == "__main__":
    out = NB_DIR / "kaggle_mate_resume2.ipynb"
    out.write_text(json.dumps(_notebook(build()), indent=1))
    print(f"wrote {out}")
