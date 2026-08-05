# Project Memory & Decisions Log

Append-only record of user decisions and rationale. Update on every change.

## 2026-08-04 — MATE thinking-run decisions

- **Positions:** 100, **the same positions as the direct-mode run** (per-position
  thinking-vs-direct comparison).
- **Run venue:** Kaggle notebook (autonomous, >6h sessions; laptop can close).
  Local runs only for quick tests.
- **First test:** whether the model can be FORCED to answer within a 2048-token
  budget via prompt instruction alone ("you MUST output an answer; if unsure,
  output your best guess from the candidates"). This is prompt-level forcing —
  the answer text still comes from the model; no runner-level fallback.
- **no_answer handling (critical):** must carefully classify WHY there is no
  answer: model gave up (finished, empty) vs cut off (truncated by budget) vs
  could not finish thinking. Store as a per-sample reason.
- **B-preference:** user does not believe it is real; NO order-swap test.
- **Primary goals:** (a) accuracy with thinking, (b) thinking-vs-direct gap.
  Live website watching is a side effect for monitoring correctness.
- **Honesty:** no fallbacks, ever. No retry on empty. Direct-mode 48.6% stays
  as the baseline arm of the paper.
- **Gemma thinking:** thinking must eventually be ON for gemma too ("otherwise
  we have no paper") — we will need to improve/fine-tune that ability; make it
  reason in a caveman-like way as deepseek does. Fine-tune approach TBD.
- **After the thinking run:** 2 gemma baseline runs, then gemma E2B fine-tune
  (50k MATE examples, 4-bit LoRA).
- **API cost:** opencode-go subscription; deepseek flash is very cheap. Do the
  math before runs. 100 positions x ~1-5k tokens = well under 1M tokens.
- **Infra:** single worker (gateway serializes per key), tmux + watchdog +
  resume + HF upload for local; Kaggle notebook for long sessions.

## 2026-08-04 — forcing-prompt test results (measured)

- Forcing prompt at 2048 total: 1/10 answered (the 1 was correct), 9 truncated.
- Forcing at 8192 total: 2/10 answered (both correct), 8 truncated.
- Forcing at 32768 total (16384 thinking budget): 4/5 answered, ALL correct,
  1 truncated, 210s/position average. The one truncation burned 33k tokens.
- CONCLUSION: we CANNOT force an answer within 2048. The thinking arm runs at
  32768 total; completion ~80%, accuracy on completions ~100% (n=5, small).
- Per-position evidence of the thinking gap: mate-sel-00477 direct=wrong,
  thinking=correct.
- Planned 100-position thinking run: --n 100 --thinking-budget 16384
  --max_new_tokens 32768 --force-answer-prompt, Kaggle session (~6h),
  same first-100 positions as the direct run.

## 2026-08-04 — Kaggle run logistics
- The MATE eval needs NO GPU (pure API calls) — kernels pushed with
  enable_gpu: false to avoid burning T4 quota. v2 supersedes v1.
- Kaggle CLI push-to-supersede is the stop mechanism (API abort endpoints
  are 404/dead).
- kernel: vedangpandeyyy/mate-thinking-100 (private, CPU, internet on).

## 2026-08-04 — Kaggle versioning gotcha (FIXED)
- Pushing a new kernel version does NOT cancel the running one — both ran
  simultaneously, both pushed live.json, and the dashboard cycled through
  positions with no visible answers. GPU version also kept burning quota.
- Fix: `kaggle kernels delete <owner>/<slug>` kills all versions; then
  re-push ONE clean version (enable_gpu false). Single run_id verified.

## 2026-08-04 — Codebase audit (correctness pass)

Full-repo audit for correctness/validity defects. Findings and fixes:

**Run-blocking**
- `HFModel.generate` had no `on_chunk` / `thinking_budget` / `thinking_disabled`
  parameters, but both runners pass them. Every non-smoke gemma run died with
  `TypeError` on position 1 — the entire local half of the study could not run.
  Added the parameters plus a streamer that honours `on_chunk`.

**Measurement validity**
- mate-in-2 was prompted with the mate-in-1 objective ("Deliver CHECKMATE in
  exactly one move") on every position — an impossible demand — then scored
  against the first move of a two-move line. Added `build_mate2_prompt`.
- Gateway errors were returned as `content="ERROR HTTP ..."` and scored as
  model text: an infrastructure failure counted as `parse_error`, and an error
  body containing `"code":"e4"` parsed as the legal SAN move `e3e4`. Now
  `api_error` with empty content, excluded from every rate.
- `thinking_enabled` was hardcoded to `model == "deepseek-v4-flash"`, so the
  direct-mode run was recorded as thinking-enabled. Now records the flag.
- MATE A/B parsing took the FIRST `MoveA`/`MoveB` mention. Harmless for
  deepseek (terse `content`), wrong for any model that reasons inside
  `content` — i.e. every gemma run. Now takes the last, as the from-to rule
  already did. Re-scored the shipped 1000: 1000/1000 identical.
- SAN extraction took the first legal token; now the last, matching `MOVE:`.

**Data integrity**
- Re-running a cell APPENDED to its samples JSONL, mixing attempts (and
  scorer versions) in the file the paper figures read. `ResultWriter` now
  truncates unless resuming.
- `n_samples` counted only the last process's rows: the completed 1000-position
  MATE run recorded 192, which also told `run_suite --resume` it was unfinished.
- `analyze_paper_figures` knew only the tactical status vocabulary, so every
  MATE sample landed in the `no_answer` bucket with a 0 legal rate.

**Sampling**
- `build_lichess_mates` bucketed on a hash of the puzzle id (not the rating,
  despite `RATING_BANDS`) and had no per-band quota, so all 250 records came
  from one bucket. Fixed to real rating-band stratification + seeded output
  shuffle. **The committed sets still predate this and are unchanged** —
  rebuilding replaces the data the Phase-1 numbers were measured on.
- `build_bestmove_evals` selected in FEN-lexicographic order, so all 120
  positions have an empty a8. Now a seeded random sample of the eligible pool.

**Paper**
- Table 2 "Model picks A/B: 51%" -> 48.8% / 48.5% (recomputed from raw).
- Phase 1 described as "twenty positions"; it is 5 positions x 4
  representations. "Two illegal moves, the only illegal moves in the study" ->
  three (2 bitboard, 1 piece list), as Table 1 itself shows.

**Open / needs a decision**
- Rebuild the committed task sets with the fixed builders? Verified dry-run:
  mate1 805-2397 (median 1320) and mate2 801-2730 (median 1654), ~25/band, vs
  the current 801-1936 / 801-2212 with 70% under 1200.
- `configs/suite.yaml` `full_n: 40` — a "full" sweep scores 40 positions per
  task, not the 250/250/120 the README and paper Table 1 state.
- The paper names candidate-order randomization as the immediate next
  experiment; the 2026-08-04 log says the user does not want an order-swap
  test. One of the two needs updating.
- `kaggle_mate.ipynb` keeps `--force-answer-prompt` per the logged decision,
  but the direct-mode baseline used the plain spec: prompt and thinking mode
  differ together. A forced direct-mode cell is needed as the control.

## 2026-08-04 — Resume the 14 truncated MATE positions, unbounded budget

Of the original 100-position thinking run (run_id 2026-08-03T22:12:40Z): 81
correct, 5 wrong (both final — not re-run), 2 api_error (already re-run
locally this session, both resolved correct), 14 no_answer/truncated —
verified genuine budget truncation (12/14 have reasoning_tokens == max_new_tokens
exactly; the other 2 share the same finish signal and comparable reasoning
length but the provider didn't report exact usage for those two calls).

**User decision:** drop `--thinking-budget` (a hint the gateway is documented
to ignore on many requests — it implied a control that wasn't really there).
Raise `--max_new_tokens` from 32768 to 131072 so the model is never cut off
before reaching a conclusive answer, right or wrong. Confirmed this isn't
overkill: a probe at 65536 already resolved one of the 14
(`mate-sel-00543`, correct, using only 10,789 tokens).

**Scope, explicit:** only the 14 no_answer positions are re-run. The 81
correct and 5 wrong are copied forward untouched — the 5 wrong are a final,
publishable result, not a data-collection failure.

**Execution:** `notebooks/build_mate_resume_notebook.py` →
`notebooks/kaggle_mate_resume.ipynb` (git-tracked, no secrets — seeds the 86
final rows from HF, asserts the no_answer set matches the expected 14 exactly
before writing anything, runs `run_mate_eval.py --resume --live-push --verbose`
with the new budget). Pushed to Kaggle (CPU-only,
`vedangpandeyyy/mate-thinking-100-resume-14-truncated`, private) reusing the
credential-setting cell from the original `mate-thinking-100` kernel verbatim
— not retyped, not added to git. Results land in
`results/mate-selection-thinking100-unbounded/`, auto-upload to
`vedangfake/chess-bench-results` under a fresh run_id, and stream live to
chess-bench-live.pages.dev via the fixed MATE-run dashboard.

**Flag for the user:** the source kernel (`mate-thinking-100`) has
`GITHUB_TOKEN`/`HF_TOKEN`/`OPENCODE_API_KEY` hardcoded as plaintext in a code
cell rather than attached via Kaggle's Secrets UI (the notebook's own markdown
says to use Secrets — this predates that). It's a private kernel so not
publicly exposed, but worth migrating to Secrets and rotating those three
tokens when convenient.

---
---

## 2026-08-05 — Gemma 4 E2B 1000-position campaign (2 GPU workers + CPU aggregator)

**Goal:** the SAME 1000 positions DeepSeek direct-mode scored (48.6% strict),
with local gemma4-e2b thinking ON (check10-validated methodology: --local-
thinking, --force-answer-prompt, --max_new_tokens 32768).

**Parallelism:** Kaggle free tier allows 2 concurrent GPU kernels => w1 =
positions [0:500), w2 = [500:1000). Each worker: --worker-id wN
--live-namespace gemma --hf-upload-every 25 --resume, own run_id on HF
(gemma1000-w1 / gemma1000-w2).

**Demo-first:** each worker notebook runs a 2-position demo through the REAL
pipeline + an inspect cell that RAISES if parsing/thinking-split is broken on
the assigned GPU (verified on P100 via check10: 78.7s/pos -> 500 positions ~
11h, within the 12h session limit). The demo shares run id/output dir, so the
full run resumes past it.

**Backups:** HF upload every 25 positions + at end; recovery cell pulls the
worker's own checkpoint from HF before --resume (/kaggle/working is wiped on
every session restart — without this a died session restarts from zero).

**Aggregator:** aggregate_live_state.py gained --namespace/--run-id (defaults
unchanged for the deepseek campaign); a third CPU-only kernel runs it in a
45s loop combining monitor/gemma/workers/* into the gemma dashboard page.

**Secrets:** hardcoded env vars injected at build time into
notebooks/push_gemma1000/ (gitignored — .gitignore now covers notebooks/push_*/).
User runs the kernels manually from the Kaggle UI and selects the
accelerator themselves (free tier auto-assigns T4/P100; cannot choose).

## 2026-08-05 — Gemma E2B live arm: three verified bugs, all fixed (run stopped)

**Stop:** the first `mate-gemma-e2b-100-positions` kernel was deleted mid-run at
the user's request (100-run was ~16/100) after the live dashboard showed three
problems. The 5-position probe data is safe on HF
(`runs/mate-selection-e2b-100-20260805`, 2/5 correct, all 5 parsed).

**Verified root causes (code + data):**
1. **Thinking never in the right box, answer box filled with the thought
   blob** — `_LocalStreamer` decoded with `skip_special_tokens=True`, stripping
   the `<|channel>thought` markers mid-stream, so every chunk went out as
   `content` with `reasoning: ""`. The final data was CORRECT (the end-of-
   generation `parse_response` split worked — probe samples have reasoning
   split, tokens counted); only the live view was broken.
2. **Site never shows a final answer** — the scored state IS written per
   position, but live.json uploads are throttled to 1/300s and the next
   position's streaming chunks (every 2s) overwrite the pending bytes before
   the throttle reopens. The dashboard only ever showed mid-generation blobs.
3. **"Expert's choice hangs a queen"** — display issue, not a scoring bug.
   mate-sel-01873: black is in check (Qa4-e8); truth B (`e8f7`) is correct and
   verified legal; candidate A (`a5b5`) is the queen-hang (Qxb5). The site
   printed "candidates A a5b5 / B e8f7" directly under the "expert choice"
   label and the misplaced thinking blob discussed a5b5, so it read as if the
   site endorsed the queen hang. Dataset-wide legality scan: 0/1000 illegal
   candidate or truth moves. (One suspicious truth found in the first 100:
   mate-sel-01655 `b5c4` — engine check pending, not treated as noise yet.)

**Fixes (commit ae7001e on mate-e2b-kaggle):**
- Streamer decodes with special tokens when `local_thinking` and re-splits the
  thought channel on every chunk (`_LocalStreamer._split`, marker-split safe
  across token boundaries); unit-tested: split, truncated, marker-less, and
  non-thinking paths. The end-of-generation `parse_response` split remains
  authoritative.
- `run_mate_eval.py`: scored live states (`phase: "scored"`) bypass the 300s
  throttle — every position's verdict is published (same exemption the
  complete state had).
- Dashboard: MATE boards draw the non-expert candidate in gray next to the
  green expert pick; legends updated on both pages.
- New `scripts/watch_kaggle_kernel.py`: polls a Kaggle kernel, logs + macOS
  notifies + downloads output on completion.

**Next:** validate with the check10 kernel (`mate-gemma-e2b-check10-fixed-
live-arm`, pushed 2026-08-05) — needs GITHUB_TOKEN + HF_WRITE_TOKEN attached
as Kaggle secrets BEFORE it finishes the probe; then re-run the full 100.

**Frontend deploy:** `npx wrangler pages deploy frontend/ --project-name
chess-bench-live` needs a CLOUDFLARE_API_TOKEN (not stored in the repo); the
arrow/legend fix ships to the live site only when the user runs it.

---

## 2026-08-04: local gemma4-e2b MATE thinking arm (--local-thinking)

**Decision:** run Gemma 4 E2B (4-bit, Kaggle T4, no API) on the EXACT 100
positions of the DeepSeek thinking-final arm (`results/mate-selection-
thinking100-final/`, 94/100), with the thought channel ENABLED. New
`--local-thinking` flag on `run_mate_eval.py`: local gemma renders with
`enable_thinking=True`; the `<|channel>thought ... <channel|>` block is split
off via `processor.parse_response` (the checkpoint's shipped
response_template) with a channel-marker fallback in `src/models.py`.
Thinking text lands in `reasoning`, `token_usage.reasoning_tokens` counts the
thinking block, and samples/summary/report upload to HF exactly like the
gateway arms. `thinking_enabled` is recorded from the flag, not from the
model name.

**Why not the default:** gemma E2B with thinking ON and a small budget
observably burns everything on reasoning (parse_rate 0.0 at 1024 tokens), so
thinking stays an explicit per-arm choice — the FEN suite keeps its fixed
disabled-thinking behavior.

**Confound vs DeepSeek thinking-final (flagged):** that arm had a 16384
thinking budget inside its 32768 total; local gemma has ONE undivided budget
(the whole stream is `--max_new_tokens 32768`).

**Flow:** probe 5 positions first (real pipeline, same output dir, shared
BENCH_RUN_ID so the HF archive gets one complete cell), inspect cell raises
if 0/5 parse, then the 100-run resumes past the probed 5.
`notebooks/build_gemma_e2b_mate_notebook.py` →
`notebooks/kaggle_mate_gemma_e2b.ipynb` (branch `mate-e2b-kaggle`).
