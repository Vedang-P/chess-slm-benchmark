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
