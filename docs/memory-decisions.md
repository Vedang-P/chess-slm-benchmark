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
