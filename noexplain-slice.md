# Noexplain-First Vertical Slice — the provable pipeline milestone

Decision 2026-08-12. First goal: **beat MATE on noexplain (63.5%), stretch
to beat deepseek (92.2% (922/1000))** — the purest chess signal (no
explanation text in the prompt to lean on). Prove the full pipeline on ONE
subset, then broaden to all four formats.

## Ground truth (no re-runs — the baselines exist and are verified)

- **deepseek noexplain baseline: 92.2% (922/1000)** — measured, HF archive
  (`mate-noexplain-w1..w5`, thinking ON, unbounded, protocol-comparable).
- **deepseek strategy baseline: 85.8% (858/1000)** (Wilson CI [0.835,0.878],
  verified from the HF archive). The 91.0% merged artifact was investigated
  and rejected — it is not a baseline.
- Deepseek efficiency anchor: ~8,800 reasoning tokens/position, ~79s latency
  (monitor). Our tokens-per-correct win is measured against this.

## Step 0 — Noexplain SFT data (build 1-2h CPU, train 15-20h T4)

- **Labels**: ~500-600k rows from the noexplain pool (1.42M available),
  phase-natural (~91/6/3), test-FEN-excluded, `MoveX:<move>` answers.
- **3k verified lucid traces** (the deepseek budget): positions from the
  noexplain pool, difficulty-gated (near-equal Stockfish evals — the
  positions labels can't decide), phase-natural, format naturally noexplain.
  Every claim Stockfish-verified at d14 (final choice == engine best,
  intermediate moves legal + eval-stable ±100cp), `Verified: yes` footer.
- **Self-generated verified traces** (20-60k, free): best-of-N samples from a
  competence checkpoint, Stockfish-filtered, on-policy, same distribution.
- Eval: noexplain 1000, thinking ON, protocol parity with the 92.2% (922/1000)
  baseline run, tokens-per-correct.

## Step 1 — RLVR (remaining budget)

GRPO + outcome(1.0)/process(0.3)/calibration(0.2)/style(0.1) rewards,
DAPO/Dr.GRPO/S-GRPO stabilizers, near-equal-eval rollout pool.

## Step 2 — Provable result

- **Bar 1 (must beat): MATE 8B fine-tune on noexplain = 63.5%.** SFT alone
  should clear this (target 70-78% on noexplain) — that alone proves the
  pipeline beats MATE's recipe at 1/4 the model size and a fraction of the
  data.
- **Bar 2 (stretch): deepseek on noexplain = 92.2% (922/1000).** +RLVR +
  self-consistency toward/over this.
- If the slice clears Bar 1, broaden to all 4 formats (labels 25% per format
  pool + traces per format) — additive, same machinery.

## Slice caveats (honest)

- Noexplain-only training may drift on explanation formats — accepted for
  the slice; the full run adds format coverage (format-balanced labels).
- Baseline protocol: match the archive runs exactly (thinking ON, unbounded,
  unforced ANSWER_SPEC, last-mention parse).
