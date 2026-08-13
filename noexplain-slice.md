# Noexplain-First Vertical Slice — the provable pipeline milestone

Decision 2026-08-12. First goal: **beat deepseek on the noexplain testbed** —
the hardest subset (MATE's 8B fine-tune: 63.5% noexplain vs 89.7% strategy)
and the purest chess signal (no explanation text in the prompt to lean on).
Prove the full pipeline on ONE subset, then broaden to all four formats.

## Ground truth established 2026-08-12

- **No deepseek noexplain baseline exists in our records.** Monitor archives
  cover strategy only (deepseek sweep; gemma strategy = 611/1000 = 61.1%).
  README claims noexplain "ran" but no number was committed.
- The strategy baseline (85.8%) itself needs protocol clarity: the strategy
  campaign config shows `force_answer_prompt: true`, while clean-1000 samples
  show unforced prompts. One protocol must be locked and applied everywhere.
- Deepseek strategy efficiency anchor: ~8,800 reasoning tokens/position,
  ~79s latency (monitor). Our tokens-per-correct win is measured against this.

## Step 0 — Establish the noexplain baseline (gateway time, no GPU, ~4-5h)

- deepseek-v4-flash on `mate-selection-test-noexplain.json` (1000),
  thinking ON, unbounded, `ANSWER_SPEC` unforced, last-mention parse
  (`run_mate_eval.py` default; 5 parallel CPU workers × own API key, the
  strategy-campaign pattern).
- Outputs: (a) the number we must beat, (b) deepseek's reasoning-token
  distribution on noexplain (efficiency baseline), (c) protocol-parity
  template for every later eval.
- ALSO: a 2-worker re-run of deepseek strategy (unforced, clean) to
  re-confirm 85.8% under the locked protocol — one protocol, two numbers.

## Step 1 — Noexplain SFT (build 1-2h CPU, train 15-20h T4)

- **Labels**: ~500-600k rows from the noexplain pool (1.42M available),
  phase-natural (~91/6/3), test-FEN-excluded, `MoveX:<move>` answers.
- **3k verified lucid traces** (the deepseek budget): positions from the
  noexplain pool, difficulty-gated (near-equal Stockfish evals — the
  positions labels can't decide), phase-natural, format naturally noexplain.
  Every claim Stockfish-verified at d14 (final choice == engine best,
  intermediate moves legal + eval-stable ±100cp), `Verified: yes` footer.
- **Self-generated verified traces** (20-60k, free): best-of-N samples from a
  competence checkpoint, Stockfish-filtered, on-policy, same distribution.
- Eval: noexplain 1000, thinking ON, locked protocol, tokens-per-correct.

## Step 2 — RLVR (remaining budget)

GRPO + outcome(1.0)/process(0.3)/calibration(0.2)/style(0.1) rewards,
DAPO/Dr.GRPO/S-GRPO stabilizers, near-equal-eval rollout pool.

## Step 3 — Provable result

- SFT alone: target 70-78% on the HARDEST subset (MATE 8B: 63.5%).
- +RLVR + self-consistency: toward/over the deepseek noexplain baseline.
- If the slice works on noexplain, broaden to all 4 formats (labels 25% per
  format pool + traces per format) — additive, same machinery.

## Slice caveats (honest)

- Noexplain-only training may drift on explanation formats — accepted for
  the slice; the full run adds format coverage (format-balanced labels).
- Eval baseline must be re-measured under the locked protocol before any
  "beat deepseek" claim; the 85.8% strategy number is protocol-suspect.
