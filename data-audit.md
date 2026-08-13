# Data Audit — MATE train & eval, measured 2026-08-12

Every number below was computed today from the local corpus
(`data/raw/mate-train/*/*.jsonl` + `data/positions/mate-selection-test*.json`),
using the committed phase classifier (`scripts/build_phase_dataset.py`).

## 1. Corpus size and dedup

| subset | raw rows | unique FENs | test-FEN overlap |
|---|---|---|---|
| noexplain | 1,420,247 | 1,419,501 | 1 |
| strategy | 1,364,888 | 1,364,797 | 1 |
| tactic | 349,619 | 349,568 | 0 |
| both | 349,619 | 349,568 | 0 |

- Rows ≈ unique FENs: the corpus is essentially deduped already *within*
  subset (the "near-duplicate FENs" note in the old builder is about
  *cross-subset* duplicates — same FEN can appear in noexplain AND strategy).
- Contamination vs the 4 test sets (2952 FENs total): **2 FENs out of
  3,483,000+** — effectively zero. Both are pre-existing in noexplain/strategy
  train; the builder's exclusion handles them (already implemented).

## 2. Phase imbalance — the big finding

Classifier (v1, published): endgame = non-king material ≤13 pts AND ≤6
non-king pieces; opening = fullmove ≤12 AND ≥16 non-king pieces; else
middlegame; sparse flag = ≤3 non-king pieces.

**Training distribution (every subset):**

| subset | opening | middlegame | endgame | sparse |
|---|---|---|---|---|
| noexplain | 5.8% | 91.5% | 2.3% | 0.4% |
| strategy | 5.8% | 91.5% | 2.3% | 0.4% |
| tactic | 5.9% | 91.5% | 2.2% | 0.4% |
| both | 5.9% | 91.5% | 2.2% | 0.4% |

**Eval distribution (the 4×1000 we will score on):**

| eval set | opening | middlegame | endgame | sparse |
|---|---|---|---|---|
| strategy | 5.9% | 90.8% | 2.8% | 0.5% |
| noexplain | 6.3% | 90.6% | 2.8% | 0.3% |
| tactic | 5.3% | 92.0% | 2.4% | 0.3% |
| both | 5.3% | 92.0% | 2.4% | 0.3% |

**Conclusion: the eval is NOT phase-balanced — it mirrors the train
distribution (~91/6/3).** This inverts a naive design assumption: training
phase-balanced (33/33/33) would spend a third of capacity on a phase that is
only 6% of the eval, trading headline accuracy for a benchmark we invented.
Implication (see engineering-decisions.md §2.2):

- Main SFT mix: **natural-proportional** (≈91/6/3) so train matches eval.
- The phase-stratified benchmark (300/phase) remains the separate instrument;
  per-phase accuracy is *reported* on it. If endgame is weak there, a small
  endgame-oversampled boost stage fixes it *after* the main run, not during.

## 3. Label balance

| subset | MoveA share |
|---|---|
| noexplain | 50.0% |
| strategy | 50.0% |
| tactic | 50.0% |
| both | 50.0% |

Perfectly balanced (authors shuffle candidates). A model defaulting to
"always MoveA" would score 50% — no label-bias confound in any direction.

## 4. Format imbalance (second stratification axis)

noexplain/strategy each have ~1.4M rows; tactic/both each ~350k (**4× fewer**).
The four eval sets are equally sized (1000 each) and differ in *prompt text*
(explanation presence). Design consequence:

- If we sample train rows proportionally to corpus size, the model sees ~4×
  more bare-format prompts than tactic/both prompts → format drift on the
  tactic/both testbeds.
- Fix: **format-balanced sampling** (25% of the mix from each of the four
  formats), while staying phase-natural inside each format. Cheap, faithful,
  and directly protects 2 of the 4 testbeds.

## 5. Baseline integrity — the 91% file must NOT be used

`results/clean-1000/summary.json` reports 91.0% (910/1000), but the samples
file is a **merged multi-run artifact**: 965 samples from `deepseek-v4-flash`
across ~6 different run_ids + **35 samples from `deepseek-v4-flash-free`**
(a different model), with rescued/retried samples mixed in. It is not a
single-model single-protocol measurement.

**The official baseline to beat remains the README matrix:**
`deepseek-v4-flash` strategy = **85.8% (858/1000)** — same model, same
protocol (`run_mate_eval.py` default `ANSWER_SPEC`, thinking enabled,
unbounded). Base gemma = **61.1% (611/1000)**.

Evaluation of our fine-tune must use the identical protocol: one model, one
run, `ANSWER_SPEC` (NOT the forced variant), 4 subsets × 1000. This is why the
SFT prompt is byte-identical to the eval prompt (build_mate_lora_data.py §
comment).

## 6. What this means for the design (delta vs engineering-decisions.md)

1. §2.2 mix: **60/40 labels:verified-traces stays**; inside that, sampling is
   format-balanced (25% × 4) and phase-natural (~91/6/3). Endgame/opening are
   *reported* per-phase, not force-balanced in the main mix.
2. Verified-lucid trace channel: generate traces on a phase-oversampled
   sample (endgame positions are ~2.3% of the corpus; to get enough verified
   endgame traces, sample ~30% endgame for the *trace* channel only — the
   trace channel is 40% of data → endgame ends up ~12% of total, still
   dominated by middlegame but with enough endgame signal to measure).
   Expect lower endgame trace yield (deepseek is weakest there + tightest
   filter) — the audit of yield per phase is itself a paper result.
3. Baselines: never merge runs/models; re-measure deepseek on all 4 subsets
   with the same protocol if the committed numbers for tactic/both/noexplain
   are missing (README shows them "ran" — numbers to be confirmed from the
   monitor archive).

## 7. The decisive finding (2026-08-12): the four formats are one position pool

Cross-format FEN overlap in the TRAIN corpus:

| pair | overlap |
|---|---|
| strategy ∩ noexplain | ~100% (1,365,488 of 1,364,797 strategy FENs) |
| tactic ∩ noexplain | 100% (all 349,568) |
| both ∩ noexplain | 100% (all 349,619) |
| union (noexp ∪ strat ∪ tactic) | 1,419,501 = exactly noexplain's unique count |

**The MATE train corpus is ~1.42M unique positions; "strategy", "noexplain",
"tactic", "both" are the same positions with different explanation text.**
tactic/both are a 350k-position subset of the pool that received tactic
annotations.

Cross-format FEN overlap in the EVAL sets:

| pair | overlap |
|---|---|
| tactic ∩ both | **1000 positions** (identical sets) |
| strategy ∩ noexplain | 30 |
| strategy ∩ tactic | 11 |
| noexplain ∩ tactic | 7 |

**The tactic and both testbeds are the same 1000 positions with different
prompt text** — format-invariance is directly rewarded by the eval.

**Consequence (final, 2026-08-12):** the formats being one position pool
means **format coverage is a sampling concern, not a saturation concern** —
the SFT mix draws 25% of rows from each format pool (every prompt style
covered at ~450-550k unique positions) and does NOT train (position, format)
pairs: the model receives no format signal at eval, and retraining positions
in all four dresses wastes capacity without adding chess information. Full
3.48M-row training is MATE-scale overkill; our claim is beating that recipe
at less compute with verified lucid traces + RLVR.
