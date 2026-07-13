# Refined Research Idea (v2 — cross-lingual spatial navigation)

Produced by /novelty-check on 2026-07-13. Novelty score: 6/10 (see `novelty-assessment.md`).

## Problem Statement

On-device SLMs (1-4B parameters) are increasingly deployed as navigation/agentic assistants, but
a single existing data point (MazeEval, arXiv:2507.20395) shows a striking effect: the same model
solves mazes 3-4 sizes smaller when the identical task is posed in Icelandic instead of English.
If spatial-navigation reasoning is this sensitive to input language, on-device agents built for
non-English-speaking users may be silently and substantially less capable — a real deployment
concern nobody has systematically measured. A comprehensive multilingual spatial-reasoning
benchmark exists (MentalMap, arXiv:2605.28277) but tests static scene question-answering, not
navigation, and treats small on-device models as boundary cases rather than the subject. Whether
the gap is explainable by training-data availability, and whether it's fixable cheaply, remain
open questions even after MentalMap.

## Proposed Approach

**Step 1 — Build the multilingual navigation eval set.** Take the existing GridRoute and Lost in
Aggregation task instances (grids/mazes + ground-truth optimal paths already generated in this
repo) and produce parallel NL instruction sets in 8-10 languages spanning high-resource
(Spanish, Mandarin, French, German), mid-resource (Hindi, Icelandic — matching MazeEval),
low-resource (e.g. Swahili, Nepali), and differing scripts/directionality. Critically, translate
the fixed instruction *templates* (a handful of sentence patterns, per `grid_to_nl_variants` in
`src/grid_generator.py`) once per language rather than per-instance — this holds task difficulty
constant across languages and isolates language as the only varying factor, avoiding the confound
of per-instance translation-quality noise.

**Step 2 — Measure the gap.** Run several on-device SLMs (Gemma 4 E2B; Qwen2.5-1.5B/3B for
cross-model comparison, matching AlphaMaze/Gideon/SmallPlan's reference points) across all
languages on the same navigation tasks. Primary metric: valid/optimal path rate per language,
relative to that model's own English baseline (within-model, within-task normalization — avoids
comparing absolute capability across models, isolates the language effect specifically).

**Step 3 — Test the training-data-availability hypothesis.** Using a public proxy for
per-language pretraining data volume (e.g. known corpus statistics from mC4/OSCAR/CC-100, or each
model's own reported training-data language breakdown where available), quantitatively correlate
per-language performance drop against data availability. This is the test MentalMap's own text
gestures at but never runs numerically.

**Step 4 — Test the mitigation.** For each model/language pair, compare direct-language solving
against a two-step "translate the instruction to English internally, then solve" prompt strategy.
Per Left Behind's finding (arXiv:2603.21036), expect and explicitly test for
architecture-dependent heterogeneity (some models gain, some don't) rather than reporting one
pooled effect size.

## What Is Novel

- First systematic (8-10 language) cross-lingual study of spatial *navigation* reasoning — the
  only existing navigation-specific data point (MazeEval) covers 2 languages.
- First to center on-device SLMs (1-4B) as the primary subject for this question, rather than as a
  boundary/scale-control case (MentalMap's treatment).
- First quantitative test of the training-data-availability hypothesis for spatial reasoning
  specifically — an existing paper (MentalMap) notices the pattern qualitatively but doesn't test
  it numerically.
- First test of a training-free mitigation for cross-lingual spatial reasoning specifically.

## Key Assumptions

1. Translating fixed instruction *templates* (not per-instance) into each language, by a
   competent method (see Next Actions — needs a real decision, not machine-translation-and-hope),
   produces natural, task-equivalent instructions across languages. Bad translations would
   confound "language effect" with "translation quality effect."
2. A usable proxy for per-language pretraining-data volume exists and is reasonably comparable
   across the specific models tested (models differ in training corpora; this is a real
   measurement challenge, not a given).
3. Gemma 4 E2B and the Qwen2.5 comparison models have adequate multilingual coverage to even
   attempt all chosen languages — needs a quick check before committing to the full language list.

## Evaluation Plan

- **Models:** Gemma 4 E2B (primary), Qwen2.5-1.5B and 3B (comparison/reference points).
- **Tasks:** GridRoute (full-observability grids, already in repo) and Lost in Aggregation
  (multi-scale mazes, already in repo) — navigation tasks, reusing existing ground-truth optimal
  paths for scoring. MazeEval-style partial observability as a stretch goal if time allows.
- **Languages:** 8-10 spanning resource level and script, including Icelandic (direct MazeEval
  comparison point) and at least one RTL script (e.g. Arabic, also in MentalMap for
  cross-referencing).
- **Metrics:** valid-path rate and optimality rate per language (reusing/fixing the existing
  `src/evaluation.py`, which has known bugs — hardcoded `valid_move_ratio`, `compliance_ratio`
  conflated with `feasibility_ratio` — that must be fixed before reuse), normalized against each
  model's own English baseline; correlation coefficient (e.g. Pearson/Spearman) between
  normalized performance and the data-availability proxy; effect size of the translate-first
  mitigation, reported per model (not pooled).
- **Baselines:** each model's own English performance (the natural baseline here, not an external
  method).

## Risks

- **Translation quality is the biggest methodological risk.** Machine-translating templates with
  the same SLM being tested (or a similarly imperfect translator) risks confounding "the model is
  bad at this language" with "the instruction is bad in this language." Mitigate by using a
  strong, separate translation source for the fixed templates (human review of the small number
  of unique template sentences is feasible — there are only a handful, not per-instance).
- **Data-availability proxies are noisy and inconsistently defined across corpora/models.** The
  correlation test needs honest error bars and should not overclaim precision.
- **MentalMap re-analysis risk:** if MentalMap's authors or someone else publishes the
  quantitative correlation test before this work does, that specific contribution shrinks —
  time-sensitive, per the novelty assessment's recommendation to move promptly.
- **Null result is a real possible outcome** (gap doesn't correlate with data availability, or
  mitigation doesn't help) — still publishable as a clean negative result extending MentalMap to
  navigation tasks, but changes the pitch; decide in advance this is an acceptable outcome.

## Next Actions

1. Decide the translation approach for instruction templates (human-reviewed MT, a professional
   translation API with review, or bilingual-speaker review) — this gates everything else.
2. Confirm Gemma 4 E2B and the Qwen2.5 comparison models' multilingual training coverage claims
   for the chosen language list before committing to it.
3. Select and justify the training-data-availability proxy metric.
4. Fix `src/evaluation.py`'s known bugs before reuse.
5. Write `experiment-plan.md` for Step 1-2 (build the multilingual set + measure the gap) as the
   first concrete, fundable Modal run — Steps 3-4 (correlation + mitigation) follow once Step 2's
   data exists.
