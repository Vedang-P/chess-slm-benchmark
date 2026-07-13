# Experiment Plan — Step 1-2: Cross-Lingual Navigation Gap Measurement

First concrete, Modal-runnable experiment for the cross-lingual spatial navigation direction
(`refined-idea.md` v2). Scoped to Step 1-2 only (build the multilingual set + measure the gap);
Step 3 (data-availability correlation) and Step 4 (translate-first mitigation) are separate,
later experiments that consume this one's output.

## Objective

Measure whether on-device SLMs solve the same GridRoute navigation tasks at different success
rates depending on the language of the instruction, holding task difficulty exactly constant
across languages (same grids, same start/goal, same obstacles — only the instruction language
varies).

## Dataset

- **Source tasks:** GridRoute size_10 tasks, already generated in `data/gridroute/` /
  reproducible via `src/grid_generator.py` (seed=42, matches existing repo convention).
- **Sample:** 100 tasks (stratified across the existing map/pair structure, not just the first
  100 — avoid any ordering bias from generation).
- **Instruction language:** GridRoute's "direct" NL variant, in English + 9 translated languages
  from `data/multilingual/gridroute_direct_template.json` (Spanish, French, Mandarin, Hindi —
  high confidence; Icelandic, Thai, Vietnamese — medium confidence; Swahili, Nepali — medium-low
  confidence; all caveats documented in that file and to be reported transparently in any writeup).
- **Total instruction instances:** 100 tasks x 10 language conditions = 1,000 per model.

## Model Architecture

No architecture changes — off-the-shelf inference only, this is an evaluation study, not
training:
- **Gemma 4 E2B** (primary, on-device story) — via HuggingFace `transformers`, standard
  precision available on Modal's A100 (bf16; no need for the aggressive local
  quantization used on the RTX 4050, since Modal isn't VRAM-constrained the same way, and we no
  longer need Ollama's thinking-tag-stripping-specific plumbing — but must confirm Gemma 4 E2B's
  `transformers` chat template behavior, including whether it still emits thinking-tag-style
  output that needs stripping, as part of code generation).
- **Qwen2.5-1.5B and Qwen2.5-3B** — comparison points already used as references by
  AlphaMaze/Gideon/SmallPlan, and Qwen has documented broad multilingual training coverage
  (relevant given the whole study depends on the models having some multilingual capability to
  even measure a gap in).

## Training Protocol (inference protocol — no training in this phase)

- Temperature 0.0 (deterministic), consistent across all conditions.
- Max generation tokens: enough for a full path on a 10x10 grid plus reasonable slack (reuse the
  ~1024-2048 token budgets already validated in the old pipeline's `src/baselines.py`).
- Seed 42 throughout.
- Same parsing/path-extraction logic across all languages — this must not silently favor English
  (e.g. don't rely on English-only regex keywords like "Path:" to find the answer — use
  coordinate-pattern extraction, which is language-agnostic, matching the more robust strategies
  already in `src/baselines.py::_extract_coords_from_text`).

## Baselines

- **Each model's own English-language performance** is the baseline every other language is
  normalized against (within-model, within-task-difficulty comparison — not comparing absolute
  capability across models).
- **Pure A\* ground truth** (already implemented, `src/astar_solver.py`) for computing true
  optimal path length per task, used for the optimality metric regardless of language.

## Evaluation Metrics

- **Valid-path rate** per (model, language): obstacle-free, in-bounds, correct start/end, unit
  steps — reusing `src/evaluation.py` logic, but only after fixing its known bugs (hardcoded
  `valid_move_ratio=0.0`, `compliance_ratio` conflated with `feasibility_ratio` via
  `max(feasible, 1)`) as part of code generation, not deferred.
- **Optimality rate** per (model, language): path length matches true A* optimal length.
- **Normalized gap**: (English rate − language rate) / English rate, per model per language — the
  primary quantity Step 3's correlation analysis will consume.
- **Token count and latency** per condition, as a secondary check that failures are genuine
  reasoning failures and not e.g. truncation differences across languages (tokenizers vary in
  efficiency per language, which could confound results if max-token budgets aren't generous
  enough for lower-resource-language tokenizations).

## Ablations

- **Cross-model comparison**: does the gap pattern (which languages are hardest) look similar
  across Gemma 4 E2B and the two Qwen2.5 sizes, or is it model-specific? Informs whether any
  eventual finding is about "SLMs generally" or one model family's training data specifically.
- **Model-scale check within Qwen2.5** (1.5B vs 3B): does the gap narrow with scale even within
  the on-device range, addressing whether this is purely a capacity effect.

## Deferred to Later Experiments

Step 3 (quantitative correlation against a training-data-availability proxy) and Step 4
(translate-first mitigation test) are separate follow-on experiments once this run's data exists.
Lost in Aggregation and MazeEval-style partial observability are deferred until the GridRoute
result is in hand and the pipeline is validated at small scale.
