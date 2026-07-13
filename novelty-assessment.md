# Novelty Assessment (v3 — GRPO fine-tuning for Gemma 4 spatial reasoning)

## Candidate Idea

GRPO fine-tune Gemma 4 E2B for spatial/maze navigation (AlphaMaze's recipe, new model), then test
whether the improvement generalizes across structurally different benchmarks (GridRoute →
Lost in Aggregation), not just within the training format.

## Closest Comparators

| Paper | Relationship | Why |
|---|---|---|
| **AlphaMaze (arXiv:2502.14669)** | **direct methodological precedent** | Same recipe (SFT+GRPO), different model (DeepSeek-R1-Distill-Qwen-1.5B, not Gemma), single format only (5x5 maze), never tests transfer to a different benchmark |
| **Ji et al. (arXiv:2507.13362)** | **closest overlap — must differentiate explicitly** | Already shows GRPO > SFT for spatial-reasoning OOD generalization, on PaLI-Gemma2-3B. Different modality (vision-language, not text-only), different OOD type (rephrased queries, not structurally different benchmarks) |
| **Xi et al. (arXiv:2603.12011)** | **predicts our likely result, not a competitor** | General LLM-agent finding: RL generalizes well within an environment, poorly to genuinely unseen ones. Not spatial-specific, not tested on maze/grid navigation |
| **Beyond Specialization (arXiv:2605.02528)** | **analogous, different literature** | Same cross-environment transfer problem, but classical DRL policy networks, no language models involved |
| Unsloth Gemma 4 GRPO guide | **feasibility precedent, not a research contributor** | Confirms GRPO-on-Gemma-4 is a solved engineering problem (~9GB VRAM), demonstrated on Sudoku — not spatial reasoning, not a research claim |
| DUPLEX, SmallPlan, Gideon | background | Neuro-symbolic / distillation approaches to SLM planning; different method family, still relevant related work |

## What Is Novel

Nobody has combined: (a) GRPO fine-tuning (b) a text-only on-device SLM (c) Gemma 4 specifically
(d) spatial/maze navigation (e) tested for true cross-benchmark structural transfer, not
same-format difficulty variation or rephrased-query OOD. Every individual piece of this has a
close precedent; the combination doesn't.

## What's Not Novel (Be Honest About This In The Paper)

"GRPO generalizes better than SFT for spatial reasoning" is already demonstrated (Ji et al.).
"You can GRPO fine-tune Gemma 4" is already documented (Unsloth). The paper's contribution is
specifically the cross-benchmark structural transfer test in this new setting, not the general
pattern.

## Novelty Score: 6/10

Real, correctly scoped, "genuine novelty in application/insight" — not higher because the general
finding this extends is already established in adjacent settings (VLM spatial OOD, general LLM
agent environment transfer); not lower because the specific combination and the practically
relevant on-device/text-only setting are genuinely untested, and either outcome (generalizes or
doesn't) is a real, citable contribution.

## Recommendation

Proceed. This is a live space (Ji et al. and Xi et al. are both from 2026, within the last few
months) — move to building rather than continuing to search.
