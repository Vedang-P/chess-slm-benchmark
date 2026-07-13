# Novelty Assessment (v4 — diagnose, then fix cross-benchmark generalization)

## Candidate Idea

GRPO fine-tune Gemma 4 E2B for spatial/maze navigation (AlphaMaze's recipe, new model), test
whether the improvement generalizes across structurally different benchmarks (GridRoute →
Lost in Aggregation), run a rigorous multi-angle failure analysis on the transfer gap, then
propose and test a cross-format-consistency GRPO reward -- motivated directly by that analysis --
against a naive mixed-format-training baseline.

## Update From v3 (6/10 -> 7/10)

The earlier version was fundamentally a measurement paper: does a known pattern (GRPO generalizes
better than SFT) hold in a new setting. This version adds a genuinely proposed technique (the
consistency reward), motivated by original diagnostic work (the failure analysis), not just a
test of an existing idea. That's a stronger claim, hence the higher score -- but see the open
item below before treating 7/10 as final.

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

## Novelty Score: 6/10 (revised back down from 7/10 -- see below)

"Genuine novelty in application/insight," not "genuine novelty in method." The diagnostic half
(does GRPO-trained spatial reasoning transfer cross-benchmark, and why) is solid for the reasons
already established -- the general pattern is known elsewhere, this specific setting isn't. The
technique half (Step 6) is real but more modest than first assessed: it's an *application* of an
existing technique to a new domain, not a new technique.

## Step 6 Novelty Check Result (resolved)

**Elhady et al. (arXiv:2606.01464, May 2026)** already do the core mechanism proposed for Step 6:
an unsupervised RL reward for producing the same correct answer when the same problem is posed in
different *representations*, requiring no gold labels for every representation -- tested on
cross-*lingual* math reasoning (MGSM), with real gains (up to 21.7%). This is the same underlying
idea as our cross-format-consistency reward, just for language variation instead of structural-
format variation, and math instead of spatial navigation. The honest framing for Step 6 is
therefore: **testing whether a very recently validated technique (consistency-reward RL),
demonstrated in exactly one domain (cross-lingual math), transfers to a structurally different one
(cross-format spatial navigation)** -- not inventing a new technique.

Two other "consistency-aware GRPO" papers looked close on title alone but checked out as
non-overlapping: GRPO-CARE (arXiv:2506.16141) and Faithful GRPO (arXiv:2604.08476) both define
"consistency" as coherence between a model's own reasoning trace and its final answer *within one
response* -- a different, unrelated notion from cross-representation agreement. Cite as related
work, not competitors.

## Recommendation

Proceed with the full pipeline. Frame Step 6 honestly in any writeup as an application/transfer
test of Elhady et al.'s technique to a new domain, not as a novel method -- claiming otherwise
would be an easy, embarrassing catch for a reviewer who knows that paper.
