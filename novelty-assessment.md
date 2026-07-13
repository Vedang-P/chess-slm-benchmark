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

## Novelty Score: 7/10

"Genuine novelty in method, application, or insight." The diagnostic half (does GRPO-trained
spatial reasoning transfer cross-benchmark, and why) sits at the same 6/10 level as before for
the same reasons -- the general pattern is established elsewhere, this setting isn't. The
technique half (a consistency reward motivated by original failure analysis) pushes it higher:
proposing and validating a fix, not just measuring a gap, is a stronger contribution class.

## Open Item Before Treating This As Final

The cross-format-consistency reward specifically has NOT been searched for yet -- everything
checked so far (AlphaMaze, Ji et al., Xi et al., Beyond Specialization, Unsloth's Gemma 4 docs)
was about the diagnostic half of the plan. Before committing real training time to Step 6, run a
targeted search for: reward shaping for cross-domain/cross-format consistency in RL-fine-tuned
LLMs, and specifically whether anyone has done format-consistency or representation-consistency
rewards in GRPO training generally (not necessarily spatial). This is a reasonably natural idea --
worth confirming it's not already done before building around it.

## Recommendation

Proceed with the diagnostic pipeline (Steps 1-5) now -- fully checked, no blockers. Run the
Step 6 novelty spot-check before finalizing the consistency-reward design specifically.
