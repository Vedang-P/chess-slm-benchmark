# Novelty Assessment (v7 -- pure performance push, no headline technique)

## Candidate Claim

Build the best on-device SLM (Gemma 4, primarily) at GridRoute natural-language grid navigation,
evaluated jointly with MazeBench (AlphaMaze's token-format benchmark), using AlphaMaze's SFT+GRPO
recipe extended to a new model and format. No proposed technique is the headline contribution.

## Change From the Consistency-Reward-Technique Version

The version this replaces (see `idea.md`'s Changelog) scored 6/10 on the strength of a genuinely
proposed technique (cross-format-consistency GRPO, adapting Elhady et al.) as its central claim.
Dropping that changes the shape of the novelty argument from "genuine novelty in
application/insight" (a technique-transfer claim) to something closer to "first result of its
kind, done rigorously" -- a real but more modest basis for a workshop paper. Scored down
accordingly (see below), not because anything got worse, but because the claim itself is smaller.

## Closest Comparators

| Paper | Relationship | Why |
|---|---|---|
| **GridRoute (arXiv:2505.24306)** | **the benchmark itself** | Tests GPT-4 Turbo, Qwen2.5-7B/72B only. No model under 7B appears anywhere in the paper or in anything found citing it. This is the core gap this project fills. |
| **AlphaMaze (arXiv:2502.14669)** | **direct methodological precedent** | Proved SFT+GRPO takes DeepSeek-R1-Distill-Qwen-1.5B from ~0% to 93% on MazeBench's token format. Never tested a different model family (Gemma), never tested a natural-language format, never tested GridRoute. |
| **Ji et al. (arXiv:2507.13362)** | background, not a direct comparator now | GRPO > SFT for spatial OOD generalization, vision-language PaLI-Gemma2-3B, reworded-query OOD -- relevant context for why GRPO is a reasonable recipe choice, not a competing claim about this project's specific setting. |
| **Elhady et al. (arXiv:2606.01464)** | **source of one candidate training recipe, not the headline** | Consistency-reward RL for cross-lingual math (MGSM). The `consistency` GRPO condition adapts this mechanism to cross-format spatial reasoning as one recipe tried among several -- report results honestly, don't frame as this paper's contribution. |
| Unsloth Gemma 4 documentation | feasibility precedent | Confirms GRPO/LoRA on Gemma 4 is a solved engineering problem via their loader -- necessary infrastructure for this project, not itself a research contribution. |

## What Is Novel

- **First SLM-scale (<8B) results on GridRoute, period.** Not "the best" or "a new approach to" --
  simply the first. This is a real, unsubtle literature gap, not a marginal framing choice.
- **First attempt to extend AlphaMaze's specific SFT+GRPO recipe to a different model family
  (Gemma 4) and a different surface format (natural language).**
- **A harness built specifically to avoid measurement artifacts this project's own history
  demonstrates are real** -- a single model's MazeBench score swung 99%/70%/88%/0% across
  successive versions of this project's own eval code (temperature, token-budget, and parsing
  differences), and a real scoring discrepancy was caught between exact-match and AlphaMaze's
  actual geometric-simulation-based metric. Worth stating as a secondary methodological point: SLM
  spatial-reasoning numbers reported without this level of care are not trustworthy, and this
  project's numbers are produced with it (clean-final-answer-only policy, generous and verified
  thinking budgets, official scoring code reused directly via a submodule rather than
  reconstructed).

## What's Not Novel (Be Honest About This In The Paper)

"GRPO improves small-model spatial reasoning" is already demonstrated (AlphaMaze). "Consistency-
reward RL helps cross-representation generalization" is already demonstrated (Elhady et al., for
language, not format). Neither is this paper's claim. If the consistency condition is included in
results, frame it explicitly as: "we also tried adapting a technique from [Elhady et al.] as one
recipe; here is whether it helped in this setting" -- not as a proposed method.

## Novelty Score: ~5/10

Lower than the previous (6/10, technique-centered) version, honestly -- this is now a "did the
literature simply never test this, and did we do it carefully" claim rather than an
application/insight claim about a specific technique. That's still a legitimate, publishable
workshop contribution (the venue's own call for papers explicitly welcomes "benchmarks and
evaluation for on-device agents" and "training under constraints" as topics, not only new
algorithms) -- but it should be pitched as exactly that: a rigorous first measurement and a
best-effort training recipe, not a novel-method paper. If a reviewer's likely pushback is "this is
just running an existing recipe on a new benchmark," the honest answer is: yes, and nobody had,
and here's the evidence that doing so carelessly (as this project's own history shows) gives
wildly wrong numbers.

## What Would Raise This Score

A genuinely surprising empirical finding (e.g., Gemma 4 transfers unusually well/poorly compared
to AlphaMaze's recipe, or the consistency condition has a large, clean effect) would strengthen
the paper considerably -- but that's a result to report if it happens, not something to assume or
engineer toward before the data exists.
