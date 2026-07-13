# Research Idea (v3 — cross-lingual spatial navigation)

## Status
Supersedes v2 (representation-behavior gap steering), which was set aside not because it was
wrong, but because the user judged probing+steering to be an overused technique in current LLM
research ("the same bs LLMs always propose") and wanted something that felt more genuinely
distinct. v1 (coordinate-extraction + A*) was superseded earlier for being near-tautological.

## Core Research Question

Does spatial *navigation* reasoning (path-finding, obstacle avoidance, sequential movement — not
static spatial question-answering) degrade differently across languages in on-device-scale SLMs
(1-4B), and if so:
1. Does the size of the cross-lingual gap correlate quantitatively with each language's
   pretraining-data availability (a hypothesis the closest related paper raises qualitatively but
   never tests numerically)?
2. Does a cheap, training-free mitigation — prompting the model to translate the instruction to
   English internally before solving — close any of the gap, and does that depend on model
   architecture (as a related non-spatial study found)?

## Why This Direction, Not the Others Considered

Two other candidates were scoped and rejected/deferred at this stage:
- Representation-behavior gap + activation steering — technically sound (see git history of this
  file / `.rstack/lit-review.jsonl` ids arxiv-2604.10690, arxiv-2502.16690, arxiv-2605.29247,
  arxiv-2603.18353) but judged too close in spirit to a saturated interpretability-paper template.
- Memorization vs. genuine generalization (procedurally-novel instances) and quantization's effect
  on spatial reasoning specifically were both confirmed as open gaps too, and remain good fallback
  directions if this one doesn't pan out.

Cross-lingual spatial reasoning was chosen because MazeEval (arXiv:2507.20395) found a real,
striking effect (models solve mazes 3-4 sizes smaller in Icelandic than English) that nobody has
followed up on systematically, and — critically — this is a *navigation* task, which turns out to
still be an open combination even after finding the closest comparator (MentalMap, below).

## Closest Related Work — Must Differentiate Explicitly

**MentalMap** (arXiv:2605.28277, May 2026) is a large, recent, comprehensive multilingual spatial
reasoning benchmark (8 languages, 13 models, ~47K items/model) asking almost the same top-level
question ("do LLM spatial world models transfer across languages?"). Verified in detail:
- Its task is static-scene spatial *question-answering* (viewpoint/frame-of-reference reasoning
  about ProcTHOR household scenes) — not path-navigation/route-finding.
- Its models are mostly 7B+; only 2 small models appear as boundary "scale controls," not the
  focus.
- It notices a language-family clustering effect and speculates ("consistent with training-data
  script coverage rather than typology") but does not quantitatively test the correlation.
- It proposes no mitigation technique.

This idea's differentiation is therefore: (a) navigation tasks specifically, reusing/extending
GridRoute, Lost in Aggregation, and MazeEval rather than scene-QA; (b) on-device SLMs (1-4B) as the
primary subject, not a boundary case; (c) a quantitative training-data-availability correlation;
(d) a concrete, tested mitigation (translate-first), informed by "Left Behind" (arXiv:2603.21036)
showing that mitigation's effect is architecture-dependent (+2.2-4.3pp for bilingual architectures,
~0 for English-dominant ones) — so the experiment should expect and test for heterogeneity across
models, not assume a uniform fix.

## Compute

Bash/tool access in this session runs on a Mac (M1) — cannot execute anything on the user's actual
RTX 4050 laptop directly. Confirmed available: a working, authenticated Modal account (used for
cloud GPU runs going forward), plus multiple Kaggle accounts for additional/parallel compute if
needed. The RTX 4050 + Ollama setup remains available to the user locally for their own
interactive use, but experiment code in this repo should target Modal by default.

## Target Venue

Efficient and On-Device AI Agents Workshop @ NeurIPS 2026. Deadline August 29, 2026. See
`docs/workshop_info.md`.
