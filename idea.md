# Research Idea (v5 — diagnose, then fix cross-benchmark generalization)

## Core Research Question

Does GRPO fine-tuning (AlphaMaze's recipe) improve Gemma 4 E2B's spatial/maze navigation
reasoning, does that improvement transfer to a structurally different benchmark than it was
trained on, and — if it doesn't fully transfer — can a technique designed specifically to target
cross-format generalization close the gap better than naive mixed-format training does?

This is not just a measurement paper. The diagnostic phase (does it generalize, and specifically
how does it fail) directly motivates and shapes the proposed technique (a cross-format-consistency
reward for GRPO). That's the headline contribution.

## Full Pipeline

1. **Baseline** — untrained Gemma 4 E2B on GridRoute and Lost in Aggregation. Built, runnable now.
2. **Train single-format** — SFT + GRPO on GridRoute only (AlphaMaze's recipe, via Unsloth).
3. **Transfer eval** — single-format-trained model tested on Lost in Aggregation (OOD).
4. **Critical, multi-angle failure analysis** of the transfer failures — not just pass/fail rates.
   Multiple independent lenses: representation/format confusion (e.g. wall/path semantic
   differences between benchmarks), planning-vs-execution error localization (does it misparse
   the problem vs. plan badly vs. lose track partway through a valid start), and cross-model
   comparison (are failure modes Gemma-specific or shared with Qwen2.5/DeepSeek-distill). This
   phase is the centerpiece — do it thoroughly, it's what motivates step 6.
5. **Train mixed-format** — SFT + GRPO on GridRoute + Lost in Aggregation combined. Naive fix
   baseline, mirrors a finding from classical (non-LLM) RL navigation literature that diverse
   training data fixes cross-environment transfer. Evaluate: does naive mixing fully close what
   step 4 found, or only partially?
6. **Cross-format-consistency reward** — the proposed technique. Design informed directly by
   step 4's specific findings (whatever failure mode dominates is what this targets). Likely
   mechanism: during GRPO, occasionally present the same underlying problem in both benchmark
   formats and reward consistent, correct answers across representations, not just per-format
   correctness.
7. **Final comparison** — consistency-reward model vs. single-format model vs. mixed-format
   model, across both benchmarks. This table is the paper's central result.

## Why This, Specifically

- AlphaMaze (arXiv:2502.14669): proved the SFT+GRPO recipe works, but only for one model
  (DeepSeek-R1-Distill-Qwen-1.5B, not Gemma), one format, never tested transfer.
- Ji et al. (arXiv:2507.13362): GRPO beats SFT for spatial-reasoning OOD generalization, but on a
  vision-language Gemma variant, and only for reworded-query OOD, not structurally different
  benchmarks.
- Xi et al. (arXiv:2603.12011): RL fine-tuning generalizes well within an environment, poorly to
  genuinely unseen ones, for LLM agents generally — not tested for spatial navigation.
- "Beyond Specialization" (arXiv:2605.02528): diverse training data fixes cross-environment
  transfer for classical RL navigation policies — not language models, not tested for LLMs.
- Nobody has combined all of: text-only on-device SLM + Gemma 4 + true cross-benchmark transfer
  + a proposed technique (not just measurement) that targets the generalization gap directly.

## Status

- Multilingual angle dropped, preserved under `archive/multilingual-direction/`.
- Feasibility: Gemma 4 E2B GRPO needs ~9GB VRAM via Unsloth (confirmed via their docs); plain
  transformers+peft cannot attach LoRA to Gemma 4 at all (Gemma4ClippableLinear isn't a
  recognized peft target module type) — confirmed by direct test on our hardware. Switched to
  Unsloth as the primary backend.
- Baseline evaluation harness (`train.py`) built, covers steps 1 and 3/5/7's eval side.
- Steps 2, 5, 6 (actual training) not yet built — next, via Unsloth's Gemma 4 GRPO guide.
- Models: Gemma 4 E2B (primary), DeepSeek-R1-Distill-Qwen-1.5B (AlphaMaze's own base model,
  doubles as a replication sanity check), Qwen2.5-1.5B/3B (generalization-of-recipe check).

## Novelty

7/10 (updated from 6/10 — see `novelty-assessment.md`). The addition of a proposed, motivated
technique (not just a diagnostic test of an existing pattern) is a genuinely stronger claim than
the earlier pure-measurement framing. Still needs a final targeted novelty spot-check on the
cross-format-consistency reward specifically before committing full training time to it.

## Target Venue

Efficient and On-Device AI Agents Workshop @ NeurIPS 2026. Deadline August 29, 2026.
