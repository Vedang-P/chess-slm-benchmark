# Research Idea (v7 -- pure performance push: best on-device SLM for GridRoute + MazeBench)

**See `refined-idea.md` for the full write-up. This file is the shorter working version plus history/changelog.**

## Core Goal

Build the best on-device SLM at BOTH GridRoute (natural-language grid navigation, 5x5 primary)
and MazeBench (AlphaMaze's token-format maze-solving), starting from Gemma 4 (E2B primary, E4B
if it fits) and AlphaMaze-v0.2-1.5B as the MazeBench-side reference point. No proposed novel
technique is the headline claim -- this is an engineering + rigorous-benchmarking contribution,
not a hypothesis-testing paper (see Changelog: this is a deliberate pivot from an earlier
consistency-reward-technique framing).

## Why This Is Still a Real Contribution

To the best of available literature search, **no SLM (<8B params) has been tested on GridRoute
at all** -- the GridRoute paper and everything found citing it only test >=7B models (GPT-4,
Qwen2.5-7B/72B). Separately, AlphaMaze proved SFT+GRPO takes a 1.5B model to 93% on its own
token-format benchmark, but never tested a different model family (Gemma) or a different surface
format (NL) at all. Combining these: first SLM-scale GridRoute results, first attempt to extend
AlphaMaze's recipe to Gemma 4, evaluated on both formats jointly. See `novelty-assessment.md` for
the honest score (this is more "first to test this specific setting, plus a genuinely rigorous
harness" than "novel method" -- say so plainly in the paper).

## Models

**Primary**: Gemma 4 E2B. LoRA/GRPO via Unsloth (`hf_models.load_trainable_model` --
Gemma4ClippableLinear isn't a recognized peft target module type, confirmed on this project's own
hardware; Unsloth's loader patches around it). E2B LoRA measured ~8-10GB elsewhere -- fits
Kaggle's free 16GB T4, not a 6GB laptop.
**Secondary**: Gemma 4 E4B, if `check_finetune_feasibility.py` confirms it actually fits (E4B LoRA
reported ~17GB elsewhere -- at/over the T4's ceiling, unconfirmed until measured on the real
hardware).
**Reference point for MazeBench**: AlphaMaze-v0.2-1.5B (already at ~93% there by construction --
the paper's own number, this is what Gemma 4 needs to approach/beat on that side, or at least not
regress on if we continue training AlphaMaze's own checkpoint on GridRoute too).
**Also runnable** (bitsandbytes+peft, no Unsloth needed): DeepSeek-R1-Distill-Qwen-1.5B,
SmolLM2-1.7B, Qwen2.5-3B -- kept in scope as cheap secondary comparisons, not primary.

## Grid Size

**5x5 GridRoute first** (matches MazeBench's 5x5, and an SLM-appropriate difficulty level --
10x10+ is plausibly too hard for this model class before 5x5 works well). 10x10 is optional,
later, only if 5x5 goes well and time permits.

## Methodology (the part that needs real care)

- **Clean-final-answer-only scoring, never mine the thinking trace.** A model's own reported
  answer is whatever it says after its thinking closes (`</think>`, or whatever marker the
  specific model family actually uses -- verify empirically, don't assume) or after a "FINAL
  ANSWER:" instruction we gave it. If it never reports one, or got cut off before reporting one,
  that's a failure -- not something to recover by scanning raw/thinking text for stray-but-
  plausible coordinates. Implemented once, centrally, in `hf_models.extract_reported_answer`, used
  by both `eval.py` and `train_grpo.py`'s reward function (a reward function that mines thinking
  tokens would directly train the model toward the sloppy behavior we're trying to avoid).
- **Generous thinking budget, don't truncate mid-thought.** `eval.py` floors max_new_tokens to
  8192 for token-format/MazeBench, 4096 for GridRoute NL; `train_grpo.py`'s max_completion_length
  defaults to 4096 (up from an earlier 512, which Phase 1 already found truncates MazeBench-style
  reasoning). This is the single biggest lever on wall-clock training cost -- run the timing test
  (`experiment-plan.md`'s Immediate Next Step) before committing to a full run.
- **Faithful replication via the real codebase, not a reconstruction.** `alphamaze_reference/` is
  now a proper git submodule (`github.com/menloresearch/visual-thinker`) -- `eval.py` imports
  their actual `benchmark_maze_solution`/`extract_answer` directly for MazeBench scoring. This
  caught a real, substantive discrepancy: their real scoring simulates candidate moves against the
  maze's actual wall structure and accepts ANY sequence reaching the target, not exact-match
  against one stored reference solution -- an earlier version of this project's `eval_mazebench`
  did exact-match, which would incorrectly fail a valid-but-different path. Their MazeBench SYS
  prompt was independently confirmed verbatim-identical to what this project already had.

## Phasing

**Phase 1** (current): baselines. Gemma 4 E2B, Gemma 4 E4B (if feasible), AlphaMaze-v0.2-1.5B on
both GridRoute 5x5 (NL) and MazeBench (token, real data). AlphaMaze should reproduce ~93% on
MazeBench by construction (it's already trained for exactly that) -- this is the harness
sanity-check. Gemma 4's numbers on both benchmarks are the real open question. Every pre-2026-07-15
number in this repo is untrusted (see Changelog) -- get clean numbers with the current, fixed
`eval.py` before anything else.

**Phase 2**: improve Gemma 4 as much as possible via SFT (`train_sft.py --format {nl,token,mixed}`)
and GRPO (`train_grpo.py --condition {single,mixed,consistency}`) -- these three conditions are
kept as engineering options to try, not a scientific hypothesis test (the consistency condition is
one candidate recipe among several, adapting Elhady et al. 2606.01464's cross-lingual consistency
reward to cross-format spatial reasoning -- try it, report honestly whether it helps, don't
present it as the point of the paper). Report whichever recipe actually produces the best joint
GridRoute+MazeBench performance, including negative results if nothing beats the simplest baseline.

Full detail in `experiment-plan.md`.

## Changelog (why the plan looks like this)

- **2026-07-13 to 07-14**: original idea -> cross-lingual spatial navigation (dropped) -> GRPO
  fine-tuning for Gemma 4 spatial reasoning -> expanded to "diagnose, then propose a fix" (Gemma 4
  + GridRoute + Lost in Aggregation OOD + cross-format-consistency reward as the headline
  technique, novelty 6->7->6/10).
- **2026-07-14 to 07-15**: separately, replicated AlphaMaze on MazeBench+GridRoute; this work
  updated README.md but wasn't back-ported into the other docs for a full pivot's worth of time
  (see the 07-15 cleanup entry below).
- **2026-07-15, first cleanup pass**: caught and fixed a vacuous-truth bug in path-validity
  checking, collapsed six duplicate/inconsistent eval scripts into one `eval.py`, collapsed
  training scripts, reconciled docs around a "cross-format-consistency-reward technique on
  MazeBench vs. GridRoute" framing (dropping Gemma 4/Lost-in-Aggregation), built a Kaggle notebook
  and a paper skeleton.
- **2026-07-15, second pass (this one)**: reconsidered scope with fresh input. Decided: (a) drop
  the consistency-reward technique as the paper's headline claim -- pure performance push instead,
  since no SLM has been tested on GridRoute at all, which is itself a sufficient motivation; (b)
  revive Gemma 4 as the primary model (Unsloth's Gemma 4 support has matured since the 07-13
  finding that blocked it -- verify current status before assuming either way, but current
  evidence suggests it's no longer the blocker it was); (c) 5x5 GridRoute first, 10x10 optional;
  (d) clean-final-answer-only scoring policy, generous thinking budgets, faithful replication via
  the real AlphaMaze codebase (now a proper submodule, was previously a broken/orphaned gitlink
  with no `.gitmodules` entry) -- this last change caught a real discrepancy: MazeBench's real
  scoring is geometric-simulation-based (any valid path counts), not exact-match against a stored
  reference, which an earlier version of `eval_mazebench` got wrong.
