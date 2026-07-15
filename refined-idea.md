# Refined Research Idea (v7 -- pure performance push: best on-device SLM for GridRoute + MazeBench)

Novelty score: see `novelty-assessment.md` (~5/10 -- honest framing: this is a first-of-its-kind
empirical result plus a methodologically rigorous harness, not a new algorithm).

**Last synced with `idea.md`, `novelty-assessment.md`, `experiment-plan.md`, `README.md`:
2026-07-15.** If reading this later, check `git log --oneline -- idea.md refined-idea.md
novelty-assessment.md experiment-plan.md README.md` -- these describe one plan and must agree;
this has drifted out of sync once already (see idea.md's Changelog).

## Problem Statement

No small language model (<8B parameters, plausibly on-device-feasible) has been tested on
GridRoute -- every published result (GridRoute's own paper, everything citing it) tests only
>=7B models. Separately, AlphaMaze proved SFT+GRPO takes a 1.5B model to 93% on its own
token-format maze benchmark, but never tried a different model family or a natural-language
surface format. This project asks a direct, practical question: how good can we actually make an
on-device SLM (Gemma 4, primarily) at GridRoute, and can we do that without wrecking MazeBench
performance if starting from or jointly training with AlphaMaze's recipe? No specific proposed
technique is the point -- the point is a real number, honestly measured, for a setting nobody has
reported before, using the best training recipe we can find.

## Approach

1. **Baseline** (current phase): Gemma 4 E2B, Gemma 4 E4B (if it fits, see
   `check_finetune_feasibility.py`), and AlphaMaze-v0.2-1.5B, all untrained/off-the-shelf, on
   GridRoute NL 5x5 and MazeBench. AlphaMaze should land near its published 93% on MazeBench --
   this is the harness sanity check (Phase 1), using their real scoring code via the
   `alphamaze_reference` submodule, not a reconstruction.
2. **SFT warm-start** (`train_sft.py --format {nl, token, mixed}`): teach the reporting format
   ("FINAL ANSWER:" for NL, plain move tokens after thinking for token-format) and basic
   task competence before GRPO.
3. **GRPO** (`train_grpo.py --condition {single, mixed, consistency}`): three recipes to try, not
   three hypotheses to test scientifically --
   - `single`: GridRoute NL only.
   - `mixed`: NL + our own token-maze encoding (`src/token_maze.py`) interleaved, naive per-format
     reward.
   - `consistency`: same mixed data, reward includes a bonus when the model's own completions on
     the NL and token versions of the same underlying problem agree on validity/optimality
     (mechanism adapted from Elhady et al., arXiv:2606.01464 -- cross-lingual consistency-reward
     RL, applied here to cross-format spatial reasoning as one candidate recipe, not the paper's
     headline claim).
4. **Report whichever recipe wins**, on both benchmarks, including honest negative results if
   nothing beats the simplest baseline. Cross-model check (DeepSeek-R1-Distill-Qwen-1.5B,
   SmolLM2-1.7B) as a cheap secondary comparison, time permitting.

## What Is (Modestly) Novel

- First SLM-scale (<8B) results on GridRoute at all -- the literature gap here is not subtle, it's
  a total absence.
- First attempt to extend AlphaMaze's SFT+GRPO recipe to Gemma 4 and to a natural-language
  surface format, evaluated jointly with the original token-format benchmark.
- A demonstrably more rigorous evaluation harness than ad hoc alternatives: this project's own
  history shows a single model's MazeBench score swinging 99%/70%/88%/0% across successive harness
  versions (temperature/token-budget/parsing bugs), and a real discrepancy caught between
  exact-match scoring and AlphaMaze's actual geometric-simulation-based scoring. Worth stating
  explicitly in the paper as a secondary contribution: careless harness design measurably
  distorts apparent SLM spatial-reasoning ability, and this project's harness (clean-final-answer
  policy, generous thinking budgets, faithful reuse of official scoring code) is built specifically
  to avoid that.

## What's Not Claimed

No new training technique, no new benchmark, no theoretical contribution. The consistency-reward
GRPO condition is Elhady et al.'s mechanism, applied to a new domain as one candidate recipe among
several -- report it as "we also tried this, here's what happened," not as an invention.

## Key Assumptions

1. Gemma 4 E2B/E4B can actually be LoRA/GRPO fine-tuned via Unsloth without the transformers
   version conflict noted in this project's 2026-07-13 history -- current evidence (Unsloth's own
   docs advertise Gemma 4 support) suggests this is resolved, but verify directly on the actual
   training hardware before committing real compute, don't just trust a web search summary.
2. Gemma 4 E4B's LoRA footprint (~17GB reported elsewhere) may not fit Kaggle's free 16GB T4 --
   `check_finetune_feasibility.py` needs to confirm this empirically; treat E4B as optional until
   it does.
3. Our own token-maze format (not a byte-identical clone of AlphaMaze's undocumented exact
   grammar) is close enough in vocabulary to interact meaningfully with any token-format skill a
   model already has, for the mixed/consistency conditions specifically.
4. Gemma 4's actual thinking-block delimiter is unconfirmed (multiple inconsistent descriptions
   found; see `hf_models.THINKING_END_MARKERS`'s docstring) -- verify against real raw output from
   the first actual Gemma 4 generation before trusting any score that depends on
   `extract_reported_answer` correctly finding the end of its thinking.

## Evaluation Plan

- **Models**: Gemma 4 E2B (primary), Gemma 4 E4B (if feasible), AlphaMaze-v0.2-1.5B (MazeBench
  reference point). DeepSeek-R1-Distill-Qwen-1.5B / SmolLM2-1.7B / Qwen2.5-3B as cheap secondary
  comparisons.
- **Benchmarks**: MazeBench-v0.2 (real data, real scoring code via submodule). GridRoute NL, 5x5
  primary (10x10 optional, later, if time permits).
- **Metrics**: MazeBench: their own simulation-based accuracy. GridRoute: valid-path rate,
  optimal-path rate (`src/evaluation.py`, fixed for the vacuous-truth bug). Report both benchmarks
  for every training condition -- the central table is "best joint performance," not a
  before/after on one axis.
- **Baseline**: untrained Gemma 4 (both sizes), untrained AlphaMaze, all four DeepSeek/SmolLM2/
  Qwen2.5 secondary models, all runnable now via `eval.py`.

## Risks

- **Gemma 4 training may simply not converge well, or E4B may not fit at all** -- real possible
  outcomes; a clean negative result (Gemma 4 doesn't get much better at GridRoute no matter the
  recipe) is still a valid, honestly-reportable finding given nobody has this baseline yet either.
- **The consistency condition roughly doubles per-step compute** (partner-completion generation
  inside the reward function) -- get a real timing number before committing to a full run.
- **Kaggle's free-tier weekly GPU quota (~30h) is a real constraint** against three training
  conditions x up to two model sizes -- prioritize E2B + single/mixed conditions first, treat E4B
  and the consistency condition as stretch goals if quota is tight.

## Next Actions

1. Run `check_finetune_feasibility.py` (now includes gemma4-e2b/e4b) on the actual training
   hardware (Kaggle T4) to get real VRAM numbers before assuming either fits.
2. Run Phase 1 baselines: `eval.py` for {gemma4-e2b, gemma4-e4b, alphamaze} x {mazebench,
   gridroute-nl}. Confirm AlphaMaze lands near 93% on MazeBench using the real scoring code --
   this validates the harness before trusting anything else.
3. Get a real GRPO timing number (`train_grpo.py --condition single --max_steps 20`) before
   committing to a full step count, especially for the consistency condition later.
4. SFT + GRPO single-format on Gemma 4 E2B; evaluate both benchmarks; decide from there whether
   mixed/consistency are worth the additional compute given what single-format alone achieves.
