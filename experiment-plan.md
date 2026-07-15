# Experiment Plan -- Best On-Device SLM for GridRoute + MazeBench

## Phasing

**Phase 1 -- baselines, using the real official codebase where one exists.**
`eval.py --model alphamaze --benchmark mazebench --n 100` runs AlphaMaze's own public checkpoint
through their own real scoring code (via the `alphamaze_reference` submodule --
`github.com/menloresearch/visual-thinker`), not a reconstruction. **Every number from before
2026-07-15 in this repo is untrusted** -- the old harness had a vacuous-truth bug in path-validity
checking, used exact-match scoring for MazeBench when the real metric is geometric-simulation-
based (any valid path counts, not just one stored reference), and swung 99%/70%/88%/0% across
successive runs that changed more than one variable at a time. Re-run clean with the current
`eval.py` before reporting any number.

Phase 1 also means: Gemma 4 E2B, Gemma 4 E4B (if `check_finetune_feasibility.py` confirms it fits),
and AlphaMaze-v0.2-1.5B, all untrained/off-the-shelf, on GridRoute NL 5x5 and MazeBench. AlphaMaze
landing near its published 93% on MazeBench (with the real scoring code) is the harness sanity
check; Gemma 4's numbers on both benchmarks are the actual open question, since **no SLM has been
tested on GridRoute anywhere in the literature found so far**.

**Phase 2 -- improve Gemma 4 as much as possible**, detailed below. Not a hypothesis-testing
pipeline -- try recipes, report whichever works, including honest negative results.

## Benchmarks

- **MazeBench-v0.2** (`Menlo/Maze-Bench-v0.2`, loaded directly): token format, 100 mazes. Scored
  with AlphaMaze's own real `benchmark_maze_solution`/`extract_answer` (imported directly from the
  submodule in `eval.py`) when available -- simulates candidate moves against the maze's actual
  wall structure, accepts any sequence reaching the target. Falls back to exact-match against the
  `Response` field, with a loud warning, only if the submodule isn't initialized
  (`git submodule update --init`).
- **GridRoute** (`src/grid_generator.py`): NL format, **5x5 primary** (matches MazeBench's size and
  an SLM-appropriate difficulty -- 10x10 is optional, later, only if 5x5 goes well and time
  permits; the codebase supports 10x10/20x20/30x30 if wanted).
- **Our own token-maze encoding** (`src/token_maze.py`) of GridRoute grids: training-time
  mechanism for the mixed/consistency GRPO conditions (pairing the same underlying grid across
  both formats). Can also be reported as a robustness check via `eval.py --benchmark
  gridroute-token`, not a primary result.

## Models

- **Gemma 4 E2B** -- primary. LoRA/GRPO via plain bitsandbytes+peft (`hf_models.load_trainable_model`),
  the same path every other model uses. `Gemma4ClippableLinear` needs `peft>=0.19.0` to be a
  recognized target module type (pinned in requirements.txt) -- older peft genuinely can't attach
  LoRA to it, which is why this project routed through Unsloth for a while before confirming
  peft's native support and dropping Unsloth entirely (see `hf_models.load_trainable_model()`'s
  docstring). E2B LoRA measured ~8-10GB elsewhere -- fits Kaggle's free 16GB T4, not a 6GB laptop.
- **Gemma 4 E4B** -- secondary, if feasible. ~17GB LoRA footprint reported elsewhere, at/over the
  T4's 16GB ceiling -- `check_finetune_feasibility.py` must confirm this empirically before
  committing a full training run to it.
- **AlphaMaze-v0.2-1.5B** -- the MazeBench-side reference point (already ~93% there by
  construction). Also a candidate to continue-train on GridRoute, same bitsandbytes+peft path.
- **DeepSeek-R1-Distill-Qwen-1.5B, SmolLM2-1.7B, Qwen2.5-3B** -- cheap secondary comparisons via
  bitsandbytes+peft, not primary.

## Methodology Requirements (apply to every run)

- **Clean-final-answer-only scoring.** `hf_models.extract_reported_answer` -- never mine
  coordinates/moves out of a thinking trace the model didn't itself present as its answer. Applies
  to both `eval.py` and `train_grpo.py`'s reward function (a reward that mines thinking tokens
  would train toward exactly the behavior we don't want).
- **Generous thinking budget.** `eval.py` floors: 8192 tokens for MazeBench/token-format, 4096 for
  GridRoute NL. `train_grpo.py`'s `max_completion_length` defaults to 4096 (was 512 -- found to
  truncate MazeBench-style reasoning in Phase 1's original debugging). This is the single biggest
  lever on wall-clock cost -- always run a short timing test first (see Immediate Next Step).
- **Verify thinking-delimiter assumptions against real output.** `hf_models.THINKING_END_MARKERS`
  includes markers whose exact correctness for Gemma 4 specifically is NOT independently confirmed
  (multiple inconsistent descriptions found in web search) -- check the raw output field of your
  first real Gemma 4 eval run and adjust if the model's actual delimiter differs. AlphaMaze/
  DeepSeek-distill's `</think>` convention IS confirmed directly against their real code.
- **Faithful replication over reconstruction, wherever an official implementation exists.** Use
  `alphamaze_reference`'s real code for anything MazeBench-related rather than re-deriving it.

## Training Protocol

`train_sft.py --format {nl,token,mixed}` then `train_grpo.py --condition {single,mixed,
consistency}` -- three recipes to try, not three conditions to prove/disprove a hypothesis about:

- **single**: SFT + GRPO on GridRoute NL only.
- **mixed**: SFT + GRPO on GridRoute NL + our token-maze encoding, interleaved over the same
  underlying grids, naive per-format reward.
- **consistency**: same mixed data, reward includes a bonus for cross-format agreement on the same
  underlying problem (mechanism adapted from Elhady et al., arXiv:2606.01464 -- one candidate
  recipe, not the paper's headline). Roughly doubles per-reward-call compute (generates a partner
  completion to score against) -- budget accordingly.

Seed 42 throughout.

## Baselines

- Untrained Gemma 4 E2B/E4B, untrained AlphaMaze, untrained DeepSeek/SmolLM2/Qwen2.5, all on both
  benchmarks (`eval.py`, runnable now).
- Each training condition is itself a baseline for evaluating whether the next one is worth its
  additional compute (e.g. if `single` already gets Gemma 4 doing well on both benchmarks, `mixed`
  and `consistency` may not be worth running).

## Evaluation Metrics

- MazeBench: AlphaMaze's own simulation-based accuracy (see Benchmarks above).
- GridRoute: valid-path rate, optimal-path rate (`src/evaluation.py`, fixed for the vacuous-truth
  bug -- requires `len(path) >= 2`).
- Central table: every training condition x both benchmarks. The point is best joint performance,
  not an isolated before/after on one axis.
- Secondary: failure-category breakdown (`src/evaluation.py`'s existing taxonomy: no_output,
  start_end_mismatch, obstacle_collision, out_of_bounds, invalid_step).

## Ablations (time permitting, in priority order)

1. SFT-only vs. SFT+GRPO, single condition -- cheap, confirms the RL stage's contribution over
   the warm-start alone.
2. Cross-model check (DeepSeek-R1-Distill-Qwen-1.5B, SmolLM2-1.7B) -- is any effect Gemma-specific.
3. Grid-size scaling (10x10) for whichever recipe wins, if time permits.

## Timeline (against the Aug 29, 2026 AoE deadline; today is Jul 15)

| Dates | Milestone |
|---|---|
| Jul 15-17 | Harness fixed; `check_finetune_feasibility.py` run on real hardware for Gemma 4 E2B/E4B; clean Phase 1 baselines for Gemma 4 E2B/E4B + AlphaMaze on both benchmarks |
| Jul 17-22 | SFT + GRPO single condition on Gemma 4 E2B (Kaggle) |
| Jul 22-26 | SFT + GRPO mixed condition |
| Jul 26-30 | SFT + GRPO consistency condition -- timing test first (this one's cost is least predictable) |
| Jul 30-Aug 2 | E4B attempt if quota allows; cross-model check; final comparison table |
| Aug 2-8 | Buffer for reruns/debugging |
| Aug 8-20 | Paper writing with real results |
| Aug 20-27 | Revisions, formatting, polish |
| Aug 29 | Submit to OpenReview (AoE) |

## Immediate Next Step

1. Run `check_finetune_feasibility.py` on the actual training hardware (Kaggle T4) -- confirms
   whether Gemma 4 E2B/E4B LoRA actually fits, on this exact hardware, not an estimate from
   elsewhere.
2. Run Phase 1 baselines (`eval.py`, all models x both benchmarks). Confirm AlphaMaze lands near
   93% on MazeBench using the real scoring code -- if it doesn't, the harness has a problem to fix
   before trusting anything else.
3. Get a real GRPO timing number (`train_grpo.py --condition single --max_steps 20`) on Gemma 4
   E2B before committing to a full step count.

## Not Yet Run (built, not executed -- no GPU access in this session)

Everything above the "Immediate Next Step" line still needs to actually run on real hardware. See
`notebooks/kaggle_train.ipynb` for the Kaggle-runnable version of this whole pipeline.
