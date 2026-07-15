# Best On-Device SLM for GridRoute + MazeBench

**Target Venue**: Efficient and On-Device AI Agents Workshop @ NeurIPS 2026
**Deadline**: August 29, 2026 (AoE)
**Hardware**: RTX 4050 laptop GPU (6 GB VRAM) for dev/debugging, Kaggle T4 (16 GB, free tier) for
Gemma 4 training -- see `notebooks/kaggle_train.ipynb`
**Primary model**: Gemma 4 E2B (E4B secondary, if it fits -- see `check_finetune_feasibility.py`)

See `idea.md` / `refined-idea.md` for the full research idea and `experiment-plan.md` for the
detailed protocol and timeline. This file is the repo map + quick start + running log.

## The Goal

No small language model (<8B params) has been tested on GridRoute anywhere in the literature found
so far -- the benchmark's own paper and everything citing it test only >=7B models. Separately,
AlphaMaze (arXiv:2502.14669) proved SFT+GRPO takes a 1.5B model to 93% on its own token-format
maze benchmark, but never tried a different model or a natural-language format. This project:
extend AlphaMaze's recipe to Gemma 4, get it as good as possible at GridRoute (5x5 primary, NL
format) without regressing MazeBench (token format), and report real numbers for a setting nobody
has measured before. No specific proposed technique is the headline claim -- see
`novelty-assessment.md` for the honest framing (~5/10: a first-of-its-kind result plus a
demonstrably more rigorous harness than ad hoc alternatives, not a new algorithm).

## Status (2026-07-15)

**Phase 1 (baselines) has not produced a trustworthy number yet.** A repo-wide cleanup pass on
2026-07-15 found and fixed real bugs in the eval harness (vacuous-truth path validity, MazeBench
scored by exact-match instead of AlphaMaze's real geometric-simulation-based metric, six
duplicate/inconsistent eval scripts) -- **every numeric result below or in `results/` from before
today is historical, not trustworthy.** Phase 2 (actual training) hasn't started -- the code
(`train_sft.py`, `train_grpo.py`, `notebooks/kaggle_train.ipynb`) is ready, but this session had no
GPU access to run it (Bash/tool execution here runs on a Mac, not the training hardware).

### Historical numbers (untrusted, kept for the record -- see idea.md's Changelog)

| Run | Benchmark | Result | Note |
|---|---|---|---|
| 2026-07-14, three separate passes | MazeBench 5x5 | 99% / 70% / 88% (paper: 93%) | different sampling/token-budget settings each time, not a controlled comparison |
| 2026-07-15 | MazeBench 5x5 | 0% | same checkpoint, yet another run -- contradicts the 88% above |
| 2026-07-14/15, various | GridRoute, several models | 0% across the board | at least one run used a path-extraction function with a confirmed bug |

Do not cite any of the above. Re-run via `eval.py` and use those numbers instead.

## Repo Map

```
eval.py               Eval entry point: MazeBench (real data + AlphaMaze's real scoring code via
                       the submodule) + GridRoute (NL or our own token-maze format).
train_sft.py           SFT warm-start: --format {nl, token, mixed}.
train_grpo.py          GRPO training: --condition {single, mixed, consistency} -- recipes to try,
                       not a hypothesis test (see module docstring).
check_finetune_feasibility.py   VRAM feasibility check, including Gemma 4 E2B/E4B.
hf_models.py           Model loading (bitsandbytes 4-bit + peft for most models, Unsloth for
                       Gemma 4 via load_trainable_model()), the canonical NL path parser, and
                       extract_reported_answer() -- the clean-final-answer-only scoring policy.
src/grid_generator.py GridRoute task generation + the shared NL "FINAL ANSWER:" prompt convention.
src/token_maze.py     Our own AlphaMaze-vocabulary token-maze encoder/decoder.
src/astar_solver.py   A* solver, used for SFT ground-truth paths.
src/evaluation.py     Path validity/optimality scoring, shared by eval.py and the GRPO rewards.
alphamaze_reference/  Git submodule (github.com/menloresearch/visual-thinker) -- AlphaMaze's real
                       inference/benchmark code, used directly by eval.py for faithful MazeBench
                       scoring rather than a reconstruction.
notebooks/kaggle_train.ipynb   Kaggle-runnable end-to-end training notebook.
paper/                LaTeX writeup (NeurIPS workshop template).
data/lost_in_aggregation/, data/gridroute/   Downloaded/generated task data (Lost in Aggregation
                       isn't used in the current plan, kept for a possible future robustness check).
archive/               Earlier abandoned research directions, kept for methodology-section
                       reference (see project memory / idea.md's Changelog for what's where).
```

## Quick Start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
git submodule update --init   # pulls in alphamaze_reference/

# Check GPU feasibility for the candidate models, including Gemma 4 E2B/E4B
python check_finetune_feasibility.py

# Phase 1: baselines. AlphaMaze should land near 93% on MazeBench (real scoring code) --
# this is the harness sanity check.
python eval.py --model alphamaze --benchmark mazebench --n 100
python eval.py --model gemma4-e2b --benchmark mazebench --n 100
python eval.py --model gemma4-e2b --benchmark gridroute-nl --grid_size 5 --n 50

# Phase 2: SFT warm-start, then GRPO, single-format condition, on Gemma 4 E2B
python train_sft.py --model gemma4-e2b --format nl --grid_size 5 --n_tasks 400
python train_grpo.py --model gemma4-e2b --condition single --grid_size 5 --n_tasks 50 --max_steps 100

# Eval the resulting checkpoint on both benchmarks
python eval.py --model gemma4-e2b --checkpoint ./results/grpo_gemma4-e2b_single --benchmark mazebench --n 100
python eval.py --model gemma4-e2b --checkpoint ./results/grpo_gemma4-e2b_single --benchmark gridroute-nl --grid_size 5 --n 50
```

For Kaggle (free T4, no local GPU needed): open `notebooks/kaggle_train.ipynb`, which runs this
same pipeline end to end and saves results as downloadable JSON.

## Project Log

### 2026-07-15 -- Scope pivot: pure performance push, Gemma 4 revived, faithful replication
- **Pivoted the paper's framing**: dropped the cross-format-consistency-reward technique as the
  headline claim (kept it as one of three GRPO recipes to try, `train_grpo.py --condition
  consistency`, reported honestly rather than as a proposed method). New framing: first SLM-scale
  results on GridRoute at all, best-effort recipe on Gemma 4, evaluated jointly with MazeBench.
- **Revived Gemma 4 as primary**: Unsloth's Gemma 4 support appears to have matured past the
  transformers-version conflict found on 2026-07-13 (verify on real hardware before fully
  trusting). Added `hf_models.load_trainable_model()`: routes Gemma 4 through Unsloth, everything
  else through bitsandbytes+peft. Un-archived and rebuilt from `train.py`/`train_lora_sft.py`
  (archived the previous session) into the current `train_sft.py`/`train_grpo.py` conventions,
  then removed the now-fully-superseded archived copies.
- **Fixed the broken `alphamaze_reference` submodule**: it was a dangling gitlink with no
  `.gitmodules` entry and an empty local directory. Properly set up as a real submodule
  (`github.com/menloresearch/visual-thinker`). This caught a real bug: MazeBench's actual scoring
  (their `benchmark_maze_solution`) simulates candidate moves against the maze's real wall
  structure and accepts any path reaching the target -- `eval.py` was previously doing exact-match
  against a stored reference solution instead, which would incorrectly fail a valid-but-different
  path. `eval.py` now imports and uses their real scoring code directly when the submodule is
  present. Their MazeBench system prompt was independently confirmed verbatim-identical to what
  this project already had.
- **Added the clean-final-answer-only scoring policy**: `hf_models.extract_reported_answer()` --
  never mine coordinates/moves out of a model's thinking trace; if it never reports a clean
  answer, or got cut off before doing so, that's scored as a failure. Fixed a real bug where
  `eval.py`'s GridRoute path silently fell back to scanning the entire raw text (thinking included)
  when its "FINAL ANSWER:" marker wasn't found. Applied to `train_grpo.py`'s reward function too,
  since a reward that mines thinking tokens would train toward the sloppy behavior being avoided.
- **Raised thinking-budget defaults**: `eval.py` floors to 8192 tokens (MazeBench/token-format) /
  4096 (GridRoute NL); `train_grpo.py`'s `max_completion_length` default raised from 512 to 4096.
- **Unified the NL answer-format convention**: SFT, GRPO, and eval all now use the same
  `GRIDROUTE_NL_ANSWER_SUFFIX` ("FINAL ANSWER:" instruction) -- previously SFT taught a different
  phrasing than GRPO/eval expected, meaning the model was never actually trained on the convention
  it got scored against.

### 2026-07-15 -- First cleanup pass (superseded by the above where it conflicts)
Fixed a vacuous-truth bug in path-validity checks, collapsed six duplicate eval scripts into one
`eval.py`, collapsed training scripts, removed an accidentally-committed Unsloth compilation cache,
built the first version of the Kaggle notebook and paper skeleton. Full detail in git history
around this date.

### July 14, 2026
AlphaMaze-v0.2-1.5B downloaded and replicated (numbers since found untrustworthy, see Status
above). Feasibility confirmed for DeepSeek 1.5B / SmolLM2 1.7B on 6GB. Literature review: GridRoute
paper tested 7B-72B models only; best published number GPT-4 at 84% FR.

## Timeline

See `experiment-plan.md` for the full week-by-week table.
