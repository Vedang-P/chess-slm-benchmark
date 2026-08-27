# Project Status — Chess SLM Benchmark (2026-08-27, evening)

Current working status. Keep updated as the direction changes.

## The objective (unchanged)
Make a small language model play chess better than DeepSeek V4 Flash via
natural-language reasoning, no engine at inference. Target: Efficient and
On-Device AI Agents Workshop @ NeurIPS 2026. Compute: Kaggle free tier only
(P100, ~30h/week/account, 2 accounts). Honesty protocol: only the exact
noexplain-1000 (same positions used for gemma 58.1% / DeepSeek) counts.

## Direction history

### 1. RLVR / GRPO (ABANDONED — measured infeasible)
GRPO on gemma-4-E2B-it with Stockfish rewards, uncapped thinking.
- v9 (256 cap): trained but all rollouts clipped, reward 0 (model never
  emits EOS on MoveA/B prompt; `MoveMoveMove` repetition).
- v11 (G8 uncapped): 2h timeout, 0 rollouts completed.
- v12 (G2 uncapped): step 0 never completed in 3h+; killed.
- Built + verified live-thinking infra (token stream -> stdout + HF
  live-thinking.txt every 15s; demo-2: real reasoning, terminates with
  `MoveA:...<turn|>`, 4695 tok/163s).
- Verdict: uncapped thinking on base 2B never terminates on P100. RLVR
  infeasible at this compute. User decision: pivot.

### 2. EGSD / SIL (PROPOSED, PARKED)
- EGSD (engine-graded self-distillation): user rejected ("caveman SFT with
  a filter + a loop"). Script + plan exist, not active.
- SIL (Search-in-Language): verbalized engine search trees + self-consistency
  voting. 5000 traces built (`results/searchlang-traces.jsonl` + _sft).
  Parked; traces remain reusable data.

### 3. Improve searchless_chess (ACTIVE DIRECTION)
Open-source DeepMind chess models (Ruoss et al., NeurIPS 2024):
9M/136M/270M transformers, searchless, trained on ChessBench (10M games,
15.3B Stockfish-16 action-values). 270M = 2895 Lichess Elo vs humans (GM).

**Measured result (exact noexplain-1000, local CPU, official ActionValueEngine):**
- 9M: **98.2%** (982/1000) — complete, full 1000 rows
- gemma-4-E2B base baseline: 58.1%
- 136M / 270M: NOT YET RUN (sweep was stopped per user order)

**Improvement plan (committed: `improve-searchless-plan.md`):**
DPO self-play (precedent dbest-isi/searchless-chess-9M-dpo: +25 Elo, +1%
puzzles from 1000 games / 36k pairs / 50 steps). Target: close the
136M->270M gap (+40 Elo) at half the params — "GM-level at 1/2 params",
evaluated on the exact noexplain-1000.

## Key files
- `improve-searchless-plan.md` — active plan (DPO self-play on 136M/270M)
- `searchlang-plan.md` — parked SIL plan
- `egsd-plan.md` — parked EGSD plan
- `scripts/eval_searchless_mate.py` — MATE eval for searchless models
- `scripts/build_search_traces.py` — SIL trace builder (5000 done)
- `scripts/egsd_sample.py` — EGSD sampler (parked)
- `scripts/train_mate_grpo.py` — GRPO trainer (abandoned; keep for infra:
  live traces, HF checkpoints, _ThinkingProcessor)
- `results/searchlang-traces.jsonl` (+ _sft) — 5000 verbalized search traces

## External resources (searchless_chess)
- Paper: https://arxiv.org/abs/2402.04494 (Ruoss et al., NeurIPS 2024)
- Repo (open, Apache-2.0): https://github.com/google-deepmind/searchless_chess
- Weights: storage.googleapis.com/searchless_chess/checkpoints/{9M,136M,270M}.zip
- Dataset ChessBench: data/download.sh (10M games, 15.3B labels)
- DPO precedent: https://huggingface.co/dbest-isi/searchless-chess-9M-dpo
- MAV successor (weights NOT released): https://arxiv.org/abs/2412.12119

## Environment gotchas (measured)
- 2024 checkpoints are orbax-ocdbt: need jax 0.4.35 + orbax 0.5.5 era stack.
  Modern jax can't read them; Kaggle preinstalls jax 0.11 (pip
  ResolutionImpossible on era pins). Local venv `/tmp/slvenv` WORKS — do
  local CPU eval, not Kaggle, for inference.
- Local CPU speed: 9M ~0.28s/position, 270M ~0.85s/fwd (jit-less).
- Background bash jobs with output pipes can appear hung (low CPU) — run
  sweeps foreground/streaming.


## Baselines (paper comparison data)
- **Gemma-4-E2B baselines — SAFE**: HF dataset `eval-results/caveman-sft-{a1,a2,b1,b2,pretest}/` (5 variants, noexplain samples+summary), restored locally to `results/baselines/`. Note: these are 250-row win-condition slices (examples), NOT full-1000 accuracy.
- **Full clean-1000 (gemma 58.1% + DeepSeek V4 Flash samples) — LOST 2026-08-27** (deleted during cleanup; never git-tracked, not on HF). MUST re-run on the exact noexplain-1000 via `scripts/run_mate_eval.py` (gemma local, DeepSeek API) and store on HF + a non-gitignored location.

## Remaining work
1. **Run 136M + 270M on the EXACT noexplain-1000** (local, ~1-1.5h) — the
   honest full table: 9M=98.2% (done), 136M=?, 270M=? vs gemma 58.1%.
2. Interpret: pick target model for improvement.
3. DPO pilot (2k self-play games, SF d20-24 mistake pairs, 50-200 steps) —
   gate: +Elo vs base on the exact noexplain-1000.
4. Full DPO iterations (2-3), eval each.
5. Frontier comparison + paper writeup.
