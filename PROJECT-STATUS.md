# Project Status — Chess SLM Benchmark (2026-08-30)

Current working status. Keep updated as the direction changes.

## The objective
Find the best searchless chess action-value model below 9M parameters, with a
primary target of matching or exceeding Ruoss 9M on MATE and official puzzles,
then test whether the same method can approach the 136M/270M accuracy frontier.
Target: Efficient and On-Device AI Agents Workshop @ NeurIPS 2026. Compute:
Kaggle free tier only (T4/P100, ~30h/week/account, 3 accounts). Honesty
protocol: the exact frozen MATE sets and official 10K puzzle protocol are never
used for training or model selection.

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

**Measured result (exact MATE subsets, local CPU, official ActionValueEngine):**
- 9M: **98.2%** (982/1000) — complete, full 1000 rows
- 9M: **98.9%** on tactic, both, and strategy/full (989/1000 each)
- 136M: **99.4%** on all four subsets (3976/4000 combined)
- 270M: **99.4%**, **99.4%**, **99.4%**, **99.5%** on noexplain, tactic, both, strategy/full (3977/4000 combined)
- 9M official puzzle harness: **86.13%** (8613/10000 full solution sequences)
- gemma-4-E2B base baseline: 58.1%


## Key files
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
1. Repair the controlled 5M student: fix the double-log-softmax bug, use the
   real training distribution rather than a test-bag derivative, and add full
   HF-resumable state.
2. Implement the 3–6M square-token Geometric Action-Value Network (GAVN),
   with chess relation bias, source/destination action factorization, and
   distribution + scalar-Q + ranking objectives.
3. Run a frozen-protocol ablation matrix across the three Kaggle accounts.
4. Select the Pareto frontier by held-out MATE accuracy, 10K puzzle accuracy,
   parameters, FLOPs, latency, calibration, and error overlap.
5. Convert the result into a reproducible A* workshop paper with uncertainty
   estimates and negative-result documentation.
