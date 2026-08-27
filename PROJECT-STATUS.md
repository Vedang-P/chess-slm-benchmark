# Project Status — Chess SLM Benchmark (2026-08-27)

Current working status of the project. Keep this updated as the direction
changes (it has changed twice).

## The objective (unchanged)
Make a small language model play chess better than DeepSeek V4 Flash via
natural-language reasoning, no engine at inference. Target: Efficient and
On-Device AI Agents Workshop @ NeurIPS 2026. Compute: Kaggle free tier only
(P100, ~30h/week/account, 2 accounts).

## Direction history (why we pivoted twice)

### 1. RLVR / GRPO (ABANDONED — measured infeasible)
Plan: GRPO on gemma-4-E2B-it (2B) with Stockfish rewards, uncapped thinking.
- v9 (capped 256 tok): trained 2 steps but every rollout clipped, reward 0 —
  the model never emits EOS on the MoveA/B prompt; `MoveMoveMove` repetition.
- v10: `_ThinkingProcessor` not callable (fixed).
- v11 (G8 uncapped): 2h notebook timeout hit; 0 rollouts completed.
- v12 (G2 uncapped): step 0 never completed in 3h+ (watchdog ineffective vs
  CUDA-blocked generate). Killed.
- Live-thinking infrastructure (token stream -> stdout + HF live-thinking.txt
  every 15s) was built and works (demo-2 verified: real reasoning, terminates
  with `MoveA:...<turn|>`, 4695 tokens/163s on an easy position).
- Verdict: uncapped thinking on the base 2B model never terminates on P100.
  RLVR at this compute is infeasible. User decision: pivot.

### 2. EGSD / SIL (PROPOSED, REJECTED/PARKED)
- EGSD (engine-graded self-distillation): user rejected as "caveman SFT with
  a filter + a loop" — fair critique. `scripts/egsd_sample.py` + plan doc
  exist but are NOT the active direction.
- Search-in-Language (SIL): verbalized engine search trees + self-consistency
  voting. 5000 traces built (`results/searchlang-traces.jsonl` + _sft).
  User moved on before this was run. Parked; traces remain useful data.

### 3. Improve searchless_chess (ACTIVE)
Open-source chess models from DeepMind (Ruoss et al., NeurIPS 2024):
9M/136M/270M transformers trained searchless on ChessBench (10M games, 15.3B
Stockfish-16 action-values). 270M reaches 2895 Lichess Elo vs humans (GM).

**The MATE transfer result (LOCAL, verified 2026-08-27):**
- 9M (9M params!) scores **92%** on MATE noexplain-1000 (50-row local smoke)
  vs gemma-4-E2B base **58.1%**. Chess specialists crush generalists at
  expert 2-choice judgment.
- Full 3-model x 4-test-set sweep running locally (CPU).

**Improvement plan (committed: `improve-searchless-plan.md`):**
DPO self-play pipeline (precedent: dbest-isi/searchless-chess-9M-dpo, +25
Elo from 1000 games). Target: close the 136M->270M gap (+40 Elo) at half the
params — "GM-level at 1/2 the parameters", evaluated on MATE.

## Key files
- `improve-searchless-plan.md` — the active plan (DPO self-play on 136M/270M)
- `searchlang-plan.md` — parked SIL plan
- `egsd-plan.md` — parked EGSD plan
- `scripts/eval_searchless_mate.py` — MATE eval for searchless models
- `scripts/build_search_traces.py` — SIL trace builder (5000 done)
- `scripts/egsd_sample.py` — EGSD sampler (parked)
- `scripts/train_mate_grpo.py` — GRPO trainer (abandoned, keep for infra:
  live traces, HF checkpoints, _ThinkingProcessor)
- `results/searchlang-traces.jsonl` (+ _sft) — 5000 verbalized search traces

## External resources (searchless_chess)
- Paper: https://arxiv.org/abs/2402.04494 (Ruoss et al., NeurIPS 2024)
- Repo (open, Apache-2.0): https://github.com/google-deepmind/searchless_chess
- Weights: storage.googleapis.com/searchless_chess/checkpoints/{9M,136M,270M}.zip
- Dataset ChessBench: data/download.sh (10M games, 15.3B labels)
- DPO precedent: https://huggingface.co/dbest-isi/searchless-chess-9M-dpo
- MAV successor (DeepMind, weights NOT released): https://arxiv.org/abs/2412.12119

## Environment gotchas (measured)
- The 2024 checkpoints are orbax-ocdbt format: need jax 0.4.35 + orbax 0.5.5
  (era stack) — modern jax can't read them ("No structure could be
  identified" / `PositionalSharding` removed in jax 0.5).
- Kaggle preinstalls jax 0.11: era stack install collides (ResolutionImpossible).
  Local venv (/tmp/slvenv) with the era stack WORKS — prefer local CPU eval
  over Kaggle for inference; Kaggle needed only for DPO training later.
- Local Mac: no JAX until venv; CPU inference at ~0.85s/fwd (270M arch).

## Current activity
- Full MATE sweep (3 models x 4 test sets) running locally.
- Next: interpret sweep -> pick target model -> DPO pilot (G2 gate: +Elo vs
  base on MATE) -> full DPO iterations -> paper.
