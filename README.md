# Chess SLM Benchmark: Small Models vs Frontier Models at Chess

**Vedang Pandey** — Efficient and On-Device AI Agents Workshop @ NeurIPS 2026

## Objective

Make a small language model play chess better than DeepSeek V4 Flash via
reasoning — no external search, no engine at inference time. Chess is the
probe: state, legal actions, and oracles are explicit, making it a clean
testbed for reasoning in small models.

Compute constraint: Kaggle free tier only (P100, ~30h/week/account, 2
accounts). Everything in this repo is designed to fit that budget.

Honesty protocol: evaluation always uses the exact noexplain-1000 test set
(same 1000 positions used to score gemma base = 58.1% and DeepSeek V4 Flash).
No subsets, no cherry-picking.

---

## What has been tried (and what we learned)

### 1. RLVR / GRPO on gemma-4-E2B-it (ABANDONED — measured infeasible)

Plan: GRPO with Stockfish rewards, thinking ON, on a 2B model.

**Results (all measured, all dead ends):**
- **v9** (256-token cap): trained 2 steps, but every rollout clipped,
  reward 0 — the model never emits EOS on the MoveA/B prompt, producing
  `MoveMoveMove` repetition instead of an answer.
- **v10**: `_ThinkingProcessor` wrapper not callable (fixed, but moot).
- **v11** (G8, uncapped): 2h notebook timeout; 0 rollouts completed.
- **v12** (G2, uncapped): step 0 never completed in 3h+; watchdog SIGINT
  ineffective against CUDA-blocked generation. Killed.
- **Live-thinking demo** (the one working piece): streamed real gemma
  reasoning token-by-token to stdout + HF (`live-thinking.txt`, 15s
  cadence). Demo-2 verified: real reasoning, terminates with
  `MoveA:...<turn|>`, 4695 tokens / 163s on an easy position.

**Verdict:** uncapped thinking on the base 2B model never terminates on
P100. RLVR at this compute is infeasible. **Infrastructure built here is
reusable**: HF checkpoint persistence (adapter + optimizer + scheduler +
trainer_state), live trace streaming, `_ThinkingProcessor`, era-pinned
stack knowledge.

### 2. EGSD / SIL (PROPOSED, PARKED)

- **EGSD** (engine-graded self-distillation): iterative SFT-only loop
  (sample → Stockfish grade → SFT on accepted). User rejected as "caveman
  SFT with a filter + a loop" — a fair critique.
- **SIL** (Search-in-Language): SFT on verbalized engine search trees +
  self-consistency voting at inference. 5000 verbalized traces built
  (`results/searchlang-traces.jsonl` + `_sft.jsonl`) — reusable data.

### 3. Improve open searchless_chess models (ACTIVE)

Open-source DeepMind chess models (Ruoss et al., NeurIPS 2024):
9M/136M/270M transformers, searchless, trained on ChessBench (10M games,
15.3B Stockfish-16 action-values). The 270M reaches 2895 Lichess Elo vs
humans (grandmaster level).

**The master table — all models × all 4 MATE subsets (exact 1000s).**
Baselines archived from the 2026-08-19 campaign (git); searchless models
measured locally with the official ActionValueEngine (2026-08-27).

| model | params | noexplain | tactic | both | full | status |
|---|---|---|---|---|---|---|
| deepseek-v4-flash (thinking, unbounded) | frontier | **85.8%** | **92.2%** | **94.0%** | **92.8%** | ✅ complete |
| gemma4-e2b (thinking, 32768) | 2B | **58.1%** | **60.5%** | **61.5%** | **60.8%** | ✅ complete |
| gemma4-e2b caveman-SFT | 2B | **55.4%** | — | — | — | ✅ complete (regressed) |
| **searchless 9M** (Ruoss) | 9M | **98.2%** | TBD | TBD | TBD | 🔵 noexplain done |
| **searchless 136M** (Ruoss) | 136M | TBD | TBD | TBD | TBD | ⬜ pending |
| **searchless 270M** (Ruoss) | 270M | TBD | TBD | TBD | TBD | ⬜ pending |
| **searchless 136M-DPO** (ours) | 136M | TBD | TBD | TBD | TBD | ⬜ pending |
| **searchless 270M-DPO** (ours) | 270M | TBD | TBD | TBD | TBD | ⬜ pending |

Reference points from the MATE paper (their fine-tuned LLaMA-3-8B
zero-shot): 63.5% (N), 89.7% (S), 94.6% (T), 95.2% (ST).

**The result so far:** a 9M-param searchless chess specialist (98.2%)
beats the 2B generalist gemma (58.1%) AND the frontier model DeepSeek V4
Flash (85.8%) at expert 2-choice chess judgment — at 1/300th of gemma's
params, no engine at inference. The transfer question is answered:
ChessBench-trained action-values transfer to MATE-style expert tasks.

**Next:** score 136M + 270M on all 4 exact subsets (expect ≥98%), then
improve via DPO self-play and score the DPO variants (see
`improve-searchless-plan.md`).

---

## The plan: improve searchless_chess (see `improve-searchless-plan.md`)

DPO self-play pipeline on 136M/270M:
1. **Self-play**: current model plays games vs itself.
2. **Mistake ID**: Stockfish (d20-24) finds plies where the model's move
   ≠ engine best with |Δeval| > 0.3 pawns → (chosen=SF, rejected=model)
   preference pairs. Precedent: `dbest-isi/searchless-chess-9M-dpo`
   (+25 Elo, +1% puzzles from 1000 games / 36k pairs / 50 steps).
3. **DPO train**: JAX/Haiku, lr 1e-5, beta 0.1, EMA 0.999.
4. **Eval**: exact noexplain-1000 + head-to-head BayesElo.
5. **Target**: close the 136M→270M gap (+40 Elo) at half the params —
   "GM-level at 1/2 parameters".

## External resources

- Paper: Ruoss et al., arXiv:2402.04494 (NeurIPS 2024)
- Repo (open, Apache-2.0): github.com/google-deepmind/searchless_chess
- Weights: storage.googleapis.com/searchless_chess/checkpoints/{9M,136M,270M}.zip
- Dataset (ChessBench): data/download.sh in the repo
- DPO precedent: huggingface.co/dbest-isi/searchless-chess-9M-dpo
- MAV (DeepMind successor, weights NOT released): arXiv:2412.12119

## Repo layout

```
scripts/
  eval_searchless_mate.py   ACTIVE — MATE 2-choice eval for searchless models
  build_search_traces.py    ACTIVE — verbalized search traces (5000 built)
  train_mate_lora.py        SFT trainer (reusable)
  run_mate_eval.py          eval protocol (gemma/DeepSeek baseline scoring)
src/                        eval/SFT support (models, mate_metrics, report)
data/positions/             the 4 MATE eval sets (noexplain/both/tactic/full)
results/                    searchlang traces + rlvr-pool (reusable data)
improve-searchless-plan.md  ACTIVE plan
PROJECT-STATUS.md           current status
engineering-decisions.md    decision log
```

## Environment gotchas (measured)

- The 2024 searchless checkpoints are orbax-ocdbt format: need
  **jax 0.4.35 + orbax 0.5.5** era stack (modern jax can't read them;
  `PositionalSharding` removed in jax 0.5). Kaggle preinstalls jax 0.11 →
  era pins hit `ResolutionImpossible` there; **run inference locally in a
  venv with the era stack, or on Kaggle only for DPO training later**.
- Local CPU speed: 9M ~0.28s/position, 270M ~0.85s/fwd (jit-less).
- Run sweeps foreground/streaming — background jobs with pipes can look
  hung (low CPU).

## Status

See `PROJECT-STATUS.md` for the current snapshot. Short version: 9M scored
98.2% on the exact noexplain-1000 (vs gemma 58.1%); 136M/270M eval + DPO
improvement are next, pending compute allocation.
