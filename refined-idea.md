# Refined Research Idea (v8): Anti-Goal Instruction Following in Small Language Models — A Paired Win/Lose Chess Benchmark with Exact Oracles

Last synced: 2026-07-31. Supersedes the v7 refined idea (maze format study — retained in git
history as the maze track). Target venue: Efficient and On-Device AI Agents Workshop @ NeurIPS
2026 (ODI, Sydney), deadline Aug 29 2026 (AoE). Inference-only study; training half is
secondary and budget-gated.

## Problem Statement

Small language models (SLMs, 1.5-4B params) are increasingly deployed as on-device agents that
must follow user instructions — including instructions that run against the model's default
behavior ("do NOT optimize for X", "fail on purpose", safety-critical negation). Text-domain
studies (Inverse IFEval 2509.04292; reversed-performance personas 2504.06460) show even
frontier models resist counterfactual instructions, but their compliance is soft to verify and
they never test small models. Games — specifically chess — make the question crisp: the rules
are unambiguous (legality is machine-checkable) and the objective is unambiguous (game value is
computable on small boards, since 3x3/5x5 minichess is exactly solvable, 1307.7118). We ask: do
SLMs follow a *lose* instruction when we can verify every aspect of compliance — and can
training teach them to?

## Proposed Approach

**Benchmark (inference-only, the paper's core).** Build a paired win/lose benchmark over a
fixed, seeded position set:

- **Boards.** 3x3 and 5x5 minichess (simplified piece sets; K/R/N/B/Q/P as needed; pawns
  single-step, promote-to-queen; no castling/en-passant — documented rule variant). 8x8
  standard chess **puzzle tasks only** (no full games: SLM skill floor makes full 8x8 play
  noise — LLM Chess 2512.01992 evidence).
- **Tasks.**
  1. *Full-game single move*: given a position, output one move. Two conditions per position:
     WIN ("choose the move most likely to win") vs LOSE ("choose a legal move that does NOT
     lead toward checkmating your opponent"). ~100 positions per board size.
  2. *Mate-in-1* (8x8): WIN = deliver checkmate; LOSE = avoid checkmate while staying legal.
     Exact via checkmate detection.
  3. *Max-opponent-mobility* (8x8): LOSE-variant = choose the move giving the opponent the most
     legal replies; WIN-variant = fewest. Exactly computable (no search needed).
  4. *Play-to-lose game rollout* (3x3/5x5, optional): 2-4 ply self-consistent rollouts scored
     by minimax game value.
- **Oracle.** Our own ~150-line rule engine for 3x3/5x5 (legal moves, check, checkmate,
  minimax with full depth — positions are small); `python-chess` for 8x8 legality, checkmate
  detection, and move enumeration.
- **Metrics (all external, no model self-judgment — Collins 2510.10930).**
  - *Legality rate*: fraction of outputs that parse to a legal move.
  - *Anti-goal compliance*: for LOSE conditions, fraction where the chosen move is NOT the
    oracle-optimal win move AND (for full-game) reduces the position's minimax value relative
    to the position's value; reported against three baselines: uniform-random, oracle-random-
    legal, and the model's own WIN-condition move.
  - *Divergence*: probability that LOSE-condition move differs from WIN-condition move on the
    same position (the within-model control — this is what makes "deliberate" measurable
    holding skill constant).
  - *Win/lose outcome gap*: for rollouts, actual game outcome under WIN vs LOSE instruction.
- **Models (default set, all 4-bit on Kaggle T4):** gemma4-e2b, gemma4-e4b, deepseek-r1-
  distill-qwen-1.5b, smollm2-1.7b, qwen2.5-1.5b, qwen2.5-3b. Gemma requires an HF access token
  (Kaggle secret HF_TOKEN); if unavailable, drop to the four ungated models.
- **Prompts.** Faithful, format-explained instructions per task: board rendering (coordinates
  + piece list), explicit rules summary, output format (SAN or coordinate move, single line),
  and the objective stated twice ("your goal is to LOSE"). Same template for WIN and LOSE
  conditions, differing only in the objective sentence (paired-control discipline).

**Training half (secondary, budget-gated — only if baselines show a clear failure and Kaggle
quota allows).** SFT on ~500-1000 (position, lose-move) pairs with legality-filtered targets,
then GRPO with an exact anti-goal reward (minimax value delta + legality penalty), following
the AlphaMaze recipe arc (2502.14669) with verifiable composite rewards (2509.15557). Success
criterion: anti-goal compliance improves while legality rate does not collapse (monitor move
diversity per 2607.19523).

## What is Novel

- First paired win/lose benchmark where "deliberately losing" is **exactly verifiable**
  (solved minichess oracles + exact checkmate/mobility) — no oracle ambiguity, unlike any
  text-domain anti-goal study.
- First anti-goal evaluation at **SLM scale** (all prior anti-goal/counterfactual work tested
  frontier models; all prior SLM game work tested the win direction).
- The **legality-vs-objective decomposition**: rule-following and objective-following become
  two independent numbers on a board; text domains cannot separate them.
- (If training half runs) first recipe that trains a model to **comply with a lose objective
  without breaking legality** — a probe with direct relevance to safety-relevant negation
  instructions on-device.

## Key Assumptions

1. SLMs can produce legal-ish moves on 3x3/5x5 minichess at a measurable rate (supported by
   2410.02426: small models learn latent rules; verify in check-mode pilot before full runs).
2. Minimax oracles on 3x3/5x5 with our simplified rules are exact and fast enough for
   per-position scoring at our n (100 positions/board — trivial; solved variants exist).
3. The win/lose divergence metric is sensitive at SLM scale (i.e., models play measurably
   differently under WIN vs LOSE instructions — if divergence is ~0, the finding is "models
   ignore the anti-goal", which is itself the paper's result, not a dead end).
4. 4-bit inference on all six models on a T4 fits the Kaggle session budget (~6-10h total
   inference; no training in the base plan).
5. Gemma 4 gated access available via HF_TOKEN; else the 4-model subset suffices for the core
   claim.

## Evaluation Plan

- **Baselines:** uniform-random legal moves; oracle-random; model's own WIN-condition move
  (within-model). Compare legality rate, compliance, divergence across models, board sizes,
  and task types (mate-1, max-mobility, single-move, rollout).
- **Sanity checks:** (a) rule-engine unit tests + self-play legality fuzz in check mode;
  (b) oracle validation — for a subset of positions, verify minimax values against exhaustive
  search; (c) repeat 10 positions × 3 seeds for variance (brittleness caution, 2605.17565).
- **Reporting:** per-model × per-task × per-condition tables; failure taxonomy (illegal move,
  no parse, parse-but-wrong-piece, compliance failure vs legality failure); main table is the
  paired WIN/LOSE divergence + compliance columns.
- **Contextual anchors for the paper:** compare against LLM Chess legality/completion numbers,
  Inverse IFEval / reversed-persona failure rates (the text-domain phenomenon), and
  spec-gaming behavior (2502.13295).

## Risks

- **Skill floor:** SLMs may produce mostly illegal moves even on 3x3 — the check-mode pilot
  (n=5/position-set) decides whether 3x3/5x5 remain, or whether the paper leans on the
  mobility/mate tasks where legality demand is minimal. Mitigation: task family spans
  difficulty by design.
- **Boring finding:** "SLMs can't lose deliberately" collapses into "SLMs can't play" — this
  is exactly what the WIN-condition control and divergence metric are designed to prevent;
  if even WIN-condition play is at floor, the paper reports the legality/understanding
  results as the primary contribution (ChessQA-style).
- **Oracle implementation bugs:** simplified-rule engine must be fuzz-tested against
  python-chess where rules overlap (5x5 subset).
- **Scope creep:** the training half is the first cut if anything slips; the benchmark paper
  stands alone at 4 pages.
- **Deadline (Aug 29):** hard cap — revamp+suite by Aug 10, check run Aug 11-12, full run
  Aug 13-17, paper Aug 18-27. Any slippage past Aug 17 in data collection = cut the rollout
  task, not the paper.

## Next Actions

1. Repo revamp: `src/benchmarks/games/` (rule engine, oracles, position generator, tasks),
   `scripts/run_suite.py` + `scripts/run_eval.py`, `configs/` (models, prompts, suite),
   restored `notebooks/build_notebook.py` → `kaggle_check.ipynb` + `kaggle_run.ipynb`
   (inference-only), `legacy/` for the training scripts.
2. Check-mode Kaggle run (n=5): rule-engine fuzz, oracle validation, legality-rate pilot on
   the 4 ungated models → decides board sizes and task mix.
3. Full Kaggle run: 6 models × (3x3, 5x5, 8x8-puzzles) × paired WIN/LOSE.
4. (Budget-gated) SFT+GRPO anti-goal training on the best-scoring board/model.
5. Paper: 4-page short paper, results + failure taxonomy + (if run) training result.
