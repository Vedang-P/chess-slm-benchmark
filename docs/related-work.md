# Related work: chess x LLM benchmarks (2024-2026)

What exists, what each one does, which is closest to ours, and what we
borrow from each. Verdict up front:

> **Our primary base: MATE** (NAACL 2025) — fine-tuned an 8B model on 1M
> expert-annotated positions (strategy + tactic, with/without language
> explanations) and beat GPT/Claude/Gemini on move selection. We replicate
> its recipe on the LATEST models (LoRA-tuned Gemma 4 E2B/E4B vs
> deepseek-v4-flash), evaluate on its public testset plus our tactical
> battery, and tighten the evaluation (standard chess, larger n, more
> tasks). Supporting bases: **ChessQA** (task taxonomy), **LLM CHESS**
> (game-play protocol), **ChessBench** (dataset ancestry).

---

## 0. MATE — "Explore the Reasoning Capability of LLMs in the Chess
## Testbed" (Wang et al., NAACL 2025) — OUR BASE

- **What**: 1M chess positions, each with candidate moves annotated by
  chess experts (incl. women's world champion Yifan Hou) for STRATEGY and
  TACTIC, with and without language explanations (MATE-Strategy,
  MATE-Tactic, MATE-Strategy-Explanation, MATE-No-Explain, MATE-Both).
  Fine-tunes LLaMA-3-8B; compares against GPT-4o/Claude/Gemini on move
  selection; the 8B fine-tune WINS. Finding: language explanations
  measurably enhance the model's move-selection reasoning.
- **What it gets right (borrow)**: the finetune-then-compare recipe; the
  dual strategy/tactic annotation frame; explanations as a reasoning
  aid; a public dataset + testset (HF: OutFlankShu/MATE_DATASET).
- **What it gets wrong / how we improve**:
  1. **Models are dated**: LLaMA-3-8B (2024) vs 2024-era GPT/Claude/Gemini.
     We fine-tune the LATEST small VLMs (Gemma 4 E2B/E4B, 2026) via LoRA
     and compare against the latest frontier (deepseek-v4-flash).
  2. **Evaluation is thin**: a single move-selection metric against
     commercial APIs, small testset, non-standard-rules ambiguity.
     We add: mate-in-1/2 and best-move batteries under standard chess
     (python-chess), legality probes, larger n, and per-sample verdicts
     with full thinking chains archived.
  3. **No representation control**: MATE uses one prompt format. We ran a
     Phase-1 representation study (FEN wins decisively) and standardize
     on FEN, with the evidence reported.
  4. **Fine-tuning cost**: 8B full fine-tune. We use 4-bit LoRA on the
     free T4 tier — cheaper, reproducible, and the paper's "can a small
     fine-tuned model fit on-device" story. 

---

## 1. ChessBench (Keller & Hutter, 2024)

- **What**: six chess datasets for LLM probing: move classification (24M
  lichess positions), lichess eval (6.7M), mate-in-1/2/3 (~5M total),
  AlphaZero eval, and Chen-Pasukonis game-state prediction (~14M).
- **Design**: standard LLM eval — accuracy per dataset; used to argue
  LLMs can learn chess from text with enough data.
- **Borrow**: the *dataset menu* (lichess evals + lichess puzzle mates +
  game states). We rebuild the same sources with full metadata and a
  representation axis; we do **not** adopt its 100k-sample sizes — ours is
  a tight benchmark, not a training-data study.

## 2. ChessQA (Wen, Tang & Anderson, 2025)

- **What**: 5-category chess-understanding benchmark: **Structural** (board
  reading, rules), **Motifs** (tactic pattern recognition), **Short Tactics**
  (mate-in-N), **Position Judgment** (eval), **Semantic** (move-by-move
  natural-language explanations).
- **Why it matters**: the closest *conceptual* match to "rethink our tests
  entirely". It measures ability levels instead of one number, and it's
  dynamic (evolving prompts/answer keys, released code + datasets).
- **Borrow**: the **category taxonomy** — we keep mate1/mate2/bestmove as
  Short-Tactics + Position-Judgment cores and plan Structural/Motifs/Semantic
  arms (README future scope). Also its **error analysis by category** idea
  for the paper.

## 3. LLM CHESS (Kolasani et al., 2025)

- **What**: agentic game-play benchmark — 50+ open/closed models play
  full games against a random opponent; metrics: win/loss, move legality,
  move quality, hallucinated actions, game duration; Elo estimates for top
  reasoning models by playing against a fixed-strength engine.
- **Key finding**: even strong models struggle to finish games; clear
  reasoning vs non-reasoning separation.
- **Borrow**: the **game-play protocol** (fixed random opponent, legality +
  quality + Elo) for our future game-play arm, and the anti-saturation
  argument (dynamic evaluation beats static memorization-prone benchmarks).

## 4. Easy2Hard-Bench (Ding et al., NeurIPS 2024 D&B)

- **What**: 6 benchmarks incl. chess puzzles, each item annotated with a
  numerical difficulty (IRT / Glicko-2 from real human/LLM attempt data).
- **Borrow**: **difficulty stratification** — our lichess puzzle ratings
  are already Glicko-based; Easy2Hard shows how to turn them into clean
  easy→hard curves for generalization analysis (we can reuse the method,
  not the data).

## 5. KinGPT / "Generalization or Memorization?" (Tang, 2026)

- **What**: trains KinGPT, a 25M-parameter character-level model whose only
  domain is chess. It uses three training distributions: Woodpecker (13.34M
  unique Lichess puzzle positions), Beaver (54,681 positions from Stockfish
  18 self-play), and Chimera (the combination). It then tests on a theme-wide
  held-out suite of mate-in-1/2/3 puzzles rather than only reporting training-
  distribution performance.
- **Key result**: Woodpecker reaches 81.7% position accuracy, Chimera 84.6%,
  while Beaver reaches only 2.2%. A tiny model trained on puzzle-shaped data
  can therefore beat larger chess models on the same narrow puzzle suite,
  while failing to demonstrate broad chess understanding. This warns against
  interpreting a high mate score as general reasoning.
- **Inference controls**: normal pass@1, a "cheating" prompt that supplies
  the position evaluation, pass@10 sampling, and LLM-Modulo. LLM-Modulo uses
  two hard critics: one asks whether the move is legal; the other asks
  whether it improves the engine evaluation. Failed candidates are re-
  prompted with feedback. For RedPajama 3B, this raises best-move accuracy
  from 1.2% to 21.2% and validity from 19.3% to 95.3%.
- **Metrics**: position-wide accuracy, puzzle-wide accuracy (every move in a
  puzzle must be correct), and sanity (`1 - invalid parses / positions`).
  Example: a mate-in-3 puzzle can contribute three position decisions but is
  only puzzle-correct if all three are right.
- **Borrow directly**:
  1. Split by puzzle theme and remove FEN overlap between training and eval;
  2. report both position accuracy and whole-puzzle accuracy;
  3. report legality/sanity separately from correctness;
  4. add a verifier-loop ablation later, clearly separating native model
     behavior from verifier-assisted behavior;
  5. do not claim visible thinking traces are faithful explanations without
     independently validating them.

## 6. Geometric stability (Song et al., 2025)

- **What**: tests LLMs under board rotation, mirroring, color inversion,
  format conversion (~3,000 positions; GPT-5.1 accuracy collapses >600% on
  rotations; Claude/Kimi robust). "Accuracy–stability paradox."
- **Borrow**: the **perturbation methodology** — our representation axis is
  precisely a format-conversion perturbation. We can add mirror/rotation
  arms cheaply (our engine renders any board orientation).

## 7. Tracking World States (Harang et al., 2025)

- **What**: model-agnostic state-tracking eval: measure how well LLMs
  preserve legal-move distributions across predicted states (no internal
  probes needed).
- **Borrow**: the **legal-move-distribution metric** as a secondary signal
  for our legality work.

## 8. Spec-gaming in reasoning models (Bondarenko et al., 2025)

- **What**: o3 / DeepSeek R1 *hack* chess benchmarks (engine-exploits)
  when told to win. A caution for any "beat the engine" framing.
- **Borrow**: prompt hygiene — our prompts never ask for a win against an
  engine; they ask for a single move in a fixed position. Document that
  choice against spec-gaming.

## 9. Others surveyed (context only)

- **ZeroSumEval** (FAIR 2025): inter-model competition eval incl. chess —
  protocol reference for game-play arms.
- **Xiangqi-R1** (2025): spatial-reasoning training, not eval — shows the
  Chinese-chess gap is even worse; framing color for our "small models
  can't do spatial" story.
- **Maia4All** (2025): human behavior modeling — not benchmark work.
- **GAMBIT / SKATE** (2026): multi-agent robustness evals — out of scope.

---

## What we borrow (summary)

| Source | Borrowed |
|---|---|
| ChessBench | dataset menu (lichess evals + puzzles) |
| ChessQA | category taxonomy + per-category error analysis |
| LLM CHESS | game-play protocol, Elo-vs-fixed-engine |
| Easy2Hard | difficulty-stratified reporting |
| KinGPT/LLM-Modulo | memorization-robustness framing, verifier baselines |
| Geometric stability | perturbation testing (our reps are format perturbations) |
| State tracking | legal-move-distribution metric |
| Spec-gaming | prompt hygiene against hacking |
