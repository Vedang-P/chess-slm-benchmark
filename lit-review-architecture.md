# Literature Review: Techniques to Make a 2B Model Beat a Frontier at Verifiable Reasoning (Chess)

Extended scope review, 2026-08-12. 83 new papers appended to
`.rstack/lit-review.jsonl` (now 123 records), sourced from arXiv via four
parallel research passes: (A) chess LLM post-training, (B) math-reasoning
RLVR/PRM/distillation, (C) general ML PEFT + inference techniques,
(D) compressed/lucid/latent reasoning. Every arXiv ID was verified against the
arXiv API by the search agents.

## The engineering question

We have: a 2B Gemma, ~30h of T4-class GPU, unlimited training positions
(MATE train 1.42M rows all four formats, self-play), a perfect oracle
(Stockfish) **at training time only** (no engine at inference), and a target:
beat deepseek-v4-flash (85.8% strategy) on all 4 MATE subsets, measured as
accuracy + tokens-per-correct, with high novelty.

## Theme 1 — The recipe that works: SFT competence → RLVR refinement

The dominant 2025-26 finding across chess AND math:

- **SFT on best-move prediction gives the strongest RL starting point, and RL
  generalizes where SFT memorizes** (2604.05134 chess study; 2501.17161
  SFT-vs-RL generalization). **But RL plateaus if the base lacks competence**
  (Chess-R1 2507.00726), so SFT-first is mandatory.
- **The full template exists at our scale**: C1 Master Distillation
  (2603.20510) = SFT on engine-verified CoT traces → RLVR → theme-balanced
  data; a 4B beats its teacher. Same compute as ours: 2503.16219 (1.5B GRPO,
  24h, $42); Xiangqi-R1 proves GRPO+engine rewards work at 0.5B.
- **Budget reality check**: 2607.16097 — RL gains are bounded by what
  pretraining/SFT provided; at 2B expect RL to sharpen known moves, not
  conjure new hard ones. So the SFT stage must carry most of the chess
  competence; RL adds generalization + style + calibration.

## Theme 2 — Verifiable-reward RL engineering (the math community's gift)

- **GRPO-family**: DeepSeekMath (2402.03300) → DAPO (2503.14476, clip-higher +
  dynamic sampling for zero-advantage groups — most chess groups will be
  uniformly-correct) → Dr. GRPO (2503.20783: **drop length normalization or
  RL teaches verbosity for free**) → λ-GRPO (2510.06870, learnable token
  weights, validated at 1.5B/3B) → S-GRPO (2508.05928, noise-aware advantage —
  Stockfish evals ARE noisy near-equal positions).
- **Rarity-aware credit** (2608.03467): GRPO starves rare-but-correct moves —
  a real chess failure mode (rare tactical motifs).
- **Adaptive rollout budget** (2608.11368 PAIR): spend rollouts where
  positions are hard.
- **No-signal groups** (2608.09826): when both candidate choices are correct,
  distill the *reason* from an answer-conditioned teacher instead of dead
  GRPO steps.
- **Zeroth-order escape hatch** (2505.13430) and **speculative RL rollouts**
  (2608.04962) if memory/throughput bind on T4.

## Theme 3 — Process supervision: chess has a free PRM

- Chess = every move is a verifiable step. `Let's Verify Step by Step`
  (2305.20050), **Math-Shepherd** (2312.08935: label steps via MC rollouts =
  Stockfish rollouts from the resulting position), **step-semantics PRM**
  (2409.12122: label a move by the consequence position's eval — the natural
  chess move-quality label), **PRIME** (2502.01456: implicit PRM from outcome
  labels via DPO, works at 1.5B), **TDRM** (2509.15110: TD-consistent PRMs
  with 2.5k samples — eval(move_i) ≈ eval(move_{i+1}) is exactly the chess TD
  target), **VPS** (2605.12519: outcome-RL degrades reasoning quality;
  process supervision restores it — direct evidence for verifying every step
  of a lucid trace).
- **Curriculum via verifiable subproblems** (2605.22074) — phase/endgame →
  middlegame curricula.

## Theme 4 — Small models learn short traces better

- **"Small Models Struggle to Learn from Strong Reasoners"** (2502.12143):
  ≤3B models do *better* on short, simple chains — the core theoretical
  justification for the lucid/caveman distillation thesis.
- Style-shaping tools: ThinkPO (2502.13173, DPO verbose-vs-lucid), Walk
  Before You Run (2505.21178, two-stage RL: long then concise), LCPO
  (2503.04697, prompt-given length budget), ORION (2511.22891, Mentalese
  curriculum), BARD (2511.01470, budget-aware distillation), O1-Pruner
  (2501.12570, length-harmonized pruning), Key-Point distillation
  (2407.10167, skeleton summaries — = chess candidates+motifs).
- **Guardrail**: ciphered-language reasoning degrades accuracy (2510.09714) —
  test early that the 2B can emit chess notation fluently before betting the
  pipeline on compressed output.

## Theme 5 — Latent reasoning (extreme compression)

- Survey (2507.06203); Thinking States (2602.08332, SFT-only latent); Fast
  Thinking codebook (2509.23633, sketch at train, no thought at inference);
  rStar-Math (2501.04519, MCTS + process-reward self-play without a teacher).
  Mostly optional for us — but "emit sketches at training, strip thinking at
  inference" is the logical endpoint of lucid style.

## Theme 6 — Test-time compute (no engine at inference)

- Self-consistency voting (2203.11171) — free points on a 2-candidate task.
- Compute-optimal allocation (2408.03314): verifier-guided BoN beats voting
  at higher budgets → train a *self-verifier* (V-STaR 2402.06457 reranker;
  rStar 2408.06195 critic; SVR 2607.28457 verdict+confidence RL) so the model
  selects among its own samples using its internalized oracle — no engine.
- s1 budget forcing (2501.19393), early-stopping via attention (2606.15070)
  or non-convergence detection (2607.21433) for tokens-per-correct.
- **Pass@K matters**: 2608.11829 — distillations that look weak greedily win
  with sampling budget.

## Theme 7 — PEFT engineering on the T4

- Rank/module/quant sweep first (2607.25583); rank allocation by
  representation sensitivity (2607.09757); per-task adapters + merging
  (2607.20561 CT-Merging) or per-slice routers (2602.04447 Mixture of
  Masters); orthonormal init for RLVR (2606.31813); adapter hand-off resume
  (2504.15610); QAT if 4-bit inference needed (2606.15682); contrastive
  hidden-state RL (2603.17305).

## Theme 8 — Evaluation hygiene

- Memorization vs generalization: novel/held-out positions mandatory
  (2601.16823, 2605.17565); brittleness testing (perturbed/rotated FENs);
  outcome-reward shortcutting warning (2604.22074) — must show the model
  doesn't just memorize Stockfish-selected labels.
- Efficiency metrics: LLMThinkBench overthinking score (2507.04023), ReEfBench
  process audit (2601.03550) — our tokens-per-correct is the chess analog.

## Design implications (what we're borrowing)

1. **SFT carries competence**: 200k+ rows, all 4 prompt formats, phase-stratified,
   lucid-verified traces (short — 2B learns short better).
2. **RLVR refines**: GRPO-family with DAPO clip-higher + Dr.GRPO (no length
   normalization) + S-GRPO noise-aware + rarity-aware credit; process reward
   via per-step Stockfish verification; concise-reward stage for lucid style.
3. **Internalized verifier**: train the model to emit verdict+confidence
   (SVR) / a reranker (V-STaR) — inference-time selection among own samples.
4. **Test-time compute**: self-consistency voting + self-verifier selection.
5. **Eval with novelty positions** for credibility.
