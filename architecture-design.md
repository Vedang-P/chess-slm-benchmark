# Architecture Design: "Grounded Lucid Reasoning" — a 2B chess policy that internalizes its oracle

Design date: 2026-08-12. Compute: 30h T4-class (Kaggle, account `vedanggggg`).
Model: gemma-4-E2B (2B, 4-bit QLoRA). Eval: MATE 4×1000 (strategy /
noexplain / tactic / both). Target: beat deepseek-v4-flash (85.8% strategy)
on all four subsets, measured as accuracy AND tokens-per-correct. Hard
constraint: **no engine at inference** — Stockfish is a training-time oracle
only.

---

## 0. The thesis (what makes this novel)

Frontier models get chess accuracy with verbose RL-grown reasoning; small
models learn **short, verified** traces better than long prose (2502.12143).
We therefore teach a 2B model to (1) reason in compressed "lucid" traces, (2)
**verify every claim of those traces against an oracle during training**, and
(3) **internalize that verifier** — emitting a self-verdict + confidence so
that inference-time selection among its own samples needs no engine. We call
this **Grounded Lucid Reasoning (GLR)**. The four novelty pillars:

1. **Verified lucid traces** — compressed reasoning where every intermediate
   claim (candidate legality, eval-stability of claimed lines, final choice)
   is Stockfish-verified before it enters training data (extends C1 2603.20510
   + VPS 2605.12519 into the compressed-style regime).
2. **Oracle-internalizing RL** — RLVR with process + outcome + style rewards
   that trains the model's own **verdict+confidence head** (SVR 2607.28457
   style, oracle-supervised): at inference the model selects among its own
   samples by its internalized verifier, not by sampling luck.
3. **Phase-segregated data + phase-stratified benchmark** (the existing
   refined-idea pillar).
4. **Tokens-per-correct** as the headline metric (LLMThinkBench 2507.04023
   analog for chess) + memorization-vs-generalization eval (2601.16823).

---

## 1. Training stages

### Stage 0 — Rank/module ablation (full-corpus slices, compute-unconstrained)
Guided by 2607.25583/2607.09757: sweep r ∈ {32, 64, 128}, attn-only vs
all-linear, lr 1e-4/2e-4, on ~200k-row slices; pick by eval accuracy on the
4×1000 (not a synthetic holdout). Capacity is a real research variable now
that the corpus is full-size — the sweep decides, no default.

### Stage 1 — SFT: competence + lucid style (full corpus)
**Data recipe** (`scripts/build_mate_lora_data.py` extended):
- **Every (position, format) pair that exists**: ~3.48M rows (noexplain
  1.42M + strategy 1.36M + tactic 350k + both 350k), phase-natural
  (~91/6/3, matching the eval), test-FEN-excluded. The four formats are one
  position pool with different prompt text (measured) — training all pairs
  teaches format-invariance, which the eval directly rewards (tactic and
  both testbeds are the same 1000 positions).
- **Lucid trace distillation channel (A2)**: ~150–250k rows (phase-oversampled
  toward endgame) where deepseek-v4-flash generates the *lucid* reasoning
  trace (compressed, ≤4k tokens); then **every claim is verified**: final
  choice must equal Stockfish-best (deep ~14) at the position, intermediate
  candidate moves must be legal and eval-stable (|Δeval| ≤ 100cp from the
  position's eval). Only fully-grounded traces enter the corpus. Verified
  traces get a `Verified: yes` footer in training.
- **Labels channel (B)**: plain labels `MoveX:<move>` for the full
  (position, format) set — the baseline arm and the volume backbone.
- Mix ratio (labels : traces) and epochs (0.5/1/2) are **eval-decided
  checkpoints**, not assumptions.
- Prompt format identical to `run_mate_eval.py` (byte-identical, candidates +
  trailing space). Assistant-only loss. QLoRA NF4, completion_only_loss.

**Smoke gates before spending budget**: (a) emits `MoveA/B` parseable on
every eval prompt (2510.09714 cipher guardrail), (b) eval accuracy on each of
the 4 formats ≥ base+10, (c) traces still compressible (lucid stats).

### Stage 2 — Verifier-RL (GRPO-family, 12–18h) — the novelty core
Rollout: sample the policy on position → (lucid trace, choice, verdict,
confidence). Reward = **three-term, verified**:
- **Outcome**: 1 if choice = Stockfish-best else 0 (group-relative advantage,
  GRPO).
- **Process**: per-step verification of the lucid trace (legality +
  eval-stability of each claimed line; VPS 2605.12519-style). Rewards
  *grounded* traces, kills hallucinated analysis.
- **Calibration**: verdict+confidence loss — the model's self-confidence must
  predict its own accuracy (2607.00164: verifiable rewards calibrate
  probabilities). This is the piece that makes inference-time self-selection
  work.
- **Style**: concise bonus (lucid stays lucid) — implemented as Dr.GRPO-style
  *absence* of length normalization plus a small brevity reward (2505.21178
  stage-2 recipe); tokens-per-correct is optimized explicitly.

**Stabilizers (from the math community, all validated at 1.5–3B scale):**
DAPO clip-higher + dynamic sampling (zero-advantage groups are common when
both candidates are fine) 2503.14476 · S-GRPO noise-aware advantage
(Stockfish@d12 is noisy) 2508.05928 · rarity-aware credit redistribution
(rare tactical moves) 2608.03467 · difficulty-gated rollout budget (PAIR)
2608.11368 · λ-GRPO token weights 2510.06870 if cheap.

**Curriculum**: phase-gated (endgame → middlegame → opening) and
difficulty-gated self-play positions (near-equal evals — the hard, deciding
cases), 2605.22074.

**If RL is too slow on T4**: fall back to SFT + internalized-verifier SFT
(Stage 2 becomes a small DPO-style stage on verified vs unverified traces,
2608.09826 + PRIME 2502.01456) — still beats labels-only by the literature.

### Stage 3 — Inference-time selection (no engine, pure model)
1. Sample K=4 completions per position (temp 0.7, lucid traces).
2. **Self-verifier selection**: pick the completion with the highest
   calibrated self-confidence (not raw vote) — the internalized oracle.
3. Fallback: majority vote among legal choices if ties.
4. Adaptive budget: confidence gate → short lucid traces on easy positions
   (tokens-per-correct payoff); early-stop on non-convergence (2607.21433).
5. Measured: accuracy_strict, parse_rate, tokens-per-correct per subset.

---

## 2. Compute budget (30h, T4-class)

| Stage | Hours | What |
|---|---|---|
| 0 ablation | sweep | rank/module sweep on ~200k slices |
| 1 SFT | full corpus | 3.48M (position,format) pairs + 150–250k traces; epochs by eval |
| 2 RLVR | 12–18 | GRPO, N=8 groups, ~15–30k steps, LoRA |
| eval | 2–3 | 4×1000 × {greedy, K=4 self-selected} + novelty subset |
| buffer | 2–3 | retries, hand-off resume (2504.15610) |

Budget lever: if Stage 2 stalls, keep Stage 1 adapter + add the DPO-style
verifier stage (≈2h) and rely on self-consistency at inference.

---

## 3. Evaluation & the paper claims

- **Primary**: 4×1000 MATE, accuracy_strict + tokens-per-correct, vs
  deepseek (85.8% strategy baseline) and MATE anchors (63.5/89.7/94.6/95.2).
- **Credibility**: novelty positions (mirrored FENs, theme-held-out,
  transformed material) — 2601.16823/2605.17565 hygiene; phase-stratified
  per-phase accuracy; verify reasoning quality of traces (ReEfBench-style
  audit, 2601.03550).
- **Paper framing**: the four novelty pillars §0; baselines = labels-only
  SFT (B), base gemma, deepseek, MATE anchors. Primary comparison: A2-GLR vs
  B at matched tokens and at matched accuracy.

---

## 4. Immediate next steps

1. Extend `build_mate_lora_data.py` → all four formats + phase labels +
   verified-lucid channel (deepseek trace pass, Stockfish verification —
   reuses `run_mate_eval.py` machinery).
2. Ship the **first SFT run** (Stage 0 config + Stage 1 data) to Kaggle
   (`build_mate_lora_kernel_notebook.py` extension), 4–6h.
3. In parallel: write the GRPO trainer (Stage 2) against `src/models.py`
   (needs a batch generator + oracle wrapper; verify reward terms on 100
   positions before the full run).
4. Eval harness: extend `run_mate_eval.py` with self-consistency/self-verifier
   modes (K samples + selection policy).
