# Engineering Decision Record — Grounded Lucid Reasoning (GLR) on gemma-4-E2B

2026-08-12. This is the "deets": every engineering choice, why, and what it
costs. Companion docs: `architecture-design.md` (stages + novelty pillars),
`lit-review-architecture.md` (83-paper evidence base), `refined-idea.md`.

---

## 1. The problem, restated with numbers

| Item | Value |
|---|---|
| Model | google/gemma-4-E2B-it, 2B, 35-layer text tower (hidden 1536, GQA 8/1) |
| GPU | Kaggle T4-class 16GB, fp16 tensor cores only (CC 7.5; bf16 supported but unaccelerated → we train fp16) |
| Compute | 30h GPU (account `vedanggggg`), resets 2026-08-15 |
| Eval | MATE 4 subsets × 1000 (strategy/noexplain/tactic/both), exact MATE prompts, 2-candidate selection |
| Baselines | deepseek-v4-flash 85.8% (strategy, thinking); base gemma 61.1% (strategy); MATE LLaMA-3-8B SFT anchors 63.5/89.7/94.6/95.2 |
| Gap to close | ~25 points vs deepseek on strategy, from a 61% base |
| Hard rule | no engine at inference — Stockfish is training-time only |

Deepseek's 85.8% is a *thinking* model with unbounded budget. Our inference
advantage: it's a 2-candidate forced-choice task → sampling + internalized
verifier selection (Stage 3) is cheap and legal.

---

## 2. Data engineering (the largest lever, cheapest to control)

### 2.1 Source: MATE train, all four formats
Local zips already on disk (`data/raw/mate-train/`): strategy, noexplain,
tactic, both (~1.42M rows total, ~208k after FEN dedup per phase pipeline).
**Why all four**: the subsets differ in *input format* — strategy/both carry
explanation text in the prompt, tactic carries a line+motif, noexplain is
bare. A model trained on one format silently underperforms on the others.
The four eval sets are the four formats, so train coverage must match.

### 2.2 Phase labels (existing artifact)
Our deterministic classifier (`scripts/build_phase_dataset.py`): opening =
ply ≤ 12, endgame = non-king material ≤ threshold, else middlegame. Every
training row gets a phase label; sampling is phase-balanced. This is
pillar 3 (also yields the phase-stratified benchmark).

### 2.3 Two channels, mixed ~60/40
- **Labels channel (B)**: raw MATE rows → chat pair `MoveX:<move>` (~200k).
  Cheap, high volume, already built (`data/positions/mate-lora/`).
- **Verified lucid traces channel (A2)**: ~20–40k rows where deepseek-v4-flash
  writes a compressed lucid trace (≤4k tokens) then the choice. **Every
  claim is then Stockfish-verified at depth 14**:
  - final choice == engine best at the position → keep, else discard
  - each intermediate candidate/line mentioned must be legal + eval-stable
    (|Δeval| ≤ 100cp from position eval)
  - fully grounded traces get a `Verified: yes` footer in the training text
  This is pillar 1: compressed style + verified process supervision (extends
  C1's trace distillation into the compressed regime with VPS-style
  per-step grounding).
- **Why mix, not all-traces**: trace generation costs ~20–40k deepseek calls
  (gateway, cheap); labels are free and provide volume + the B baseline arm.

### 2.4 Hygiene (non-negotiable, all existing machinery)
- MATE test FENs excluded at build time (already in `build_mate_lora_data.py`).
- Train/eval splits position-disjoint.
- Eval prompt byte-identical to training prompt (candidates + trailing space;
  `ANSWER_SPEC` verbatim; `enable_thinking=False` at eval parity).
- Novelty positions for the paper: mirrored FENs, theme-held-out, material
  transforms (2601.16823 hygiene) — a separate 1–2k-position probe, not the
  primary 4×1000.

### 2.5 Cost
- Stockfish verification: depth 14 ≈ 0.1–0.5s/position on CPU → 20–40k
  positions ≈ 1–3h CPU, parallelizable, runs outside the 30h GPU budget
  (local mac or Kaggle CPU kernel).
- deepseek lucid trace pass: 20–40k × ~1.5k tokens ≈ 30–60M tokens ≈ few
  hours of gateway time, local, no GPU.

---

## 3. Stage 0 — configuration ablation (1–2h GPU)

Following 2607.25583 (rank × modules × quant sweep) and 2607.09757 (rank
allocation by representation sensitivity):

- Sweep on a 20k-row slice: r ∈ {16, 32, 64} × {attn-only, all-linear} ×
  lr ∈ {1e-4, 2e-4}. Pick by 2k held-out accuracy per format (4 numbers).
- **Default (if inconclusive)**: r=32, alpha=32, all-linear, lr=2e-4, 1 epoch,
  effective batch 16, NF4 double-quant, fp16 compute, dropout 0.
- Rationale: all-linear because chess move-selection needs MLP (pattern)
  capacity and attention (state binding); r=32 as the middle of the 2607.25583
  recommendation band; fp16 (not bf16) because T4 has no bf16 tensor cores.
- LoRA wraps ONLY `model.language_model` (never vision/audio encoders) —
  already verified against the gemma4 modeling code.

---

## 4. Stage 1 — SFT (4–6h GPU)

- Format: `apply_chat_template(enable_thinking=False, return_assistant_tokens_mask=True)`; assistant-only loss (`-100` on prompt); **no packing** (mask correctness matters more than throughput at this size).
- Seq len 2048 (lucid traces fit; 4k budget only for hard endgames).
- Trainer: HF Trainer, cosine lr, warmup 5%, eval every 2000 steps, save
  every 10000, adapter hand-off resume for interrupted T4 sessions
  (2504.15610 pattern).
- **Smoke gates before spending the 30h**: (a) parse_rate ≥ 0.98 on all four
  formats (the 2510.09714 cipher guardrail — if the model can't emit
  `MoveX:<uci>` fluently, everything downstream fails); (b) +10 points over
  base gemma per format; (c) lucid channel still compresses (tokens-per-
  correct improving, not just accuracy).
- Why not "reasoning model base": earlier notes suggest starting from an
  existing gemma reasoning distill (Ayodele01) — Stage 0 can A/B one
  adapter-from-base vs adapter-from-reasoning-distill on the 20k slice (both
  fits in the ablation hour).

---

## 5. Stage 2 — Verifier-RL (12–18h GPU) — the novelty core

### 5.1 Why GRPO, not PPO, not pure DPO
- GRPO: no value network (group-relative advantage) → halves memory vs PPO,
  fits QLoRA on 16GB. The math community's default for rule-based rewards
  (2402.03300). We have the strongest possible reward signal (perfect
  oracle), so a critic is wasted capacity.
- DPO-style (PRIME 2502.01456) is the *fallback* if GRPO is unstable/too
  slow — one stage, no rollouts. GRPO first because 2501.17161: RL
  generalizes where SFT memorizes.

### 5.2 The reward (three verified terms + style)
Per sample the model emits: `lucid trace → choice → verdict → confidence`.
```
r = 1.0·outcome + 0.3·process + 0.2·calibration + 0.1·style
```
- **outcome**: 1 if choice == Stockfish-best(d14) at the position else 0.
  (Note: MATE labels are the authors' engine labels; we re-verify with our
  own Stockfish binary to keep ground truth engine-consistent.)
- **process**: fraction of trace claims that verify — each candidate move
  legal (python-chess), each claimed eval-stable (|Δeval| ≤ 100cp vs
  Stockfish eval of that position). Kills hallucinated analysis; rewards
  grounded compressed reasoning (VPS 2605.12519).
- **calibration**: the model's self-confidence must match its own accuracy.
  Per-sample target: 1 if correct else 0; loss = MSE(conf, outcome) — trained
  jointly, so the confidence becomes trustworthy (2607.00164: verifiable
  rewards calibrate probabilities). **This is what powers engine-free
  inference-time selection.**
- **style**: brevity bonus, gated on correctness (1 − tokens/max_tokens for
  correct samples; 0 for wrong ones — never reward being short and wrong).
  Style is additive, not dominant: lucid style should survive RL.

Advantage: `(r − mean(r_group)) / std(r_group)` per GRPO, with S-GRPO
noise-aware variance (2508.05928) because Stockfish@d12 is noisy on
near-equal candidates.

### 5.3 Stabilizers (each with a 1.5–3B validation behind it)
- **DAPO clip-higher + dynamic sampling** (2503.14476): most MATE positions
  are easy → both-candidates-correct groups have zero advantage; dynamic
  sampling keeps gradients alive, clip-higher prevents collapse when the
  group is uniform.
- **Dr.GRPO, no length normalization** (2503.20783): chess reward is
  length-agnostic; vanilla GRPO would teach verbosity for free, destroying
  tokens-per-correct. We optimize length explicitly via the style term
  instead.
- **Rarity-aware credit redistribution** (2608.03467): common-move groups
  dominate GRPO credit; rare-but-correct tactical choices get starved.
- **Difficulty-gated rollouts** (PAIR 2608.11368): easy positions get 4
  samples, hard (near-equal Stockfish evals, self-play-sourced) get 16.
- λ-GRPO token weights (2510.06870) if the basic run stalls.

### 5.4 Training loop & T4 feasibility
- Rollout: policy generates group of N=8 (temp 0.7, top-p 0.95) per position;
  verification via Stockfish subprocess (CPU) in parallel with generation.
- Throughput estimate: 2B forward ≈ 0.8–1.2s/step @ 2048 tokens on T4 fp16;
  rollout ≈ 2–4s per 8-sample group (short traces ≈ 150–300 tokens); step
  ≈ 5–8s wall → **3–5k steps in 6–10h**, then scale to 12–18h if stable.
  This is honest worst-case math; the fallback is the DPO stage (§5.5).
- Curriculum: phases endgame → middlegame → opening (verifiable subgoals,
  2605.22074) + self-play-sourced hard positions (near-equal eval pairs).
- LoRA lr 1e-5 (≈10× lower than SFT — RL on adapters is unstable at SFT lr),
  clip 1.0, rollout count as in §5.4.

### 5.5 Fallback (2–3h) — the "DPO-verifier" stage
If GRPO is unstable or too slow: SFT checkpoint + one preference stage —
`Verified: yes` traces preferred over unverified/plain traces (ThinkPO
2502.13173 / PRIME 2502.01456 / 2608.09826 privileged-signal distillation).
Keeps pillars 1, 3, 4 intact; pillar 2 becomes "calibration SFT" (train
verdict+confidence on verified samples with MSE to outcome). Weaker than
full RLVR but literature-backed and T4-safe.

---

## 6. Stage 3 — inference (no engine, pure model)

Per position:
1. Sample K=4 lucid completions (temp 0.7).
2. **Self-verifier selection**: among completions whose verdict is
   `correct`, pick the max-confidence one. If none verdict-correct → majority
   vote on the choice. (Pillar 2 at inference: selection by the
   oracle-internalized confidence, not by luck.)
3. Adaptive budget: confidence gate → easy positions get short traces
   (tokens-per-correct payoff); non-converged generations (low verdict
   confidence) trigger re-sample, not extended generation (2607.21433).
4. Measured: accuracy_strict, parse_rate, tokens-per-correct × 4 subsets,
   vs greedy single-sample, vs majority-vote-only.

Why this beats plain voting: with K=4 and p≈0.65 per-sample accuracy, plain
majority voting caps around 75% on 2-class; the calibrated self-verifier
selects the *right* sample more often than the plurality. Expected:
accuracy_strict ≥ deepseek parity with far fewer tokens per correct.

---

## 7. The novelty, exactly

| Pillar | What's new | Nearest existing work | Why we're beyond it |
|---|---|---|---|
| 1. Verified lucid traces | compressed (lucid) reasoning traces where every claim is oracle-verified before training | C1 (2603.20510) verifies traces but keeps natural-language CoT; CoD/lucid line (2502.18600, 2502.20122) has style but no verification | style × grounding have never been combined; "Verified: yes" supervised corpus is new |
| 2. Oracle-internalizing RL | RLVR that jointly trains verdict+confidence so the model selects among its own samples at inference, engine-free | SVR (2607.28457) is oracle-*free* self-verification; V-STaR (2402.06457) trains a separate reranker | a single adapter that is policy + calibrated verifier, in a domain with a perfect training oracle — chess-internalized verification |
| 3. Phase-segregated data + benchmark | game phase as first-class training-data dimension + phase-stratified benchmark | C1 balances by difficulty/theme; ChessQA by ability | no one segregates by opening/middlegame/endgame (refined-idea.md, novelty 7/10) |
| 4. Tokens-per-correct + memorization hygiene | efficiency as a headline metric with generalization probes | LLMThinkBench (2507.04023) math analog; 2601.16823 eval hygiene | chess-domain efficiency metric + forced novelty-position eval |

Honest caveat: every pillar has adjacent work; the novelty is the
*combination* (compressed + verified + self-verifying + phase-aware) at 2B
scale in a domain with a perfect oracle — a defensible workshop contribution
even if the RL stage is scaled back to the DPO fallback.

---

## 8. Budget ledger (30h)

| Stage | Hours | Hard gate to proceed |
|---|---|---|
| 0 ablation | 1–2 | config picked; smoke format gates |
| 1 SFT | 4–6 | ≥ +10 pts/format vs base; parse ≥ 0.98 |
| 2 RLVR (or DPO fallback) | 12–18 (2–3) | reward terms validated on 100 positions first; stable loss |
| 3 eval | 2–3 | 4×1000 × {greedy, vote, self-verifier} + novelty probe |
| buffer | 2–3 | resume hand-offs, re-runs |

CPU-side (outside GPU budget, local or Kaggle CPU): trace generation,
Stockfish verification, self-play position sampling.

---

## 9. Risks and their answers

| Risk | Mitigation |
|---|---|
| RLVR too slow/unstable on T4 | DPO-verifier fallback (pillar 2 preserved via calibration SFT) |
| 2B ceiling below deepseek per-sample | Stage 3 test-time compute (self-verifier selection + voting) |
| Model memorizes labels (2604.22074 warning) | novelty-position eval, mirrored/transformed FENs in the paper |
| Lucid style hurts accuracy (2510.09714 guardrail) | style reward gated on correctness; smoke gate (a) before 30h spend |
| MATE label = engine label discrepancy | re-verify with our own Stockfish binary at depth 14 |
| Adapter quality variance (2607.25583) | stage-0 sweep decides config, not intuition |
