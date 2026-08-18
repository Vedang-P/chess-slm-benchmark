# RLVR Plan: teach gemma-4-E2B chess reasoning with Stockfish-rewarded GRPO

Decision 2026-08-18. Replaces the trace-SFT-first plan. Evidence-backed pivot.

## Why RLVR now (and why it's not a panic pivot)

1. **Perfect verifier domain**: MATE selection is 2-choice, Stockfish gives
   exact rewards. RLVR is designed for exactly this (2503.16219, TinyZero,
   R1's rule-based rewards).
2. **RL generalizes where SFT memorizes** (2501.17161) — the core problem we
   saw (labels-SFT at 0.1 epoch: format learned, accuracy flat) is the
   documented SFT failure mode.
3. **Small-model precedent in board games**: Xiangqi-R1 (2507.12215) trained a
   0.5B with GRPO + engine rewards to strong intermediate strength. Our base
   (58.1% noexplain) is much stronger than a 0.5B from scratch.
4. **R1's lucid style emerged from RL** — the compressed/telegraphic style we
   wanted from trace distillation arises naturally from RL with length-aware
   rewards. We don't need deepseek traces at all: the model teaches itself
   the style.
5. **Compute-fits**: RLVR on 1.5B worked in ~24h on 4×A40 (~$42, 2503.16219).
   We have 30h/week T4-class — the same regime, one GPU at a time.

## The design

### Model & training (proven stack)
- Base: google/gemma-4-E2B-it, QLoRA NF4 (our exact loader), LoRA r32
  all-linear, peft 0.14 (verified)
- **GRPOTrainer (trl 0.17)** — same version cavegemma used; compatible with
  transformers 5.13.1 (no upper bounds in its constraints)
- Group size 8, LoRA lr 1e-5 (RL on adapters is unstable at SFT lr),
  clip 1.0, no length normalization (Dr.GRPO 2503.20783 — chess rewards are
  length-agnostic; raw GRPO would teach verbosity for free)

### The reward (the design's core — each term verified by Stockfish)

```
r = 1.0 · outcome + 0.3 · process + 0.1 · style
```

- **outcome**: 1 if the model's final MoveA/MoveB == Stockfish best at the
  position (d12), else 0. Group-relative advantage (GRPO).
- **process**: per-step verification of the model's own trace — every UCI
  move mentioned must be legal (python-chess) and eval-stable (|Δeval| ≤
  100cp vs the position before the move, Stockfish d12). This is the
  "verify your reasoning" signal — the model learns to check its own claims,
  which is the verification-gap thesis in RL form. (VPS 2605.12519:
  outcome-only RL degrades reasoning quality; process supervision restores
  it.)
- **style**: brevity bonus gated on correctness: (1 − tokens/max) only when
  the outcome is correct; 0 when wrong. Never reward short-and-wrong.
  This is what produces lucid traces without distillation. (Walk Before You
  Run 2505.21178: concise RL after verbose RL; here we do it in one stage
  with the gate.)

### Position pool (self-play-free, from our data)
- MATE noexplain train positions, **difficulty-gated**: near-equal Stockfish
  evals (|evalA − evalB| small) — the deciding cases, where reasoning pays
  off. Phase-natural. Test-FEN-excluded.
- Sampled per rollout; the pool machinery exists (build_lucid_traces.py
  select stage, minus the deepseek part).

### Rollouts & budget (T4-feasible math)
- Prompt: exact eval prompt (byte-identical, thinking ON, force-answer
  variant to guarantee a MoveX answer).
- Response: lucid trace + MoveX:... (~100-300 tokens).
- Per step: 8 rollouts ≈ 2k tokens generation + 1 training step.
  At ~370 steps/hr SFT-equivalent and rollout overhead ×3-4, expect
  **~80-120 RL steps/hr → 3-4k steps in a 12h kernel** ≈ 24-32k unique
  positions seen. This is the TinyZero/2503.16219 regime.

### The eval gates (same protocol as baseline)
- 200-position probe after the run: accuracy, parse rate, trace length,
  tokens-per-correct. Compare vs base 58.1% and the partial 56.1%.
- Full 1000 eval if the probe passes.
- **The verification-gap measurement**: per-sample, does the model's own
  process-verified trace correlate with correctness? (Do verified traces
  predict right answers?) — the paper's claim.

## Milestones (GPU-blocked until Aug 22, so build first)

1. **Now-until-Aug-22**: build + locally validate the GRPO trainer
   - trl 0.17 GRPOTrainer against our loader (the gemma4 wrap is proven)
   - reward functions (outcome/process/style) with a mock oracle
   - CPU smoke: 1 batch of 8 rollouts + 1 optimizer step on a tiny slice
     (validates the whole loop before it touches GPU)
2. **Aug 22 (or fresh account)**: 200-position probe of the BASE model RL'd
   for 3-4k steps (12h kernel) → eval probe
3. Decision gate: probe ≥ 65% → continue RL (another kernel) → full 1000
   eval; probe < 60% → reassess (reward tuning, more rollouts)

## Honest risks

1. **T4 RL throughput could be 2-3x slower than estimated** — the 3-4k steps
   in 12h is an estimate. Mitigation: shorter rollouts (style reward keeps
   traces short), and the group size can drop to 4.
2. **GRPO instability on LoRA adapters** (documented) — mitigations: low lr,
   clip, DAPO clip-higher if all-correct groups dominate, λ-GRPO token
   weights if needed.
3. **Base competence too low for RL to lift** (Chess-R1's plateau) — but
   Xiangqi-R1's 0.5B result argues against; and 58.1% is real chess signal,
   not random.
4. **Stockfish reward noise at d12** on near-equal positions — S-GRPO
   noise-aware advantage (2508.05928) if needed.

## What this replaces

- The 3k-trace deepseek pipeline (not needed — RL self-generates style)
- The trace-SFT-first plan (superseded; traces-only was to teach style,
  RL does that with rewards)
- Nothing else: eval harness, phase axis, protocol, HF/wandb infra all
  reused.
