# RLVR Plan: teach gemma-4-E2B chess reasoning with Stockfish-rewarded GRPO

Decision 2026-08-18. Replaces the trace-SFT-first plan. Evidence-backed pivot.

## UPDATE (2026-08-18, user decision): SFT-first ordering

The pipeline is now three stages, not RLVR-first:

1. **Caveman SFT** — teach gemma to reason in short chess lines. Deepseek
   is prompted with the Stockfish lines and asked to write ONLY the
   explanation (no move choice — the answer is engine-given and appended
   deterministically, so every training row is correct by construction;
   1995/2000 positions have engine_preferred == expert truth_label, the 5
   disagreements drop). QLoRA SFT on gemma-4-E2B, cavegemma recipe
   (trace-training-design.md: r32/α32 all-linear, lr 2e-4, traces seen
   ~3×, completion-only loss, no packing).
2. **Test on the same 1000 eval** — thinking-ON protocol vs the 58.1%
   base. Hypothesis: allowing the model to reason with itself longer in
   caveman style yields a measurable chess increase.
3. **RLVR on top** — the GRPO design below, starting from the SFT'd
   model instead of the base.

Why the order: SFT imprints the style (the model currently produces NO
parseable reasoning — base parse rate 0.0 thinking-ON), and stage 2
measures whether style alone moves selection accuracy. RL then optimizes
selection with the style already in place. The RLVR design sections
below are unchanged; only the starting point moves to stage 3.

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

1. **Now-until-Aug-22** (build + local validation, all CPU):
   - caveman traces: 2000 deepseek explanations of engine lines,
     verified, answer appended deterministically (RUNNING)
   - SFT converter traces → chat-pair JSONL, prompt byte-identical to
     eval (built + verified)
   - RLVR trainer locally validated (DONE 2026-08-18: CPU smoke, all
     three rewards real values)
2. **Aug 22 (or fresh account), stage 1**: caveman SFT, ~340 steps
   (~1h T4) → results/caveman-sft-adapter
3. **Stage 2**: 200-probe thinking-ON → full 1000 eval. Pass bar:
   measurable accuracy gain over 58.1% base + parse rate > 0.
4. **Stage 3**: RLVR on the SFT'd model (`--from-adapter` prep),
   3-4k steps (12h kernel) → probe ≥ 65% → continue → full 1000 eval;
   probe < 60% → reassess (reward tuning, more rollouts)

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

## Kernel launch pack (Aug 22+, exact commands)

Validated locally (2026-08-18): the RLVR loop ran on CPU (SmolLM2-135M,
1 step) with all three rewards computing real values (outcome 0.375 /
process 0.417 / style 0.824 on the smoke pool; 0.375 + 0.3·0.417 +
0.1·0.824 = 0.582 = logged total). The SFT prompt is byte-identical to
the eval prompt (verified against noexplain-slice rows 2026-08-18). The
caveman-SFT pool: `results/caveman/traces-2000.jsonl` (deepseek
explanations of engine lines, verified, answer appended
deterministically) → `scripts/build_sft_from_traces.py` →
`data/positions/caveman-sft/{train,eval}.jsonl` (90/10 position-disjoint).

### Stage 1. Caveman SFT (T4 kernel, first thing on Aug 22)

    python3 scripts/build_sft_from_traces.py \
        --traces results/caveman/traces-2000.jsonl \
        --out data/positions/caveman-sft
    python3 scripts/train_mate_lora.py \
        --train data/positions/caveman-sft/train.jsonl \
        --eval data/positions/caveman-sft/eval.jsonl \
        --out results/caveman-sft-adapter \
        --base google/gemma-4-E2B-it --rank 32 --alpha 32 \
        --lr 2e-4 --epochs 3 --batch 2 --grad-accum 8 \
        --max-seq-len 2048 --train-tag caveman-sft \
        --hf-repo vedangfake/chess-slm-benchmark \
        --hf-upload-every 1800

~1800 rows × 3 epochs / 16 per step ≈ 340 steps ≈ 1h T4 (measured rate
~370 steps/hr). HF checkpoint upload/resume built in.

### Stage 2. Same 1000 eval (thinking-ON, the 58.1% protocol)

    python3 scripts/run_mate_eval.py \
        --model google/gemma-4-E2B-it \
        --adapter results/caveman-sft-adapter \
        --task-file data/positions/mate-selection-test-noexplain.json \
        --n 200 --local-thinking --force-answer-prompt \
        --output_dir results/caveman-sft-probe
    # if the 200-probe passes: full 1000, same command without --n 200

Report: accuracy, parse rate, trace length, tokens-per-correct vs base
58.1%. Pass bar for stage 2 = measurable accuracy gain (and parse rate
> 0 — the model must now emit answers).

### Stage 3. RLVR on the SFT'd model (this plan's design)

    python3 scripts/train_mate_grpo.py \
        --base google/gemma-4-E2B-it \
        --train results/rlvr-pool/train.jsonl \
        --out results/rlvr-adapter \
        --rank 32 --lr 1e-5 --group 8 \
        --oracle stockfish --depth 12 \
        --thinking \
        --max-steps 3500 \
        --hf-repo vedangfake/chess-slm-benchmark --hf-tag rlvr \
        --hf-upload-every 1800 --save-steps 500

`--thinking` enables the `<|channel>` thought block in rollouts (the
design's thinking-ON variant — byte-matches the eval prompt the 58.1%
baseline used). If the thinking processor misbehaves on T4, drop
`--thinking` for the no-think arm and note it in the probe comparison.

**Prep needed before stage 3**: train_mate_grpo.py loads `--base` and
creates a NEW LoRA — it does not yet load the stage-1 SFT adapter. Add
`--from-adapter results/caveman-sft-adapter` (PeftModel.from_pretrained
on the base, then wrap with the new trainable GRPO LoRA) or merge the
SFT adapter into a saved base. Small, mechanical; do it during the
stage-2 eval.

### Stage 3b. Resume a killed kernel

Same command + `--resume-from-hf` — pulls the latest checkpoint from
`vedangfake/chess-slm-benchmark` under `rlvr/` (adapter + optimizer +
scheduler + trainer_state, uploaded every 1800s) and continues. A crash
also writes `rlvr/run-status.txt` to the same repo so the failure reason
is readable without downloading the multi-GB dir.

### Stage 3c. RLVR probe (200 positions, same protocol as the 58.1% baseline)

    python3 scripts/run_mate_eval.py \
        --model google/gemma-4-E2B-it \
        --adapter results/rlvr-adapter \
        --task-file data/positions/mate-selection-test-noexplain.json \
        --n 200 --local-thinking --force-answer-prompt \
        --output_dir results/rlvr-probe

Report: accuracy, parse rate, trace length, tokens-per-correct vs
base 58.1% (and the partial-SFT 56.1%). `--cpu` fallback (fp32) works
but is slow — only if the quota is still exhausted.

### Stage 3d. RLVR decision gate

- probe ≥ 65% → continue RL (another 12h kernel, resume) → full 1000
  eval on `mate-selection-test-noexplain.json`.
- probe < 60% → reassess: reward tuning, group 4, DAPO clip-higher if
  all-correct groups dominate, S-GRPO noise-aware advantage.

### Stage 4. Verification-gap measurement (the paper claim, still to build)

Per sample: does the model's own process-verified trace correlate with
correctness? The probe output carries per-sample traces; the analysis
pass (`_verify_trace` reuse over probe samples vs outcome) is the
remaining Milestone-2 code. Cheap: a post-probe script, no new training.
