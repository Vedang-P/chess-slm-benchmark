# THE VERIFICATION GAP — a plan for a strong, compute-feasible paper

Working title: **"The Verification Gap: Oracle-Grounded Self-Verification as the
Efficient Lever for Small-Model Chess Reasoning"**

Date: 2026-08-16. Status: rethink after 2 failed directions (volume-SFT too slow
at ~6k rows/GPU-hr; tool-loop at inference not novel).

---

## 1. The hypothesis (novel, falsifiable, measurable)

A 2B model's MATE performance is bottlenecked by **verification, not
generation**. Its ability to *propose* the right move is closer to frontier
than its ability to *choose correctly between two candidates*. The frontier's
92.2% comes substantially from verification-style compute (long thinking =
self-checking candidates). Therefore:

> **Training a small model to verify its own chess reasoning — supervised by a
> perfect oracle (Stockfish) at training time only — improves its
> move-selection accuracy more per GPU-hour than any volume of generation-SFT,
> and deploys at inference with no engine.**

This is instantiated from a 2026 finding (2602.07594): generation and
self-verification are **asymmetric** — *learning to verify improves generation*,
while the reverse does not hold. We are the first to (a) instantiate it in
chess, (b) supervise verification with a perfect oracle instead of a larger
model or an LLM PRM, and (c) measure it on a tokens-per-correct efficiency
axis.

## 2. Why this is novel (stress-tested against the 2026 literature)

| Prior work | What it does | Our difference |
|---|---|---|
| T1 (2504.04718) | sLMs verify with *tools* at inference; finds sLMs struggle at verification even after distillation from larger verifiers | We train verification with a *perfect oracle* label source (Stockfish), deploy the model's OWN verifier — no tool at inference, no larger teacher |
| Learning to Self-Verify (2602.07594) | Shows verify→generate transfer, on math, with LLM-judge labels | Chess domain + perfect oracle labels + efficiency metric + phase axis |
| Reward Granularity in RLVR (2607.02869) | Process rewards > outcome rewards for small models (Qwen 0.5B) | We combine process-RLVR with oracle-supervised verification pretraining as the *entry point* |
| VPS (2605.12519) | Process supervision in chess RL | VPS supervises RL; we train an explicit *verifier head/behavior* the model can deploy at inference to select among its own samples |
| Can 1B beat 405B (2502.06703) | Compute-optimal test-time scaling with PRMs | We use the model's OWN learned verifier instead of a separate PRM — 1/10th the inference cost |
| Self-Distilled Reasoner (2601.18734), PAST (2608.08726) | On-policy self-distillation, no teacher | We add oracle labels to the student's own trajectories (privileged signal) |
| Reason-Reward-Refine (2607.05199) | Step-level correction with structured feedback (physics) | Same skeleton, chess oracle, and we *measure* the verify→generate transfer |

**The paper's three measurable claims:**
1. **The gap decomposition**: decompose frontier-vs-2B accuracy into proposal
   ability vs verification ability (both measurable on our testbed) — show the
   bottleneck is verification.
2. **The lever**: 12h T4 budget spent on verification training beats 12h spent
   on generation-SFT, on accuracy AND tokens-per-correct.
3. **Self-verified test-time compute**: K-samples + the model's own verifier
   approaches PRM-guided selection at a fraction of the cost — and needs no
   engine at inference (our original hard constraint preserved).

## 3. The method — four stages, all within 30h/week GPU + CPU

### Stage 0 — Verification dataset (CPU only, parallel, ~1-2 days wall)
On ~60k MATE noexplain train positions (test-FEN-excluded, phase-natural):
- **Propose**: sample the base 2B (or a quick SFT'd 2B) N=4 reasoning traces
  + a choice per position (we already have run_mate_eval machinery; cheap on
  CPU because short generations).
- **Verify (oracle)**: Stockfish labels each trace: final choice correct?
  each claimed step legal? eval-stable? (machinery exists in
  build_lucid_traces.py). Output per position:
  `{fen, candidate_a, candidate_b, traces[], truth, per-trace verdicts}`
- Result: **V-set** = the 2B's own trajectories + oracle verdicts. This is
  on-policy (student samples its own data) + privileged oracle labels
  (2601.18734 / 2608.08726 recipe), fully self-generated — **zero deepseek
  API cost**.

### Stage 1 — Verification SFT (one 12h kernel)
Train the 2B on V-set with a **verification-augmented format**:
```
user:  <position + candidates>
assistant: <lucid trace> | Verdict: MoveA | Conf: 0.7
```
- The *trace* is the student's own (on-policy, corrected toward verified-
  correct ones — 2608.08726 privileged adaptation)
- The *verdict+confidence* is oracle-supervised (correct = label truth;
  confidence target = model's own accuracy on similar cases → calibrated by
  the oracle, 2607.00164)
- Mix in the 3k deepseek lucid traces (style carriers) and, if useful,
  verified-correct self-traces only.
- Expected: ~10-12h on T4 at 60k rows. This is the ENTIRE SFT budget.

### Stage 2 — Process-RLVR (one 12h kernel)
GRPO with **step-level Stockfish process rewards** on each sampled trace
(legality + eval-stability per step + final choice), following 2607.02869
(process > outcome for small models) and VPS (2605.12519). Groups of 8,
difficulty-gated position pool (near-equal evals), phase curriculum. Style
kept lucid via concise reward (tokens-per-correct optimized).

### Stage 3 — Self-verified test-time compute (inference, cheap)
Per test position: sample K=4 traces+choices → the model emits its own
verdicts+confidence → select the highest-confidence self-verified choice;
fallback majority vote. **No Stockfish at inference.** Report:
- accuracy_strict, per-phase accuracy, parse_rate
- tokens-per-correct (the headline; we expect 5-50x better than deepseek)
- the gap-decomposition table (proposal rate × verification rate)

## 4. Experiments / ablations (the paper's tables)

| # | Arm | Expected purpose |
|---|---|---|
| 1 | Base gemma (measured 61.1%) | floor |
| 2 | Generation-SFT (12h, 60k labels, same budget) | the "spend on generation" arm — our main comparison |
| 3 | Verification-SFT (12h, V-set) | the "spend on verification" arm |
| 4 | Arm 3 + process-RLVR (12h) | the full method |
| 5 | Arm 4 + K=4 self-verified TTS | inference scaling, no engine |
| 6 | Deepseek-v4-flash (92.2% noexplain, measured) | frontier; tokens-per-correct costed |

Ablations: verifier head on/off at inference; K ∈ {1,2,4,8}; process vs
outcome reward (reproducing 2607.02869 in chess); confidence calibration
(reliability diagrams, 2607.00164); phase-stratified breakdown.

**Headline comparison (tokens-per-correct):** deepseek ~19k tokens/position
(measured from the archive). Ours: lucid trace ~200-400 tokens × K samples +
verdict. If arm 5 lands at ≥80-85% on noexplain, the efficiency claim is
5-50x — and the verification-gap decomposition explains *why*.

## 5. The novelty statement for the paper

"We show that on chess move-selection — a task with a perfect verifier — the
gap between a 2B model and a frontier reasoning model is substantially a
*verification* gap. Training the small model to verify its own reasoning
against an oracle (training-time only) improves accuracy more per GPU-hour
than generation-SFT, and yields an inference-time self-verifier that selects
among its own samples with no external tools — approaching frontier accuracy
at 1-2 orders of magnitude fewer tokens."

## 6. Feasibility audit against what we have

- ✅ Eval harness + MATE noexplain 1000 + baselines (61.1 / 92.2 / 63.5)
- ✅ Phase classifier + phase-stratified benchmark
- ✅ Lucid-trace prompt + verification machinery (build_lucid_traces.py:
  legality, eval-stability, final-choice checks)
- ✅ Pre-tokenized data pipeline + trainer (wandb, HF checkpoints, resume)
- ✅ Stockfish local + parallel CPU capacity (overnight Kaggle CPU kernels)
- ✅ 3k deepseek trace budget (style carriers, optional now)
- ⏳ NEW: V-set builder (student self-sample + oracle verdicts) — a new script,
  but reuses existing pieces
- ⏳ NEW: verification-format SFT (trainer tweak: verdict+confidence targets)
- ⏳ NEW: process-reward GRPO (stage 2) — the biggest new build, but GRPO on
  2B with verifiable rewards is exactly the 2503.16219 / TinyZero recipe
  (~$30 scale)

## 7. Risk register

| Risk | Mitigation |
|---|---|
| Verification training doesn't transfer to generation (asymmetry fails) | That is a *finding*; gap-decomposition still stands + efficiency story survives |
| GRPO unstable on T4 | DPO-style preference stage fallback (verify-pairs), still oracle-supervised |
| Accuracy ceiling below 75% | K=4 self-verified TTS is the lever; tokens-per-correct story holds at any accuracy |
| V-set quality (2B proposes poorly) | N=4 sampling + keep verified-correct; oracle filters noise |

## 8. Why this is the right paper for us

1. **Fits the compute**: two 12h kernels total (SFT + RLVR), CPU-heavy data
   build, cheap inference. No 11-day grind.
2. **Uses our hard-won assets**: eval harness, phase axis, lucid style,
   oracle verification — everything we built becomes load-bearing.
3. **Keeps the original thesis alive**: lucid/compressed reasoning + efficiency
   metric + no-engine-at-inference.
4. **Strong, honest claim**: we measure *why* small models lag (verification,
   not generation) and show the efficient fix. Even partial results produce a
   table worth publishing.
