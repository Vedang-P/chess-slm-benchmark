# Careful Design: cavegemma-style trace training for gemma-4-E2B (chess lucid reasoning)

Design date: 2026-08-16. Template: the REAL cavegemma recipe (fetched from
JuliusBrussee/cavegemma, source files read: train_unsloth.py, config.toml,
synthesize.py, filter.py). Adapted to chess + our constraints (3k deepseek
API budget, 30h/week T4, no-engine-at-inference).

---

## The design principle

cavegemma = [teacher writes target-format text] → [integrity filter] → [small
QLoRA SFT, completion_only_loss, no packing, 3 epochs over ~2k pairs].

Our analog:
- teacher: deepseek writes **lucid chess traces** (instead of claude writing
  caveman rewrites)
- integrity filter: **Stockfish verification** (instead of fence-integrity —
  strictly stronger)
- student: gemma-4-E2B, QLoRA all-linear, completion_only_loss (our
  prefix-difference mask, already built)

The ONE thing cavegemma never had to prove: that the target format improves
*task accuracy*. That is our thesis and our risk.

---

## Gate 0 (BEFORE any design commitment): fill the evidence gap

**Measure checkpoint-4000 (0.107-epoch labels model) thinking-ON on 200
positions.** We have the adapter on HF. The 61.1% base was thinking-ON. Our
live probe (flat 0.56) was thinking-OFF — it answers a different question.

Cost: ~1-2h GPU (200 positions, thinking ON) or one short kernel.

**Decision tree:**
- ckpt-4000 thinking-ON > 61.1% meaningfully → labels carry competence at
  0.1 epoch → **mixed run** (labels + upsampled traces) is the design.
- ckpt-4000 thinking-ON ≈ 61.1% → labels alone don't move selection yet →
  **traces-only imprint run first** (cavegemma exact analog), then add labels
  only if traces move style.
- Either way we learn it in <2h.

---

## Stage 1 — Synthesis (deepseek lucid traces, 3k budget)

### 1a. Prompt spec (mirrors cavegemma's CAVEMAN_INSTRUCTIONS structure)

```
You are an expert chess player analyzing a position to choose between two
candidate moves. Think in a compressed, telegraphic style.

Rules:
- Short fragments only. No full prose sentences, no filler, no hedging.
- State only what matters: material balance, the key tactical/positional
  fact, the line you trust (max 3 plies), and your choice.
- Pattern: [fact] [line] [choice].
- Every move you mention must be a legal move in the position.
- Do not mention moves other than the two candidates.
- End with EXACTLY one line: MoveA:<move> or MoveB:<move>

Position: <FEN>
MoveA:<uci> MoveB:<uci>
```

Design choices (with reasons):
- **One step, not two** (cavegemma normalized then cavemanned). MATE inputs
  are already standard; the two-step cost would halve our 3k budget. If trace
  quality on inspection is poor, we add a normalize step later — the 50-sample
  pilot decides.
- **Restricted vocabulary by instruction**, not by force — mirrors
  cavegemma's "pattern: [thing] [action] [reason]" prompt rule.
- **3 plies max** in claimed lines — keeps traces short AND makes
  eval-stability verification tractable (fewer moves to check).

### 1b. Quality pilot (10% of budget, BEFORE committing the rest)

Synthesize on 50 positions. Inspect by hand (trace length, style, parseable
MoveX ending). **Gate:** ≥80% of traces are genuinely telegraphic (target
compression band 0.10-0.85 of deepseek's natural verbose output), ≥95%
end with a parseable MoveA/MoveB, ≥70% pass Stockfish final-choice
verification (else deepseek's 92.2% final-choice rate should bound this).

If the pilot fails style quality → tighten the prompt. If it fails
verification → nothing wrong, that's the expected yield; spend the rest.

---

## Stage 2 — Filter (our stronger "fence integrity")

Per trace, drop on ANY violation (hard rules, like cavegemma):
1. **Final choice mismatch** — trace's MoveA/MoveB ≠ MATE truth (primary;
   deepseek's 92.2% accuracy bounds yield at ~90%)
2. **Illegal move mentioned** — any UCI in the trace not legal on the FEN
   (python-chess; cavegemma's byte-exact analog)
3. **Eval-instability** — any claimed move changes eval > 100cp against the
   mover (Stockfish d14, position-before vs position-after)
4. **Compression band** — trace token count outside (10, 250) tokens
   (cavegemma's compression band analog; keeps lucid lucid)
5. **Not lucid** — article-density-style gate: trace contains prose markers
   (optional, only if pilot shows verbosity bleeding through)

Expected yield: 3000 raw → ~2400-2600 verified (deepseek 92.2% final-choice
× ~95% legality × ~95% stability). Matches cavegemma's 58% overall yield
pattern (ours should be higher — our teacher is stronger on the task).

Split: 90/10 with position-disjoint eval (cavegemma pinned seed pairs).

---

## Stage 3 — Training (the careful part: what mix, what config)

### 3a. The training row format (must match eval byte-for-byte)

```
user:  <EXACT eval instruction + input + ANSWER_SPEC>   ← identical to run_mate_eval
assistant:  <lucid trace>\nMoveB:<move>                ← the verified trace + answer
```

Critical: the user turn is the byte-identical eval prompt (already built and
verified in build_mate_lora_data.py). The assistant turn is trace+answer.
`completion_only_loss` = our prefix-difference mask (already built, verified
9 assistant tokens).

### 3b. Config (from cavegemma's config.toml, adapted to 2B)

| Setting | cavegemma (31B) | ours (2B) | reason |
|---|---|---|---|
| LoRA r | 16 | **32** | 2B has less capacity; we train competence+style, not just style |
| LoRA α | 32 | 32 | keep |
| targets | all 7 projections | all-linear (our proven path) | keep |
| epochs | 3 | **traces seen ~3×** | cavegemma's 3-epoch imprint, applied to trace subset |
| batch | 2×8 | 2×8 | our measured rate ~370 steps/hr |
| lr | 2e-4 | 2e-4 | keep |
| packing | false | false (our collator never packs) | keep |
| seq len | 4096 | 2048 | traces ≤250 tokens + prompt ~250 |
| loss mask | completion_only | prefix-difference mask | keep |
| resume | checkpoint | HF checkpoint + resume (built) | keep |

### 3c. The mix — decided by Gate 0

- **If labels carry competence (ckpt-4000 > 61.1% thinking-ON):** one run,
  60k labels at 1 epoch + traces upsampled to ~15-20% of steps. Traces get
  ~3 effective passes (cavegemma's 3 epochs, trace-centric).
- **If labels don't yet:** traces-only run first (~2.4k × 3 epochs ≈ 4-6h),
  probe, then add labels in a second run if traces moved style.

The mix is decided by data, not by us. That's the careful part.

---

## Stage 4 — Evaluation (the gates that matter)

Per stage, on 200-position probe (thinking ON — the real protocol):
1. **Parse rate** (base gemma: 0.0 at 1024 budget thinking-ON — the model
   must now emit an answer; cavegemma's "eval acc 81.5%" analog)
2. **Accuracy** vs 61.1% base and 63.5% MATE bar
3. **Trace quality**: median trace length (target ≤ 150 tokens), % ending in
   parseable MoveX, compression vs deepseek's natural output
4. **tokens-per-correct** — the efficiency headline

Full 1000-position eval only after the probe passes.

---

## Compute & cost audit (real)

| Step | Cost |
|---|---|
| Gate 0: ckpt-4000 probe (200 pos, thinking ON) | ~1-2h GPU (1 short kernel) |
| 50-trace pilot (deepseek API) | 50 calls (1.7% of budget) |
| 2950 traces (deepseek API) | 2950 calls (98.3% of budget) |
| Stockfish verification (3k traces) | ~2-4h CPU (parallel kernels) |
| Stage 3 SFT (traces-only, ~2.4k×3) | ~4-6h GPU (1 kernel) |
| Stage 3 SFT (mixed, 60k+traces) | ~8-10h GPU (1 kernel) |
| Probe evals | ~2h GPU each |
| **Total** | **~22h GPU + ~4h CPU, inside 30h/week** |

---

## Order of operations

1. **Gate 0**: eval checkpoint-4000 thinking-ON (fills the evidence gap that
   decides the mix)
2. **Pilot**: 50-trace synthesis → inspect style quality → tighten prompt if
   needed
3. **Full synthesis**: 2950 traces → Stockfish verify → filter → split
4. **Train** (traces-only or mixed, per Gate 0) → probe
5. **Gate**: probe accuracy + trace quality → if pass, full 1000 eval
6. Decision: continue to verification-gap phase (self-verify TTS) or ship

---

## What we will NOT do (learned from our failures)

- No 600k-row volume runs (131h — impossible)
- No on-policy self-traces before the model can reason (gemma produces no
  traces — the exact error the user caught)
- No thinking-OFF evals as the decision metric (protocol is thinking-ON)
- No filter loosening to inflate yield (cavegemma's 1.0-band bug: it made
  the model undershoot gold compression)
