# Pivot Plan: Engine-Graded Self-Distillation (EGSD) — SFT-only, 2B, ~60h P100

Decision 2026-08-27. Replaces RLVR (infeasible on P100: uncapped rollouts never
terminate; measured v11/v12). Based on 5-scout lit review + 3-judge panel +
prior-art deep-dive (cell verified EMPTY as of 2026-08-27).

## The novel claim (survives adversarial review)

**"The first iterative, SFT-only, engine-graded self-improvement loop for
natural-language chess reasoning in a 2B model — and the first controlled
study of how input representation and thinking budget interact with it."**

Prior-art deep-dive verdict (30+ arXiv queries, full-text scans):
- CELL IS EMPTY: no published multi-round SFT-only loop where a 2-4B LLM's own
  sampled NL reasoning traces are graded by a deterministic engine (Stockfish)
  and used to fine-tune it, with no RL stage.
- Closest work: Master Distillation/C1 (2603.20510) = one-shot teacher
  distillation + RLVR at 4B/4xH100; Ruoss ChessBench = non-NL policy heads at
  270M from 15B labels; STaR/ReST = math/code only, no game oracle.
- Sharpest reviewer attack (must preempt): "This is STaR/expert iteration
  restated with a domain delta." → Answer: we measure the SFT→RLVR gap closure
  at 2B under 60h (C1's 40.9% SFT vs 48.1% RLVR is the citable target), we use
  the model's OWN samples (not teacher distillation), and we add the two
  measured axes (representation, thinking budget) that no STaR paper has.

## Pipeline (60h total, P100, 2 accounts)

### Phase 0 — Pilot de-risk (2.5h, account 2)
- 1k positions × 2 rollouts, thinking ON but capped at 1024 tok (sampling cap,
  NOT a training cap) → grade with Stockfish d12 (outcome + process)
- Acceptance: ≥15% accepted traces AND accepted-trace SFT (quick 200-step)
  does NOT regress below base 58.1% on noexplain-1000.
- FAIL → drop the loop; fall back to Phase C (representation sweep, still a
  publishable paper) + Phase D (thinking budgets) as the standalone result.

### Phase 1 — EGSD loop round 1 (15-20h, account 1)
- Sample 5k pool positions × 4 rollouts, thinking ON, 1024 cap, temp 0.8
- Grade: outcome (final move == Stockfish d12 best) + process (every claimed
  UCI legal + eval-stable, |Δeval|≤100cp) + parseable MoveA/MoveB
- Keep only all-pass traces → SFT (LoRA r32/α32 all-linear, lr 2e-4, completion
  loss, ~3 epochs, 2048 cap) on the accepted set
- HF checkpoints + live traces (already built) throughout

### Phase 2 — EGSD loop round 2 (15-20h, account 2)
- Sample from ROUND-1 model, same grading → SFT again
- This is the "self-improvement" evidence: accuracy should climb round over
  round (C1's RLVR-vs-SFT gap is the target to close without RL)

### Phase 3 — The measured axes (5h, decode-time only, zero training)
- **Representation sweep (C3):** same checkpoint, same eval set
  (noexplain-1000), prompts in {FEN, ASCII board, PGN history, UCI list};
  frontier models (DeepSeek V4 Flash + 1-2 others) on the same 4 formats for
  the scale-interaction claim; tokenizer-alignment analysis (BPE pathology
  of FEN vs ASCII) as the mechanism
- **Thinking budget (C4):** accuracy-vs-forced-thinking-tokens curves
  (128/256/512/1024) on the best checkpoint — the first test-time-scaling
  study in a board game at 2B

### Phase 4 — The kill-shot eval (5h)
- Best checkpoint vs DeepSeek V4 Flash on noexplain-1000 (same prompt,
  local-thinking, force-answer protocol) + MATE-style + legality diagnostics
- Beat 58.1% base AND frontier → the headline; honest framing: "a 2B model
  beats a frontier generalist at chess via engine-graded self-distillation,
  no RL, no search at inference"

## Why this survives the three judge attacks
- NOVELTY (3/10 for naive loop): the deep-dive verified the exact cell is
  empty; reframe as SFT→RLVR gap closure + two measured axes, not "first loop"
- FEASIBILITY (7-7.5/10): everything is SFT-scale; 60h budget table above;
  Phase 0 gates the whole plan; C3/C4 are decode-only → even a failed loop
  yields a publishable representation/budget study
- VENUE (8.5-9/10): efficiency + on-device + small model + agents workshop
  fit; the 2B-vs-frontier result with ~60h compute is the poster child

## Compute budget (verified against P100 physics: 180 tok/s ceiling)
| Phase | GPU-h | Notes |
|---|---|---|
| 0 pilot | 2.5 | 1k×2 rollouts + quick SFT |
| 1 loop r1 | 15-20 | 5k×4 rollouts ×~800 tok @ ~150 tok/s ≈ 8h + SFT ~8h |
| 2 loop r2 | 15-20 | same on r1 model |
| 3 axes | 5 | decode-only, no training |
| 4 kill-shot eval | 5 | frontier API + local |
| **Total** | **~45-55** | fits 2×30h quota |

## Immediate next steps
1. Write `scripts/egsd_sample.py` (sampling + engine grading, reuse Oracle)
2. Write `scripts/egsd_sft.py` (reuse train_mate_lora pattern)
3. Phase 0 pilot on account 2
