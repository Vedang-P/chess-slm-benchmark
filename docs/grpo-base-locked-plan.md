# GRPO-Base Locked Plan — 5k Probe (2026-08-25)

**Goal:** GRPO directly on base `google/gemma-4-E2B-it` (no SFT) to find ceiling, fair vs `SFT→GRPO`. Same 1000 MATE eval.

**Locks from discussion 2026-08-25:**

- **Pool:** 5k stratified (`30% easy>150cp /40% med 60-150 /30% hard≤60`) from `600k` MATE train slice, deduped, test-FEN excluded. Build: `scripts/build_rlvr_pool.py --sample-rows 20000 --max-rows 5000 --difficulty-gate --max-gap-cp 60` + second shard `>150` + merge. CPU 2h. If 200-step probe ≥60% vs base 58.1%, rebuild 20k for long run.
- **Model:** `google/gemma-4-E2B-it`, QLoRA `r32 α32 all-linear 528-fallback` on `language_model` only, `merge_and_unload` not nested, fp16/no-quant + grad_ckpt + expandable_segments.
- **Prompt:** byte-identical to eval — `FEN + MoveA/B UCI + explanation + forced MoveA/B`, UCI (not SAN), no legal list. Fair.
- **Rollout:** `G=8` (fallback `G=4`), `max_completion 256` probe →512 full, `temp0.7 top-p0.95`, no thinking cap, `FEN+SAN` not needed.
- **Reward:** `P1` only — score **final MoveA/B + illegal + consistency** (final == Stockfish best among 2), ignore discarded UCI mentions. So exploring a bad line and discarding is never punished. `Dr.GRPO /MAX_TOKENS` (no 1/|o|, no /std). `w*` S-GRPO `p=0.15` optional.
  - **Arm A1 (outcome-heavy-long, no style):** `R = 1.0*outcome + 0.3*process + 0.0*style`
  - **Arm B (process-heavy):** `R = 0.1*outcome + 0.9*process + 0.0*style`
  - `outcome` = Stockfish-best binary (dense variant `clip((cp_chosen-cp_other)/400)` as ablate), `process` = P1 ratio with 1.5× verbosity penalty, `style` dropped (was `1-toks/max` if correct else 0 for lucid compression — conflicts with longer-reasoning goal).
- **Optim:** `lr 1e-5→3e-6 cosine 5% warmup`, `clip0.2 (DAPO 0.28)`, `β0/0.001`, `grad1.0`, `batch8`, `save50 top-3`.
- **Compute:** 2-step gate (<2min T4, check `500-char` contains `MoveA:` + `outcome>0`) → 200 steps G8 ≈12h T4 (1,600 prompts, 32% of 5k seen) → probe 200 positions vs 58.1% → ≥60% extend 400 → full 1000.
- **Checks:** 72.7% Stockfish-vs-expert agreement on 335 hard-only → noise; 5k stratified fixes. SFT caveman 55.4% @146 tok vs base 58.1% @1368 tok — style succeeded, selection didn't.

**Evidence:** DeepSeekMath 2402.03300 (GRPO), Chess-R1 2507.00726 (dense> sparse, 150 steps plateau), VPS 2605.12519 (outcome-only collapses, process fixes), 2607.02869 (process 63.7 > outcome 53.7, λ0.9 best), 2503.20783 Dr.GRPO, 2508.05928 S-GRPO (p0.15, 20% noise collapse), TinyZero 2503.16219 (1.5B 7k $42 80% AMC23).
