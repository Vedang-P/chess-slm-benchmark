# Search-in-Language (SIL): train a 2B LM to verbalize engine search at inference

Decision 2026-08-27. Replaces EGSD (user rejected as "caveman SFT with a filter
+ a loop"). The method — not the data — is the contribution.

## The claim

**"Search-in-language: LoRA-SFT on verbalized engine search trees installs
implicit lookahead in a 2B natural-language model, and self-consistency voting
across sampled search traces gives a trainable test-time-scaling method that
beats frontier generalists at chess — no external engine at inference."**

Empty-cell check (from the C scout):
- SoS (Gandhi 2024): serialized search streams, but Countdown + from-scratch 250M
- MAV (DeepMind 2024): chess, but custom move-token format, from scratch, DeepMind-scale data
- ToT/LATS: prompted frontier models, no training
- DiffuSearch: diffusion, not NL LM
- NO ONE: NL + chess + LoRA on a pretrained 2B + verbalized engine trees +
  self-consistency at inference. Cell verified empty.

Risk (must preempt): "Transformers Struggle to Learn to Search" (Meta 2024).
Counter-evidence: SoS 250M learns Countdown search; Ruoss 270M → 2895 Elo
searchless. Our question is narrower: does NL search-verbalization SFT install
MEASURABLE lookahead at 2B — and we measure it directly (probes), not just via
accuracy.

## Method

1. DATA (CPU-only, ~30-60 min local / free CPU kernel):
   scripts/build_search_traces.py — Stockfish MultiPV=4 depth=14 per pool
   position → verbalized trace: position, candidates, per-line lookahead with
   engine evals, verdict, MoveA:/MoveB: answer. ~350 chars/trace. Every number
   comes from the engine (no self-judgment). Verified: 10/10 traces, correct
   answers, mate sign correct.
2. TRAIN: LoRA SFT (r32/α32 all-linear, lr 2e-4, completion-only loss) on ~5k
   verbalized traces — the existing train_mate_lora pipeline, new data.
3. INFERENCE = THE METHOD: sample K=1,4,8,16 search traces at temp 0.8 (cap
   1024), self-consistency majority vote on final move. K-curve = test-time
   scaling. No engine at inference.
4. MEASURE:
   a. accuracy vs K (self-consistency scaling curve)
   b. accuracy vs data depth (train on d8 vs d14 — does deeper teacher data
      buy deeper implicit search?)
   c. IMPLICIT-LOOKAHEAD PROBES: linear probe on hidden states → predict board
      2-4 plies ahead (Othello-GPT/Jenner style) — mechanistic evidence of
      installed search structure, compared across base / plain-SFT / SIL
   d. frontier comparison: DeepSeek V4 Flash on noexplain-1000, same protocol
   e. token efficiency: tokens-per-correct for SIL vs frontier

## Compute budget (~35h P100)
| Phase | GPU-h | Account |
|---|---|---|
| 0: trace build 5k | 0 (CPU local/kernel) | — |
| 1: SIL SFT | 8-12 | 1 |
| 2: eval suite (K-curves, depth ablations) | 8-12 | 2 |
| 3: probes | 4-6 | 1 |
| 4: frontier comparison | API | — |
| **Total** | **~25-35** | |

## Gates
- Phase 0 gate: ≥95% trace build success rate (done: 10/10 local)
- Phase 1 gate: SIL SFT ≥ plain-SFT on noexplain-1000 (no regression)
- Phase 2 gate: K=16 vote ≥ K=1 greedy (scaling exists) — else negative result
  on test-time scaling, still publishable as measurement study with probes

## Immediate next
1. Build full 5k trace dataset (local CPU, background)
2. Phase 0 kernel on account 1 (idle after v12 kill): SFT on 5k traces
3. Eval kernel on account 2: K-curves + probes
