# Literature Review: Teaching SLMs to reason in compressed ("lucid") style via chess

Generated on 2026-08-08. Scope: reasoning-trace distillation into small models;
the lucid/compressed reasoning style; chess as the testbed; existing gemma
reasoning fine-tunes.

## Background

RL-post-trained reasoning models (DeepSeek-R1 family) develop a characteristic
compressed, telegraphic internal monologue ("lucid"/"caveman" style) rather
than full natural-language prose. DeepSeek-R1 (arXiv:2501.12948) established
that (a) this style is elicited by RL with verifiable rewards (GRPO), and (b)
distilling R1 traces into small models (1.5B-70B Qwen/Llama) transfers
reasoning ability at low cost. The open question for us: does the COMPRESSED
STYLE itself carry the transferable signal, and can a 2B model acquire it —
measured on a cheap, verifiable domain (chess)?

## Key Themes

1. **Trace distillation beats label-only SFT for small models.** SuperCorrect
   (arXiv:2410.09008), Self-Enhanced Reasoning Training (arXiv:2502.12744),
   and Phi-4-Mini-Reasoning (arXiv:2504.21233) all show small models gain
   substantially from high-quality teacher traces vs labels alone.
2. **Trace CONTENT/STYLE matters, not just presence.** The 2025 study on
   large-teacher-to-student math reasoning (arXiv:2508.13037) finds teacher
   reasoning style and quality directly shape student gains — supporting our
   A-vs-B (style as treatment) design.
3. **Base-model ability is the binding constraint.** Chess-R1
   (arXiv:2507.00726) shows RL plateaus when the base model lacks chess
   understanding — favoring a strong reasoning-capable base (or a
   chess-aware starting point).
4. **Chess evaluation carries a memorization risk.** The generalization-vs-
   memorization study (arXiv:2601.16823) shows chess performance tracks
   prior density — MATE-only results need a novelty axis to be credible.

## Methods Landscape

- **Distillation**: SFT on (prompt → trace + answer); optional self-correct /
  self-evolution stages.
- **RL**: GRPO with verifiable rewards (math, code, chess via Stockfish:
  format + legal + top-move rewards; ChessArena arXiv:2509.24239).
- **Chess post-training**: Master Distillation (arXiv:2603.20510) = expert
  distillation + SFT + RLVR with theme-balanced sampling.

## Datasets and Benchmarks

- MATE move-selection (4 subsets x 1000; our committed eval).
- Lichess puzzle DB (6.1M positions, local) — unlimited SFT expansion.
- No standard "lucid reasoning" benchmark exists; we define tokens-per-correct
  as the efficiency metric.

## Limitations and Open Questions

- No published work explicitly treats reasoning STYLE (compressed vs natural)
  as the experimental variable with an efficiency metric. This is our gap.
- Public MATE training data (~208k rows) is ~3.5% of the authors' true scale.
- No verified gemma-4-E2B chess-specific reasoning distill exists; the
  generic reasoning distills (Gemini/Opus/Claude into E2B) are unvalidated
  on chess.

## Implications for Our Work

1. Option B (labels-only SFT) is the cheap baseline; Option A (trace
   distillation) has strong prior support from math literature.
2. Best student base: a pre-reasoning gemma-4-E2B distill (e.g.
   Ayodele01/gemma-4-E2B-Gemini-3.1-Pro-Reasoning-Distill, 32M LoRA, <think>
   tags) — skip teaching lucid style from scratch, SFT on MATE traces on top.
3. Add a novelty/generalization axis (unseen themes, transformed FENs) so
   MATE results aren't dismissed as memorization.
4. Skip RL (compute-constrained); SFT-only supports the core claim.
