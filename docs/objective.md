# Project Objective

**Short version:** Benchmark Gemma 4 on chess. Then fine-tune it to play
chess **better than DeepSeek V4 Flash** using **natural language reasoning
only**. The *how* of that fine-tuning is an open question (later).

**Long version.** This project exists to answer one question:

> Can a small, on-device model (Gemma 4 E2B/E4B) be made to outplay a
> frontier reasoning model (DeepSeek V4 Flash) at chess — using only its
> natural-language reasoning, no external search, no tools, no engine at
> inference time?

Two stages:

1. **Benchmark honestly.** Measure Gemma 4 (base) and DeepSeek V4 Flash on
   the same tasks under identical conditions: standard chess, FEN-only
   prompts, a fixed token budget, strict answer extraction, and every
   scored answer being the model's own text. Tasks: mate-in-1,
   mate-in-2, Stockfish best-move, and MATE move-selection (expert
   annotated). This gives the gap we are trying to close and the baseline
   numbers the paper reports.

2. **Close the gap by fine-tuning.** Fine-tune Gemma 4 (4-bit LoRA,
   starting with 50k MATE examples) so that its reasoning improves enough
   to beat DeepSeek V4 Flash at chess. How exactly — data mix, reward
   design, whether to teach it to "think in caveman style" like DeepSeek
   does, reasoning-trace supervision — is deliberately **not decided
   yet**. That is the next research question.

Constraints that are already decided:

- **Honesty first:** no fallback answers, no retries, no fabricated data.
  A model that doesn't answer is a no_answer with a reason.
- **Standard chess** (python-chess): castling, en passant, double-step,
  promotion are all legal.
- **FEN only** (Phase-1 evidence: FEN > grid > list > bitboard).
- **Budget parity:** 2048 tokens for all models; the DeepSeek thinking
  arm runs at a larger budget and is reported as a separate arm because
  its thinking mode does not complete within 2048 (measured).
- **Kaggle for long runs, local for tests**, single worker (the gateway
  serializes per key).
