# Research idea: Teaching SLMs to reason in compressed ("lucid") style via chess

## Question
Can a small language model (gemma 4 E2B, 2B) acquire stronger reasoning ability
by learning to reason in the compressed, telegraphic ("lucid/caveman") style
that RL-post-trained frontier models (DeepSeek) natively use, compared to
standard natural-language chain-of-thought?

## Method sketch
- Teacher: deepseek-v4-flash generates natural (already-lucid) reasoning traces
  on MATE chess positions.
- Student: gemma 4 E2B, SFT in two arms: (1) labels only (B), (2) trace+answer
  distillation (A). Compare MATE 4-subset accuracy AND tokens-per-correct.
- Optionally start from an existing gemma reasoning fine-tune instead of base.

## Domain
Chess (MATE move-selection benchmark) as a cheap, verifiable empirical testbed.
SFT only; no RL (compute-constrained: Kaggle T4 free tier).
