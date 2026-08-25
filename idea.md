# Research idea: GRPO directly on base Gemma-4-E2B for MATE move selection

## Question
What ceiling does GRPO (with Stockfish-verifiable rewards) reach when applied directly to the vanilla gemma-4-E2B (no SFT) on 2-choice MATE move selection, and how does SFT+GRPO compare? The same 1000-position MATE eval (4 subsets: strategy/noexplain/tactic/both) used for base gemma (58-61%) and deepseek (86-94%) is the fair probe.

## Method sketch
- Base: google/gemma-4-E2B-it, QLoRA on language_model (r32-64 all-linear), fp16 on Kaggle T4 (16GB, 30h/week).
- RL: GRPO (and variants DAPO/Dr.GRPO) with Stockfish d12 verifier. Rewards: outcome (sparse binary vs dense cp-delta), process (per-step legality + eval stability), style (brevity gated on correctness). Group size G=4-8, 2-5k position pool difficulty-gated (|gap|), no engine at inference.
- Baselines: base gemma, caveman SFT (55.4%), SFT+GRPO, GRPO-only. Ablations: reward choices, PPO vs GRPO vs DPO, pool size, LoRA rank/lr.

## Domain
Chess SLM RLVR; verifiable rewards; small-model RL stability; tokens-per-correct efficiency.
