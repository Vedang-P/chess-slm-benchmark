# Literature Review: Reasoning "in Lucid" (Compressed/Caveman Style) — How People Approach It

Focused follow-up to `lit-review.md`. Generated 2026-08-09. Scope: work on the
compressed, telegraphic ("lucid"/"caveman") reasoning style — what it is, why
models do it, and the four main ways people elicit/teach it.

## Background

Frontier reasoning models (DeepSeek-R1, o1 family) post-trained with RL drift
into a compressed, telegraphic internal monologue rather than natural-language
prose (documented in DeepSeek-R1, arXiv:2501.12948). The research community has
since taken this "style" seriously as an object of study. Three empirical facts
anchor the literature:

1. **Verbose traces are mostly waste.** Length-correlated analyses show the
   majority of long CoT tokens are filler, repetition, and needless backtracking
   (arXiv:2506.14755, arXiv:2508.09726, arXiv:2605.27965, arXiv:2506.23840).
2. **Length is not the causal factor — content is.** Short and long generations
   of the same model on the same question have the same accuracy; extra tokens
   do not buy extra reasoning (arXiv:2606.30128).
3. **Verbosity is a training artifact, not ability.** Theoretical analysis of
   PPO/GRPO shows length inflation arises from loss minimization on wrong
   answers, not from deeper thinking (arXiv:2504.05185).

So the "lucid" style is plausibly the *de-noised* form of reasoning, and a whole
line of work asks how to elicit or teach it. Five approaches dominate.

## How people approach it

### 1. Prompting (zero-training)
- **Chain of Draft (arXiv:2502.18600)**: instruct the model to write 2-5 token
  drafts per step ("think in compressed form"). ~90% token reduction on some
  tasks with equal or better accuracy. The canonical statement of the idea.
- **Concise CoT (arXiv:2401.05618)**: "be concise" instruction cuts response
  length ~49% at negligible accuracy cost (slight gains on math).
- **Optimizing Length Compression (arXiv:2506.14755)**: prompt constraints
  targeting "invalid thinking" (post-answer double-checking) compress chains.

### 2. RL reward design (the R1-native route)
- **Concise Reasoning via RL (arXiv:2504.05185)**: length-aware reward
  formulations fixing the PPO/GRPO verbosity bias.
- **GFPO "Sample More to Think Less" (arXiv:2508.09726)**: filter sampled groups
  by accuracy+length — concise reasoning emerges from selection alone.
- **Reasoning Shaping (arXiv:2510.09535)**: per-step shaping beats crude
  token-level length penalties (which damage accuracy).
- **The Art of Efficient Reasoning (arXiv:2602.20945)**: survey/analysis of the
  data-reward-optimization space; warns all current methods still cost some
  accuracy and no unified protocol exists.

### 3. Distillation / SFT (the small-model route — directly ours)
- **Long-Short CoT Mixture SFT (arXiv:2505.03469)**: R1-distilled students
  inherit the teacher's verbosity; mixing short+long traces in SFT makes the
  student adaptive. The closest published recipe to "caveman gemma".
- **Concise Reasoning, Big Gains (arXiv:2505.19716)**: prune long teacher
  traces (difficulty-aware) before distilling → concise, accurate students.
- **Self-Training Elicits Concise Reasoning (arXiv:2502.20122)**: models have
  latent conciseness; self-generated concise traces are sufficient teachers —
  no frontier teacher required.
- **Concise distillation backdrop**: teacher-trace SFT beats label-only SFT for
  small models (SuperCorrect arXiv:2410.09008; the 2025 reasoning-distillation
  study arXiv:2508.13037 showing trace *style* shapes student gains — the direct
  prior for our A-vs-B design).

### 4. Latent / sub-linguistic reasoning (the extreme end)
- **Implicit CoT via Knowledge Distillation (arXiv:2311.01460)**: distill
  reasoning into non-linguistic intermediate computation (special tokens).
- **Formal CoT vs Latent Thought comparison (arXiv:2509.25239)**: latent-space
  computation is strictly more expressive per unit budget.
- **Uncovering Latent CoT Vectors (arXiv:2409.14026)**: CoT behavior steered
  from activations; **LLM Reasoning Is Latent (arXiv:2604.15726)**: position
  paper that surface traces are projections, not the reasoning itself;
  **Physics of LM 2.1 (arXiv:2407.20311)** and **Hidden Reasoners
  (arXiv:2411.04282)**: reasoning runs partly below the text.

### 5. Test-time / adaptive compute (complementary levers)
- **Learning to Stop Overthinking (arXiv:2502.10954)**, **Zero-Step Thinking
  (arXiv:2510.19176)**, **SelfBudgeter (arXiv:2505.11274)**, **Correct-Concise-
  Complete multi-stage training (arXiv:2601.02972)**: models learn to spend
  tokens proportional to difficulty — compressed style + adaptive budget.

## Caveats we must carry into the paper

- **Measure on the deployed artifact.** Low-bit quantization itself inflates
  reasoning length (arXiv:2606.25519) — evaluate gemma-4-E2B (quantized), not
  the FP16 training artifact, for tokens-per-correct.
- **Report accuracy AND tokens jointly.** Every efficiency claim in this
  literature is contested unless both are reported (arXiv:2602.20945).
- **Memorization confound** in chess evals remains (arXiv:2601.16823, already in
  the main review).

## References (arXiv IDs)

1. Chain of Draft: Thinking Faster by Writing Less — 2502.18600
2. The Benefits of a Concise Chain of Thought — 2401.05618
3. Self-Training Elicits Concise Reasoning in LLMs — 2502.20122
4. Concise Reasoning via Reinforcement Learning — 2504.05185
5. Sample More to Think Less (GFPO) — 2508.09726
6. Concise Reasoning, Big Gains (difficulty-aware pruning) — 2505.19716
7. Long-Short CoT Mixture SFT — 2505.03469
8. Optimizing Length Compression in Large Reasoning Models — 2506.14755
9. Does Verbose Chain-of-Thought Really Help? — 2606.30128
10. Do Thinking Tokens Help or Trap? — 2506.23840
11. The Shape of Overthinking: Backtracking Bursts — 2605.27965
12. Mitigating Overthinking through Reasoning Shaping — 2510.09535
13. The Art of Efficient Reasoning: Data, Reward, and Optimization — 2602.20945
14. Correct, Concise and Complete: Multi-stage Training — 2601.02972
15. SelfBudgeter: Adaptive Token Allocation — 2505.11274
16. Learning to Stop Overthinking at Test Time — 2502.10954
17. The Zero-Step Thinking — 2510.19176
18. Token-Efficient RL for LLM Reasoning — 2504.20834
19. Implicit Chain of Thought Reasoning via Knowledge Distillation — 2311.01460
20. A Formal Comparison Between CoT and Latent Thought — 2509.25239
21. Uncovering Latent Chain of Thought Vectors — 2409.14026
22. LLM Reasoning Is Latent, Not the Chain of Thought — 2604.15726
23. Physics of Language Models 2.1: Hidden Reasoning — 2407.20311
24. Language Models are Hidden Reasoners — 2411.04282
25. Quantization Inflates Reasoning — 2606.25519
26. Towards Reasoning Era: Survey of Long CoT — 2503.09567

Plus prior records in `.rstack/lit-review.jsonl` (DeepSeek-R1 2501.12948,
SuperCorrect 2410.09008, etc.) and the cavegemma GitHub recipe (hf-cavegemma).
