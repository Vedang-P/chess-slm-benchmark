# Can GRPO Teach Small Language Models Spatial Reasoning in Natural Language?

**Target Venue**: Efficient and On-Device AI Agents Workshop @ NeurIPS 2026
**Deadline**: August 29, 2026
**Hardware**: NVIDIA RTX 4050 Laptop GPU (6GB VRAM)
**Models**: DeepSeek-R1-Distill-Qwen-1.5B, SmolLM2-1.7B


---

## The Question

**Primary**: Can we take AlphaMaze (already GRPO-trained on 5×5 token mazes, 88%) and
GRPO-train it further on natural-language GridRoute 5×5 — teaching it to spatially reason
in BOTH formats without losing its existing skill?

**Secondary**: Does starting from a model that already knows spatial reasoning accelerate
NL-format learning vs training from scratch on the base DeepSeek model?

If cross-format transfer works → single 1.5B model, token AND NL spatial reasoning.
If it doesn't → we quantify format interference and failure modes.

## Current Results

### AlphaMaze Replication (July 14, 2026)
| Benchmark | Format | Accuracy |
|---|---|---|
| MazeBench (5×5) | Token mazes `<|up|><|left|>` | **88%** (paper: 93%) |
| GridRoute (5×5) | NL coords "Navigate from (3,2)..." | **0%** |
| GridRoute (10×10) | NL coords | **0%** |

**Finding**: GRPO-trained spatial reasoning is **format-locked**. The model scores
88% in its native token language but 0% when asked the same task in natural language.

### Baselines
| Model | GridRoute 5×5 | GridRoute 10×10 |
|---|---|---|
| DeepSeek 1.5B (untrained) | 0% | 0% |
| Gemma 4 E2B (4.6B, Ollama) | 0% | — |
| DeepSeek 1.5B + AoT-Dijkstra | 0% | — |

**No model below 7B can do natural-language grid navigation.** The GridRoute paper
confirms this: Qwen2.5-7B gets 64% FR (feasibility ratio) with CoT prompting.

## Approach

### Option 1: AlphaMaze → NL GridRoute 5×5 (local, on-device)
1. Take AlphaMaze-v0.2-1.5B (88% on MazeBench token mazes)
2. SFT on 400 GridRoute 5×5 NL tasks (NL prompt → coordinate path)
3. GRPO on same tasks (1,000 steps, AlphaMaze recipe)
4. Test on BOTH MazeBench AND GridRoute 5×5
5. Compare vs base DeepSeek (same training, no token pre-training)
   - Hardware: RTX 4050 6GB, 4-bit QLoRA, ~14h per model

### Option 2: Gemma 4 E2B LoRA on both benchmarks (Kaggle cloud)
1. LoRA SFT Gemma 4 E2B on MazeBench token mazes + GridRoute 5×5 NL
2. Test on both benchmarks
3. Compare vs AlphaMaze (stronger base model, no GRPO)
   - Hardware: Kaggle T4 16GB free GPU, full precision LoRA
   - Note: GRPO not possible on T4 (needs 9GB+), SFT-only

## Training Plan

| Phase | AlphaMaze (local) | Gemma 4 (Kaggle) |
|---|---|---|
| SFT (1 epoch, 400 tasks) | ~2h | ~1h (T4) |
| GRPO (1,000 steps) | ~11h | ❌ (OOM on T4) |
| Evaluation (both benchmarks) | ~1h | ~30min |
| **Total** | **~14h** | **~2h** |

## Project Log

### July 14, 2026
- **Pivot**: Gemma 4 E2B GRPO infeasible on 6GB (needs 9GB). Switched to 1.5B models.
- **Feasibility confirmed**: DeepSeek 1.5B (4.2GB GRPO), SmolLM2 1.7B (3.0GB) both fit.
- **Dropped Unsloth**: incompatible with torch 2.6 + transformers 5.x. Using bitsandbytes+peft.
- **AlphaMaze-v0.2-1.5B downloaded**: Apache 2.0 weights from Menlo/AlphaMaze-v0.2-1.5B.
- **MazeBench replication**: 88% on easy-mazes-96%, medium-80%, hard-80%. Paper claims 93%.
  - Root cause: max_new_tokens=1024 too short for thinking loops. Fixed to 4096 → all think tags close.
  - Remaining gap (88% vs 93%): likely Unsloth inference vs plain transformers.
- **GridRoute baseline**: AlphaMaze 0% on all GridRoute sizes (format lock confirmed).
  - Same model: 88% on token mazes, 0% on NL coordinate format.
  - Even at 5×5 (same size as MazeBench), 0% — purely format mismatch.
- **Gemma 4 E2B tested**: 0% on GridRoute 5×5 via Ollama Q3_K_S.
- **AoT-Dijkstra tested**: 0% on DeepSeek 1.5B — model too small for algorithmic prompting.
- **Literature review**: GridRoute paper (May 2025) tested 7B-72B models only. Best: GPT-4 at 84% FR.
  Tong et al. (Apr 2026): SFT works for spatial transfer, length scaling fails. RL ≤ SFT.
  AlphaMaze (Feb 2025): SFT+GRPO on 1.5B, 93% on token mazes. Training code public.
- **Research direction set**: Can we GRPO-train AlphaMaze on NL GridRoute 5×5 and preserve
  both token AND NL spatial reasoning? Cross-format transfer learning on-device.

## Quick Start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Check GPU feasibility
python check_finetune_feasibility.py

# Replicate AlphaMaze on MazeBench
python eval_baseline.py

# SFT AlphaMaze on GridRoute 5×5 NL
python train_sft.py --model_path ./data/models/alphamaze-v0.2-1.5b --grid_size 5 --n_tasks 400

# GRPO on GridRoute 5×5 (after SFT)
python train_grpo.py --model deepseek-r1-distill-qwen-1.5b --n_tasks 400 --max_steps 1000
```

## Timeline

| Week | Dates | Milestone | Status |
|---|---|---|---|
| Jul 13–14 | AlphaMaze replication, feasibility, baselines, lit review | ✅ |
| Jul 14–16 | SFT + GRPO AlphaMaze on GridRoute 5×5 NL | 🔄 |
| Jul 16–18 | Cross-format eval, baseline DeepSeek run | ⏳ |
| Jul 18–20 | Gemma 4 LoRA on Kaggle (optional) | ⏳ |
| Jul 20–29 | Results analysis, figures, tables | ⏳ |
| Jul 30–Aug 19 | Paper writing, revisions | ⏳ |
| **Aug 29** | **Submit to OpenReview** | 📅 |
