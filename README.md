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

1. **Replicate AlphaMaze** (✅ done) — 88% on MazeBench, confirms GRPO works for token mazes
2. **SFT on GridRoute 5×5** — SFT AlphaMaze on natural-language GridRoute 5×5 tasks
3. **GRPO on GridRoute 5×5** — 1,000 GRPO steps following AlphaMaze's recipe, NL format
4. **Cross-format eval** — test on MazeBench (did NL training break token skill?) AND GridRoute 5×5
5. **Baseline comparison** — same SFT+GRPO on base DeepSeek (no token pre-training)
6. **Scale up** — if 5×5 works, extend to 10×10

## Training Plan (5×5 GridRoute)

| Phase | Time (AlphaMaze) | Time (Base DeepSeek) |
|---|---|---|
| SFT (1 epoch, 400 tasks) | ~2h | ~2h |
| GRPO (1,000 steps × 40s) | ~11h | ~11h |
| Evaluation (both benchmarks) | ~1h | ~1h |
| **Total** | **~14h** | **~14h** |

All on 6GB RTX 4050 with 4-bit QLoRA. No cloud. Two runs = ~28h total.

## Quick Start

```bash
# Install dependencies
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Check what fits on your GPU
python check_finetune_feasibility.py

# Replicate AlphaMaze on MazeBench
python eval_baseline.py

# Train a model on GridRoute (coming next)
python train_grpo.py --model deepseek-r1-distill-qwen-1.5b --n_tasks 400 --max_steps 1000
```

## Timeline

| Week | Dates | Milestone | Status |
|---|---|---|---|
| Week 2 | Jul 13–16 | AlphaMaze replication, feasibility check, baselines | ✅ |
| Week 3 | Jul 16–20 | SFT + GRPO training (both models) | 🔄 |
| Week 4 | Jul 20–23 | Transfer evaluation, failure analysis | ⏳ |
| Week 5 | Jul 23–29 | Results analysis, figures, tables | ⏳ |
| Week 6 | Jul 30–Aug 5 | Paper writing (first draft) | ⏳ |
| Week 7 | Aug 6–19 | Revisions, proofreading | ⏳ |
| **Deadline** | **Aug 29** | **Submit to OpenReview** | 📅 |

## Author
Vedang — BTech Mathematics and Computing, MIT Manipal

