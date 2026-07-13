# Improving Gemma 4's Spatial Reasoning via GRPO Fine-Tuning

**Target Venue**: Efficient and On-Device AI Agents Workshop @ NeurIPS 2026
**Deadline**: August 29, 2026
**Submission**: OpenReview, double-blind, non-archival
**Format**: Short paper (4p + refs) or Long paper (9p + refs), NeurIPS template
**Hardware**: NVIDIA RTX A5000 (24GB VRAM)

---

## The Question

Does GRPO reinforcement fine-tuning improve Gemma 4 E2B's spatial and maze-navigation reasoning —
and does that improvement hold up when the model is tested on a benchmark it wasn't trained on, or
does it only work on the exact task format it saw during training?

AlphaMaze already showed SFT+GRPO takes a small model from ~0% to 93% on maze navigation — but
only for one model (not Gemma), on one maze format, never tested for transfer to a structurally
different benchmark. Separately, GRPO has been shown to generalize better than plain SFT for
spatial reasoning in vision-language models, and reinforcement fine-tuning has been shown to
generalize well *within* a training environment but transfer poorly to genuinely unseen ones for
LLM agents generally. Nobody has tested where a text-only, on-device model + true cross-benchmark
transfer for spatial navigation lands between those two findings. That's what this project measures.

## Approach

1. **Baseline**: evaluate untrained Gemma 4 E2B on two structurally different navigation
   benchmarks — GridRoute (rectangular obstacle grids) and Lost in Aggregation (tree-structured
   mazes with topology annotations).
2. **Fine-tune**: SFT then GRPO on Gemma 4 E2B via Unsloth, adapting AlphaMaze's public training
   recipe and data.
3. **Transfer test**: re-evaluate the fine-tuned model on both benchmarks. The gap between
   in-distribution (GridRoute, the training format) and out-of-distribution (Lost in Aggregation)
   performance — and how that gap changes after fine-tuning — is the central result.

Full detail in `refined-idea.md` (approach, risks, assumptions) and `novelty-assessment.md`
(comparison against the closest existing work).

## Current Status (July 2026)

- ✅ Baseline evaluation harness built (`train.py`) — runs a model across GridRoute and Lost in
  Aggregation, reports valid-path and optimal-path rates
- ✅ Literature review and novelty check complete (score 6/10) — closest comparators (AlphaMaze,
  and two 2026 papers on GRPO spatial-reasoning generalization) explicitly differentiated
- ✅ Fine-tuning feasibility confirmed: Gemma 4 E2B GRPO training needs ~9GB VRAM (Unsloth,
  official support), comfortably within the 24GB A5000
- 🔄 SFT/GRPO training pipeline — building now, following Unsloth's Gemma 4 guide
- ⏳ Transfer evaluation, ablations (SFT-only vs. SFT+GRPO), results, paper

## Project Structure

```
├── idea.md                      # Research idea and motivation
├── lit-review.md                # Literature review
├── novelty-assessment.md        # Comparison against closest existing work
├── refined-idea.md              # Approach, risks, evaluation plan
├── experiment-plan.md           # Current experiment's detailed plan
├── train.py                     # Baseline/fine-tuned model evaluation harness
├── hf_models.py                 # Model wrapper: HF transformers backend + Ollama backend
├── check_finetune_feasibility.py # LoRA/GRPO loading feasibility check
├── src/
│   ├── astar_solver.py          # A* pathfinding (ground-truth optimal path scoring)
│   ├── grid_generator.py        # GridRoute map generation
│   ├── ollama_env.py            # Ollama API wrapper
│   └── evaluation.py            # Metrics: compliance/feasibility/optimal ratio, VMR
├── data/
│   ├── gridroute/                # GridRoute task data
│   └── lost_in_aggregation/      # Lost in Aggregation maze corpus
├── docs/
│   ├── workshop_info.md          # Target venue CFP
│   └── benchmark_details.md      # Benchmark specs
└── scripts/
    └── download_mazes.sh         # Fetch Lost in Aggregation maze data
```

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt   # transformers>=5.13.0 required -- Gemma 4 support needs it
```

For fine-tuning, install [Unsloth](https://unsloth.ai/docs/models/gemma-4/train) following their
Gemma 4 guide.

For local evaluation via Ollama instead of full-precision weights:
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull gemma4:e2b
```

## Running

```bash
# Baseline evaluation, full-precision, local GPU
python3 train.py --n_tasks 100 --models gemma4-e2b --backend hf --output_dir ./results/baseline

# Quantized/lower-VRAM evaluation via Ollama
python3 train.py --n_tasks 100 --models gemma4-e2b --backend ollama --output_dir ./results/baseline

# Smoke test (no GPU needed)
python3 train.py --smoke_test --backend hf
```

## Timeline

| Week | Dates | Milestone | Status |
|------|-------|-----------|:------:|
| Week 2 | Jul 13–16 | Baseline eval, fine-tuning feasibility, novelty check | ✅ |
| Week 2.5 | Jul 16–20 | SFT + GRPO training pipeline built and run | 🔄 |
| Week 3 | Jul 20–23 | Transfer evaluation, ablations | ⏳ |
| Week 4 | Jul 23–29 | Results analysis, figures, tables | ⏳ |
| Week 5 | Jul 30–Aug 5 | Paper writing (first draft) | ⏳ |
| Week 6 | Aug 6–19 | Revisions, proofreading | ⏳ |
| **Deadline** | **Aug 29** | **Submit to OpenReview** | 📅 |

## Author

**Vedang** — BTech Mathematics and Computing, MIT (Manipal)
Previous: ICML workshop publication
[GitHub](https://github.com/Vedang-P)
