# Neuro-Symbolic Pathfinding
## Gemma 4 E2B + A\* for On-Device Navigation Agents

**Target Venue**: Efficient and On-Device AI Agents Workshop @ NeurIPS 2026  
**Deadline**: August 29, 2026  
**Submission**: OpenReview, double-blind, non-archival  
**Format**: Short paper (4p + refs) or Long paper (9p + refs), NeurIPS template  
**Hardware**: NVIDIA RTX 4050 Laptop GPU (6GB VRAM), CUDA 580

---

## Current Status (July 10, 2026)

### ✅ Completed
- CUDA-enabled Gemma 4 E2B (q3_k_s, 4.6B params, 2.3GB) via Ollama v0.31.2
- Neuro-symbolic pipeline: Gemma extracts constraints → A* solves → optimal path
- **GridRoute size_10**: **500/500 tasks, 100% valid, 100% optimal, avg 5.6s**
- Pure SLM baseline analysis (Gemma alone): 40% valid at 4096 tokens — fails due to overthinking
- All 3 NL instruction variants handled (direct/descriptive/constrained)
- Pure A* baseline (ground truth): always optimal

### 🔄 In Progress
- GridRoute size_20 (20×20, 3 obstacles): ~200/500 tasks (checkpointed)
- GridRoute size_30 (30×30, 4 obstacles): pending
- Lost in Aggregation (3×3 to 15×15 mazes, ~750 total): pending

### Key Finding
**Small language models (SLMs) cannot do spatial reasoning reliably.**
- `pure_slm`: Gemma generates path directly → 40% valid, 50s avg, often gives up or goes in circles
- `neuro_symbolic`: Gemma extracts constraints + A* solves → **100% valid, 100% optimal, 5.6s avg**
- Gemma does NL understanding (its strength), A* does pathfinding (its strength)

---

## The Problem

A human says: *"Navigate from (3,7) to (7,2), avoid the obstacles."*

| Approach | Can read NL? | Finds valid path? | Time | Verdict |
|----------|:-----------:|:-----------------:|:----:|:-------:|
| **A\* alone** | ❌ | ✅ Optimal | 0ms | Needs explicit coords — impractical |
| **Gemma alone** | ✅ | ❌ 40% valid | 50s | Overthinks, gets confused |
| **Gemma + A\*** | ✅ | ✅ **100% optimal** | **5.6s** | Each does what it's good at |

### Gemma's raw output (pure_slm, 50s, 3420 tokens):
```
**Final Attempt (Using the bottom edge):**
1.  (3, 7)
2.  (4, 7) (Wait! (4, 7) is an obstacle! This path segment is invalid.)

... tries 12 more paths, each hitting an obstacle ...

**Since the provided obstacles make a path impossible ... I cannot provide a valid sequence.**
```
Meanwhile A* finds the optimal path in 0ms.

---

## Architecture

```
User NL Query ("Navigate from (3,7) to (7,2) on a 10x10 grid...")
    |
    v
[Gemma 4 E2B]     ──►  {"start": [3, 7], "goal": [7, 2]}
    |                  (plain-text extraction, not function calling)
    v
[A* Solver]       ──►  [(3,7), (3,6), ..., (7,2)]  (optimal path)
    |
    v
Optimal obstacle-free path returned
```

Two components:
1. **Gemma 4 E2B** (Ollama `/api/generate`): reads NL, outputs start/goal as JSON
2. **A\*** (custom `astar_solver.py`): finds optimal path using the extracted coordinates

---

## Installation

### Prerequisites
- NVIDIA GPU with CUDA 12 (tested on RTX 4050, 6GB)
- Ollama v0.31.2+ with Gemma 4 E2B model

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install Ollama (if not present)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma4-e2b:q3_k_s

# Start Ollama service
ollama serve

# Verify CUDA
ollama run gemma4-e2b:q3_k_s "test"  # Should use GPU
```

### Running Benchmarks

```bash
# Start benchmark (with auto-resume from checkpoints)
source venv/bin/activate
curl -s http://localhost:11434/api/generate -d '{"model":"gemma4-e2b:q3_k_s","prompt":"test","stream":false}'
setsid ./start_bench.sh </dev/null > benchmark_run.log 2>&1 &

# Monitor progress
tail -f benchmark_run.log
```

---

## Project Structure

```
neuro-symbolic-pathfinding/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── run_benchmarks.py            # Main benchmark orchestrator (GridRoute + LiA)
├── run_bench.sh                 # Shell launcher (sources venv, nice -n 19, unbuffered)
├── start_bench.sh               # systemd-run-compatible launcher
├── live.sh                      # Live progress monitor
├── monitor.py                   # Real-time GPU/progress dashboard
├── demo.py                      # End-to-end comparison (all methods, raw outputs)
├── validate.py                  # Pipeline validation on small samples
├── validate2.py                 # pure_slm vs neuro_symbolic comparison
├── validate3.py                 # Token budget + NL variant tests
├── src/
│   ├── ollama_env.py            # Ollama API wrapper (generate + chat, thinking tag handling)
│   ├── gemma4_env.py            # Abstract Gemma 4 interface
│   ├── grid_generator.py        # GridRoute map generation + NL variants
│   ├── astar_solver.py          # A* pathfinding implementation
│   ├── baselines.py             # pure_slm, pure_astar planners
│   ├── neuro_symbolic_pipeline.py  # NL extraction → A* → path
│   ├── evaluation.py            # Metrics (CR, FR, OR, GM, MSE)
│   └── prompts.py               # Prompt templates
├── data/
│   ├── lost_in_aggregation/     # LiA maze corpus (symlinked)
│   └── results/
│       └── gemma4-e2b_q3_k_s/   # Benchmark output (JSON per config)
└── .opencode/                   # opencode AI configuration
```

---

## Results So Far

### GridRoute size_10 (10×10 grid, 18 obstacle cells)

| Metric | neuro_symbolic | pure_astar | pure_slm (4096 tok) |
|--------|:-------------:|:----------:|:-------------------:|
| Valid paths | **500/500 (100%)** | 500/500 (100%) | 40% |
| Optimal paths | **500/500 (100%)** | 500/500 (100%) | 20% |
| Avg time | **5.6s** | 0ms | 27.2s |
| Failures | **0** | 0 | 6/10 (overthinking, truncation) |

### NL Variant Robustness

| Variant | Example | Success | Avg time |
|---------|---------|:-------:|:--------:|
| Direct | `"Navigate from (3,7) to (7,2)..."` | **100%** | 3.5s |
| Descriptive | `"Find a path from the start marker at (3,7)..."` | **100%** | 5.0s |
| Constrained | `"Plan route from A=(3,7) to B=(7,2)..."` | **67%** | 14.2s |

---

## Evaluation Metrics

| Metric | Definition |
|--------|-----------|
| CR (Compliance) | Output format is valid path |
| FR (Feasibility) | Path is obstacle-free, in-bounds, 4-directional |
| OR (Optimal) | Path length matches A* optimal |
| GM (Geometric Mean) | exp(mean(log(path_len) - log(optimal_len))) |
| MSE (Mean Square Error) | mean((path_len - optimal_len)²) |

---

## Inference Details

- **Model**: Gemma 4 E2B Q3_K_S (4.6B params, 2.3GB GGUF)
- **Backend**: Ollama v0.31.2 → llama-server (CUDA build b9878)
- **GPU**: RTX 4050, 6GB VRAM, ~90% utilization during inference
- **VRAM**: ~1.36 GiB steady (model + context)
- **Speed**: ~73 tokens/s (Gemma thinking mode), ~450+ tokens/s (direct mode)
- **Temperature**: 0.0 (deterministic), Top-p: 0.95

### Thinking Tag Handling
Gemma 4 E2B outputs reasoning between `<|channel>thought` and `<channel|>` markers.
`strip_gemma_thoughts()` extracts content after the last `<channel|>` marker.

---

## Timeline

| Week | Dates | Milestone | Status |
|------|-------|-----------|:------:|
| Week 1 | Jul 9–12 | CUDA setup, Ollama, Gemma 4 inference | ✅ |
| Week 2 | Jul 12–15 | GridRoute benchmarks (size_10/20/30) | 🔄 |
| Week 3 | Jul 16–22 | Lost in Aggregation experiments | ⏳ |
| Week 4 | Jul 23–29 | Results analysis, figures, tables | ⏳ |
| Week 5 | Jul 30–Aug 5 | Paper writing (first draft) | ⏳ |
| Week 6 | Aug 6–19 | Revisions, ablations, proofreading | ⏳ |
| **Deadline** | **Aug 29** | **Submit to OpenReview** | 📅 |

---

## Author

**Vedang** — BTech Mathematics and Computing, MIT (Manipal)  
Previous: ICML workshop publication  
[GitHub](https://github.com/Vedang-P)
