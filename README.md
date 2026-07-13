# Cross-Lingual Spatial Navigation Reasoning in On-Device SLMs

**Target Venue**: Efficient and On-Device AI Agents Workshop @ NeurIPS 2026
**Deadline**: August 29, 2026
**Submission**: OpenReview, double-blind, non-archival
**Format**: Short paper (4p + refs) or Long paper (9p + refs), NeurIPS template

---

## Current Status (July 13, 2026)

Research direction, not results yet. Full pipeline docs are in `idea.md`, `lit-review.md`,
`novelty-assessment.md`, `refined-idea.md`, and `experiment-plan.md` — this README is an
orientation summary, those files have the real detail.

- ✅ Literature review (34 papers) and novelty check complete — score 6/10
- ✅ Experiment code written (`train.py` + `hf_models.py` + `multilingual_data.py`), supports
  both a Modal/cloud backend and a local Ollama backend
- ✅ 9-language instruction templates translated, back-translated, and confidence-tiered
  (`data/multilingual/`)
- 🔄 First real pilot run (Gemma 4 E2B, 20 GridRoute tasks × 10 languages) in progress —
  one earlier attempt completed but produced invalid data (Ollama wasn't running on the
  machine it executed on), so there is **no experimental data yet**
- ⏳ Qwen2.5-1.5B/3B comparison runs, full-scale run, data-availability correlation, and the
  translate-first mitigation test are all still to come

## This Project Pivoted Mid-Stream

The project originally targeted a different idea: a small language model (Gemma 4 E2B) would
extract start/goal coordinates from natural-language navigation instructions, and a classical A*
solver would compute the path. A critical review of that approach found it near-tautological — the
solver always ran against the ground-truth grid, the LLM's extracted "obstacles" were never
actually used, and the "100% optimal" result was mathematically guaranteed by A*'s own correctness
proof whenever coordinate extraction succeeded, not evidence of spatial reasoning.

The docs, code, and benchmark results from that direction are preserved (not deleted) under
[`archive/old-direction/`](archive/old-direction/) for the record.

## Current Research Question

Does spatial *navigation* reasoning (path-finding, obstacle avoidance — not static scene
question-answering) degrade differently across languages in on-device-scale SLMs (1-4B
parameters), and if so:

1. Does the size of the gap correlate quantitatively with each language's pretraining-data
   availability?
2. Does a cheap, training-free mitigation — prompting the model to translate the instruction to
   English internally before solving — close any of the gap, and does that depend on model
   architecture?

The motivating data point: MazeEval (arXiv:2507.20395) found a model solves mazes 3-4 sizes
smaller when the identical task is posed in Icelandic instead of English. Nobody has followed
that up systematically. The closest existing benchmark, MentalMap (arXiv:2605.28277), tests
static-scene spatial question-answering on mostly 7B+ models — this project targets navigation
tasks specifically, with on-device SLMs as the primary subject rather than a boundary case, adds a
quantitative training-data-availability correlation, and tests a mitigation MentalMap doesn't.
Full comparator analysis in `novelty-assessment.md`.

---

## Project Structure

```
neuro-symbolic-pathfinding/
├── README.md                    # This file
├── idea.md                      # Research idea (v3, current direction)
├── lit-review.md                # 34-paper literature review
├── novelty-assessment.md        # Novelty scoring vs. closest comparators
├── refined-idea.md              # Buildable spec: approach, risks, evaluation plan
├── experiment-plan.md           # Current experiment's plan (dataset, metrics, ablations)
├── requirements.txt             # Python dependencies (Modal/cloud backend)
├── train.py                     # Experiment entry point (Modal or --backend ollama)
├── hf_models.py                 # Model wrapper: HF transformers backend + Ollama backend
├── multilingual_data.py         # Task sampling + multilingual template filling
├── src/
│   ├── astar_solver.py          # A* pathfinding (ground truth / optimal path scoring)
│   ├── grid_generator.py        # GridRoute map generation + NL instruction templates
│   ├── ollama_env.py            # Ollama API wrapper (thinking-tag handling, etc.)
│   └── evaluation.py            # Metrics: compliance/feasibility/optimal ratio, VMR
├── data/
│   ├── gridroute/                # GridRoute task data
│   ├── lost_in_aggregation/      # Lost in Aggregation maze corpus
│   └── multilingual/             # Translated instruction templates (9 languages + English)
├── docs/
│   ├── workshop_info.md          # Target venue CFP details
│   └── benchmark_details.md      # GridRoute / Lost in Aggregation benchmark specs
├── scripts/
│   └── download_mazes.sh         # Fetch Lost in Aggregation maze data
├── .rstack/                      # Research pipeline plumbing (lit review records, etc.)
└── archive/old-direction/        # Preserved docs + results from the superseded direction
```

---

## Installation

### Option A — Local, via Ollama (works on a 6GB laptop GPU)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt   # only needed for the Modal/HF path; Ollama path just needs `requests`

curl -fsSL https://ollama.com/install.sh | sh   # if not already installed
ollama serve &
ollama pull gemma4-e2b:q3_k_s
ollama pull qwen2.5:1.5b
ollama pull qwen2.5:3b
```

### Option B — Cloud, via Modal (full-precision models, needs more VRAM)

```bash
pip install modal
modal token new
modal secret create huggingface-secret HF_TOKEN=<your-hf-token>   # needs Gemma 4 license accepted
```

Modal requires a payment method on file before it will allocate any GPU function, independent of
free credits.

---

## Running Experiments

```bash
# Local, Ollama backend, quick pilot
python3 train.py --n_tasks 20 --models gemma4-e2b --backend ollama --output_dir ./results/pilot_run

# Cloud, Modal, full run (all 3 models)
python3 -m modal run train.py --n-tasks 100

# Local smoke test (no GPU needed, validates the pipeline end-to-end)
python3 train.py --smoke_test --backend ollama --output_dir ./results/smoke_test
```

See `experiment-plan.md` for the full experimental design (dataset, metrics, ablations) and
`refined-idea.md` for the risks and assumptions behind the approach.

---

## Timeline

| Week | Dates | Milestone | Status |
|------|-------|-----------|:------:|
| Week 1 | Jul 9–12 | CUDA/Ollama setup, original direction built and critiqued | ✅ |
| Week 1.5 | Jul 12–13 | Pivot: lit review, novelty check, new experiment code, repo cleanup | ✅ |
| Week 2 | Jul 13–16 | First valid pilot data, scale to full run | 🔄 |
| Week 3 | Jul 16–22 | Data-availability correlation + translate-first mitigation experiments | ⏳ |
| Week 4 | Jul 22–29 | Results analysis, figures, tables | ⏳ |
| Week 5 | Jul 30–Aug 5 | Paper writing (first draft) | ⏳ |
| Week 6 | Aug 6–19 | Revisions, ablations, proofreading | ⏳ |
| **Deadline** | **Aug 29** | **Submit to OpenReview** | 📅 |

---

## Author

**Vedang** — BTech Mathematics and Computing, MIT (Manipal)
Previous: ICML workshop publication
[GitHub](https://github.com/Vedang-P)
