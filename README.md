# Neuro-Symbolic Pathfinding
## Gemma 4 + A* for On-Device Navigation Agents

**Target Venue**: Efficient and On-Device AI Agents Workshop @ NeurIPS 2026  
**Deadline**: August 29, 2026 (AoE)  
**Submission**: OpenReview, double-blind, non-archival  
**Format**: Short paper (4p + refs) or Long paper (9p + refs), NeurIPS template

### Abstract (Draft)

We propose a neuro-symbolic navigation agent where Gemma 4 E2B (2.3B effective parameters) parses natural-language navigation goals into structured spatial constraints via native function calling, delegates optimal path computation to A*, and handles dynamic replanning through iterative constraint updates. Running entirely on a consumer laptop GPU (NVIDIA RTX 4050, 6GB VRAM), our system achieves near-optimal paths while consuming 10-100x fewer generated tokens than pure language model planners. Evaluated on the GridRoute benchmark (Li et al., 2025) and Lost in Aggregation (Jiang et al., 2026), Gemma 4 + A* matches or exceeds 70B+ parameter LLM planners on path optimality while reducing end-to-end latency to 300-600ms per query.

### Novelty

1. **First to combine Gemma 4's native function calling** with classical A* for on-device pathfinding
2. **NL -> structured constraint extraction** rather than NL -> path generation (SLM does what it's good at: language; A* does what it's good at: search)
3. **First SLM (<3B**) evaluated on GridRoute and Lost in Aggregation benchmarks
4. **Validates the key finding** of Lost in Aggregation (2026): LLM + deterministic algorithm hybrid is the winning strategy for spatial reasoning

### Key References

| Paper | Focus | Relevance |
|-------|-------|-----------|
| Gemma 4 Technical Report (2026) | Model architecture | Primary model (Gemma Team, arXiv:2607.02770) |
| Lost in Aggregation (Jiang et al., 2026) | LLM spatial reasoning benchmark | Evaluation dataset + validates hybrid approach |
| GridRoute (Li et al., 2025) | LLM route planning benchmark | Primary evaluation benchmark |
| DUPLEX (Hua et al., 2026) | LLM as structured extractor + symbolic planner | Methodological inspiration |
| SmallPlan (Pham et al., 2025) | SLMs for path planning | Contemporary related work |
| LLM-BabyBench (Choukrani et al., 2025) | Grounded planning benchmark | Supplementary evaluation |
| Grid2Guide (Haque et al., 2025) | A* + SLM for navigation | Inverse approach (A*->SLM NL generation) |

### Architecture

```
User NL Query ("Go from A to B, avoid the red zone")
    |
    v
[Gemma 4 E2B - NL Parser]  <-- Function calling + thinking mode
    | structured JSON {start, goal, obstacles, constraints}
    v
[Constraint Validator]  <-- Schema validation
    |
    v
[A* Solver]  <-- Classical optimal search
    | optimal path (or failure)
    v
[Gemma 4 - Replanner]  <-- Handles constraint updates / replanning
    |
    v
User gets structured path + natural language description
```

### Hardware

- **GPU**: NVIDIA RTX 4050 Laptop (6GB VRAM)
- **VRAM budget**: ~3.1 GB (Gemma 4 Q4_K_M: 2.5GB, A* + env: <0.6GB)
- **Inference**: llama.cpp + HuggingFace Transformers

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For Gemma 4 with HuggingFace
pip install -U transformers torch accelerate bitsandbytes

# For llama.cpp (optional, for GGUF quantized inference)
# Build from source: https://github.com/ggerganov/llama.cpp
```

### Project Structure

```
neuro-symbolic-pathfinding/
├── README.md                    # Project overview (this file)
├── src/                         # Source code
│   ├── gemma4_env.py            # Gemma 4 model wrapper (function calling, thinking mode)
│   ├── grid_generator.py        # Grid world generation
│   ├── astar_solver.py          # A* pathfinding implementation
│   ├── baselines.py             # Pure SLM planner, AoP/A*, Grid2Guide
│   ├── evaluation.py            # Evaluation metrics (CR, FR, OR, GM, MSE, RT)
│   ├── prompts.py               # Prompt templates for all baselines
│   ├── neuro_symbolic_pipeline.py  # Main pipeline: NL -> JSON -> A* -> path
│   └── gridroute_runner.py      # GridRoute benchmark runner for local models
├── notebooks/                   # Jupyter notebooks
│   ├── 01_gemma4_setup.ipynb    # Model loading, quantization, VRAM benchmarking
│   ├── 02_gridroute_baselines.ipynb  # Run baselines on GridRoute
│   ├── 03_neuro_symbolic.ipynb  # Full neuro-symbolic pipeline
│   └── 04_analysis.ipynb        # Results analysis and visualization
├── data/
│   ├── gridroute/               # Generated GridRoute maps
│   ├── mazes/                   # Lost in Aggregation maze corpus
│   └── results/                 # Experiment outputs
├── configs/
│   └── experiments.yaml         # Experiment configurations
├── docs/
│   ├── research_notes.md        # Research direction and ideas
│   ├── literature_review.md     # Comprehensive literature review
│   └── workshop_info.md         # Workshop details, deadlines, requirements
├── paper/
│   ├── outline.md               # Paper outline
│   ├── notes.md                 # Paper writing notes
│   └── references.bib           # Bibliography
├── requirements.txt
└── Makefile
```

### Quick Start

```bash
# 1. Clone and setup
git clone <this-repo>
cd neuro-symbolic-pathfinding
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Download Lost in Aggregation mazes
wget -P data/mazes/ https://github.com/YuhanJiang415/lost-in-aggregation/releases/download/v0.1/mazes_s3.json
wget -P data/mazes/ https://github.com/YuhanJiang415/lost-in-aggregation/releases/download/v0.1/mazes_s5.json
wget -P data/mazes/ https://github.com/YuhanJiang415/lost-in-aggregation/releases/download/v0.1/mazes_s7.json
wget -P data/mazes/ https://github.com/YuhanJiang415/lost-in-aggregation/releases/download/v0.1/mazes_s10.json

# 3. Generate GridRoute data
python -m src.grid_generator

# 4. Run baselines
python -m src.baselines

# 5. Run neuro-symbolic pipeline
python -m src.neuro_symbolic_pipeline

# 6. Evaluate
python -m src.evaluation
```

### Evaluation Metrics

| Metric | Source | Definition |
|--------|--------|------------|
| CR | GridRoute | Compliance Ratio - output format correctness |
| FR | GridRoute | Feasibility Ratio - valid obstacle-free path |
| OR | GridRoute | Optimal Ratio - matches Dijkstra optimal length |
| GM | GridRoute | Geometric Mean - path length ratio to optimal |
| MSE | GridRoute | Mean Square Error - squared length difference |
| SR | Lost in Aggregation | Success Rate - reached goal via legal path |
| VMR | Lost in Aggregation | Valid Move Ratio - grid-legal moves / total |
| Tokens | Custom | Total generated tokens per query |
| Latency | Custom | End-to-end wall time (SLM inference + A*) |
| VRAM | Custom | Peak GPU memory during inference |

### Experiment Matrix

| System | Model | Dataset | Metrics |
|--------|-------|---------|---------|
| Pure SLM Planner | Gemma 4 E2B | GridRoute | CR, FR, OR, GM, MSE, RT |
| Pure SLM Planner | Gemma 4 E2B | Lost in Agg. | SR, VMR |
| AoP (A* prompt) | Gemma 4 E2B | GridRoute | CR, FR, OR, GM, MSE, RT |
| **Gemma 4 + A*** | Gemma 4 E2B | GridRoute | All + Tokens, Latency, VRAM |
| **Gemma 4 + A*** | Gemma 4 E2B | Lost in Agg. | All + Tokens, Latency, VRAM |
| Ablation (no FC) | Gemma 4 E2B | GridRoute | All |
| Ablation (no thinking) | Gemma 4 E2B | GridRoute | All |
| GPT-4 Turbo *cited* | GridRoute paper | GridRoute | CR, FR, OR, GM, MSE |
| Qwen2.5-7B–72B *cited* | GridRoute paper | GridRoute | CR, FR, OR, GM, MSE |
| GPT-4o *cited* | Lost in Agg. paper | Lost in Agg. | SR, VMR |

### Timeline

| Week | Date | Milestone |
|------|------|-----------|
| Week 1 | Jul 9–15 | Gemma 4 setup, VRAM benchmarking, GridRoute data generation |
| Week 2 | Jul 16–22 | Pure SLM baselines + AoP baselines on GridRoute |
| Week 3 | Jul 23–29 | Neuro-symbolic pipeline implementation + GridRoute eval |
| Week 4 | Jul 30–Aug 5 | Lost in Aggregation experiments + ablations |
| Week 5 | Aug 6–12 | Results analysis, figures, tables |
| Week 6 | Aug 13–19 | Paper writing (first draft) |
| Week 7 | Aug 20–26 | Paper revision, formatting, proofreading |
| Deadline | Aug 29 | Submit to OpenReview |

### Author

- **Vedang** — BTech Mathematics and Computing, MIT (Manipal)
- Previous: ICML workshop publication
- Contact: [to be added]
