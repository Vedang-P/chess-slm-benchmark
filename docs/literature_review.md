# Comprehensive Literature Review

## Neuro-Symbolic Pathfinding: Gemma 4 + A* for On-Device Navigation Agents

---

## Primary Model: Gemma 4 E2B (April 2026)

**Gemma 4 Technical Report** (Gemma Team, arXiv:2607.02770, July 2026)
- Family: E2B (2.3B eff.), E4B (4.5B), 26B A4B (MoE), 31B (Dense)
- **Key features**: Native function calling, thinking mode, structured JSON output, 128K context window
- On-device: <1.5GB VRAM via LiteRT-LM 2/4-bit quantization
- GFN4: AIME 2026 = 37.5% (no tools), MMLU-Pro = 60.0%, GPQA Diamond = 43.4%
- Apache 2.0 license, HuggingFace: `google/gemma-4-E2B-it`

---

## Phase 1: Classic LLM + Classical Planner Hybrids (2023-2024)

### LLM+P (Liu et al., 2023) — arXiv:2304.11477
- **Contribution**: LLM translates NL planning problems to PDDL, then classical planner (Fast Downward) solves. LLM translates plan back to NL.
- **Key finding**: LLMs alone = 0% feasible on long-horizon planning. LLM+P provides optimal solutions.
- **Limitation**: Requires GPT-4; PDDL domain files written by humans; no on-device consideration.

### ReAct (Yao et al., ICLR 2023)
- **Contribution**: Interleaves reasoning traces (Thought) with actions (Action) and environment observations.
- **Key finding**: Thought-Action-Observation loop outperforms imitation learning by 34% on ALFWorld.
- **Limitation**: Uses PaLM-540B; SLM variant unexplored for planning.

### SayCan (Ahn et al., 2022) — arXiv:2204.01691
- **Contribution**: LLM affordance probabilities + learned value functions for grounding.
- **Key finding**: 84% plan success on 101 real-world tasks with PaLM (540B).
- **Limitation**: Requires PaLM-scale model; not on-device feasible.

### Plan-and-Solve (Wang et al., ACL 2023)
- **Contribution**: Adds planning step to zero-shot CoT. "Let's first understand and devise a plan... then carry it out."
- **Key finding**: Outperforms zero-shot-CoT across 10 reasoning datasets.
- **Relevance**: Prompt design inspiration for SLM decomposition.

### NSP (English et al., ICMLA 2024) — arXiv:2409.06859
- **Contribution**: LLM (GPT-4/3.5) crafts symbolic environment representation + generates path planning algorithm code. Python interpreter executes with feedback loop.
- **Key finding**: 90.1% valid paths, 19-77% shorter than pure neural.
- **Limitation**: Uses GPT-4/3.5 not SLMs; no latency benchmarking.

---

## Phase 2: SLM + Planning Hybrids (2025)

### SmallPlan (Pham et al., 2025) — arXiv:2505.00831
- **Contribution**: SLMs (Qwen-2.5-3B, Phi-4-mini) trained via GPT-4o teacher distillation. SFT + RL interleaved. Uses scene graphs for 3D navigation.
- **Key finding**: Fine-tuned 3B SLMs ~competitive with GPT-4o on path planning. RL reduces travel distance/trials.
- **Limitation**: Requires heavy SFT+RL training; scene graph-based not pure grid; LLM IS the planner.

### Gideon (2025) — arXiv:2505.08492
- **Contribution**: Qwen-2.5 1.5B for PDDL planning. Novel problem generator for scalable training data. On-device local LLM.
- **Key finding**: 66.1% valid plans (single domain). 70.6% multi-domain. 120x smaller than GPT-3 baselines.
- **Limitation**: Training inefficient; lower accuracy than larger models; PDDL domain not grid pathfinding.

### GridRoute (Li et al., 2025) — arXiv:2505.24306
- **Contribution**: Benchmark for LLM route planning on grids (10x10, 20x20, 30x30). Proposes Algorithm of Planning (AoP) prompts (DFS/A*/Dijkstra). Tests 6 LLMs (7B-72B).
- **Key finding**: AoP boosts performance vs vanilla prompting; larger models need less algorithmic guidance; algorithm guidance helps more in complex environments.
- **Open source**: github.com/LinChance/GridRoute
- **Relevance**: PRIMARY evaluation benchmark for our paper.

### LLM-BabyBench (Choukrani et al., 2025) — arXiv:2505.12135
- **Contribution**: 3-task benchmark (Predict, Plan, Decompose) on BabyAI grid world. 16 levels. Text-based interface.
- **Key finding**: LLMs struggle with grounded planning; Decompose task tests NL->subgoal translation.
- **Open source**: github.com/choukrani/llm-babybench; HuggingFace datasets.
- **Relevance**: Decompose task is closest to our NL->structured extraction capability.

### Grid2Guide (Haque et al., 2025) — arXiv:2508.08100
- **Contribution**: A* computes optimal path on binary grid -> SLM converts path steps to natural language.
- **Key finding**: SLM can describe pre-computed paths in NL.
- **Key distinction**: Grid2Guide does A* -> SLM (NL generation). We do SLM (NL parsing) -> A*. Inverse direction. Complementary.

### Collab-RAG (Xu et al., 2025) — arXiv:2504.04915
- **Contribution**: 3B SLM fine-tuned to decompose complex queries into sub-questions for black-box LLM.
- **Key finding**: 3B SLM surpasses frozen 32B LLM at decomposition tasks when fine-tuned.
- **Relevance**: Validates that SLMs can excel at *decomposition* when specialized.

---

## Phase 3: Advanced Neuro-Symbolic Architectures (2025-2026)

### DUPLEX (Hua et al., 2026) — arXiv:2603.23909
- **Contribution**: Dual-system neuro-symbolic architecture. Fast System: lightweight LLM extracts entities/relations from NL -> deterministic PDDL mapping -> symbolic planner. Slow System: high-capacity LLM activated on failure for reflection/repair.
- **Key finding**: 97.5% across 12 IPC domains. Outperforms LLM+P by 25.6%. Core insight: "Restrict LLM to what it's good at — structured semantic grounding — and leave plan synthesis to a symbolic planner."
- **Relevance**: METHODOLOGICAL FOUNDATION. Our approach is DUPLEX for spatial navigation.

### Lost in Aggregation (Jiang et al., 2026) — arXiv:2606.22219
- **Contribution**: 1,050 topology-annotated mazes (7 sizes, 3 difficulty tiers). Three modules: input acquisition, multi-scale representation, hierarchical route planning. Evaluates GPT-4o, DeepSeek-V3, Llama-3.3-70B.
- **Key findings**:
  1. Structured coordinate text is the best input format
  2. End-to-end navigation collapses to near zero by 10x10 (all models)
  3. Isolated Fine/Meso/Macro probes survive at 30-75% far beyond that size
  4. 59% of first errors are Meso (junction choices), 39% Fine (perception), 1% Macro (direction)
  5. Barrier = cross-scale AGGREGATION, not single perceptual deficit
  6. Delegating to deterministic walker + LLM at junctions lifts GPT-4o by 92 points at mid sizes
- **Open source**: yuhanjiang415.github.io/lost-in-aggregation/ (mazes released, code coming soon)
- **Relevance**: PRIMARY validation of our neuro-symbolic approach. Their Module 3 finding that LLM + deterministic algorithm hybrid is the winning strategy directly supports our method.

### PLANNINGBENCH (Zhao et al., 2026) — arXiv:2605.20873
- **Contribution**: Scalable planning data generation framework. 30+ task types. Constraint-driven synthesis.
- **Key finding**: RL on verified planning data improves unseen planning benchmarks.
- **Relevance**: Provides framework for generating additional evaluation data.

### LLM-Flax (Kim & Lee, 2026) — arXiv:2604.26569
- **Contribution**: Neuro-symbolic architecture for robotic task planning. LLM for high-level understanding, symbolic for low-level execution.
- **Key finding**: Neuro-symbolic outperforms pure neural on task planning.

---

## Benchmark Landscape Summary

| Benchmark | Task | Environment | Models Tested | Public? |
|-----------|------|-------------|---------------|---------|
| GridRoute | Route planning | 2D grids (10-30) with obstacles | 7B-72B LLMs | Code + data |
| Lost in Aggregation | Maze navigation | 2D mazes (3-30) tree-structured | GPT-4o, DeepSeek-V3, Llama-3.3-70B | Data only |
| LLM-BabyBench | Planning, decomposition | BabyAI grid world (16 levels) | GPT-4o, Claude 3.7 Sonnet, Qwen3 | Code + data |
| PlanBench | Classical planning | PDDL domains | Various LLMs | Available |
| ProcWorld | Embodied navigation | 3D rooms, partial observability | 15 foundation models | Available |

---

## Gaps Identified (Our Novelty)

1. **NO SLM (<3B)** has been tested on GridRoute or Lost in Aggregation benchmarks
2. **NO work** uses Gemma 4's function calling for NL->structured constraint extraction
3. **NO work** benchmarks neuro-symbolic (SLM parser + A* solver) vs pure SLM planner on the same hardware
4. **NO work** measures token efficiency of neuro-symbolic vs pure-LLM planning approaches
5. **NO work** has implemented the Lost in Aggregation's "Topology-aided junction delegation" suggestion using an SLM
