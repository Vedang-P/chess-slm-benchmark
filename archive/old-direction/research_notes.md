# Research Notes

## Project: Neuro-Symbolic Pathfinding with Gemma 4 + A*

### Date: July 9, 2026

---

## Core Idea

Gemma 4 E2B (2.3B effective parameters, native function calling) parses NL navigation goals into structured constraints (JSON: {start, goal, obstacles}), then A* computes the optimal path. SLM handles language; A* handles search.

### Why This Works
- **DUPLEX insight**: "The key is not to make the LLM plan better, but to restrict the LLM to the part it is good at — structured semantic grounding — and leave logical plan synthesis to a symbolic planner."
- **Lost in Aggregation finding**: LLM + deterministic walker hybrid lifts success by 92 points. We implement this with A* instead of a walker.
- **GridRoute finding**: Algorithmic guidance helps LLMs plan — using the actual algorithm should be strictly better.

### Novelty Checklist
- [x] First to use Gemma 4 function calling for NL->structured pathfinding
- [x] First SLM (<3B) on GridRoute benchmark
- [x] First neuro-symbolic system that validates Lost in Aggregation's Module 3 finding
- [x] Token efficiency analysis (10-100x fewer tokens than pure SLM)
- [ ] Ablation studies (thinking on/off, function calling vs prompt parsing)
- [ ] Replanning experiments (dynamic constraint updates)

---

## Experiment Design

### Primary Experiments
1. **GridRoute (all 3 sizes)**: Pure SLM vs AoP/A* vs Gemma 4+A*
2. **Lost in Aggregation (7 sizes)**: Pure SLM vs Gemma 4+A* on maze navigation
3. **Token efficiency**: Count generated tokens for each method
4. **Latency + VRAM**: Hardware profiling on RTX 4050

### Ablations
1. Function calling ON vs OFF (prompt-based NL parsing)
2. Thinking mode ON vs OFF
3. Different grid sizes (scaling analysis)

### Baselines (Published — cite from papers)
- GridRoute: GPT-4 Turbo, Qwen2.5-7B/32B/72B, DeepSeek-V3, LLaMA3.1-70B
- Lost in Agg: GPT-4o, DeepSeek-V3, Llama-3.3-70B
- SmallPlan: Qwen-2.5-3B, Phi-4-mini

---

## Key Claims to Prove

1. **Correctness**: Gemma 4+A* produces valid, obstacle-free paths at near-optimal lengths
2. **Efficiency**: Uses 10-100x fewer tokens than pure SLM planners
3. **Latency**: End-to-end <500ms on consumer GPU (RTX 4050)
4. **SMALL wins BIG**: 2.3B SLM+A* matches/exceeds 70B+ pure LLM planners
5. **Robustness**: Function calling > prompt-based parsing for structured extraction
6. **Replanning**: Handles dynamic constraint updates correctly

---

## Potential Weaknesses (Address Proactively)

1. **Grid environments are simple** → Acknowledge limitation; show results on both GridRoute AND complex mazes
2. **Gemma 4 was trained on internet data that may include pathfinding examples** → Acknowledge; argue that structured extraction via function calling is qualitatively different from memorization
3. **A* is guaranteed optimal, so comparison is "unfair"** → That's the point — we delegate to what's provably correct instead of making the LLM do unreliable search
4. **Small sample sizes** → Use full GridRoute dataset (100 maps x 5 pairs = 500 tasks per config)
5. **Single model** → Compare multiple model sizes if resources permit

---

## Related Work That Must Be Cited

- [x] Gemma 4 Technical Report (2026)
- [x] GridRoute (Li et al., 2025)
- [x] Lost in Aggregation (Jiang et al., 2026)
- [x] DUPLEX (Hua et al., 2026)
- [x] SmallPlan (Pham et al., 2025)
- [x] Grid2Guide (Haque et al., 2025)
- [x] LLM+P (Liu et al., 2023)
- [x] LLM-BabyBench (Choukrani et al., 2025)
- [x] NSP (English et al., 2024)
- [x] ReAct (Yao et al., 2023)
- [x] PlanBench (Valmeekam et al., 2023)
- [x] Gideon (2025) — arXiv:2505.08492
- [x] PLAHX (Tang et al., 2025) — arXiv:2501.15214
