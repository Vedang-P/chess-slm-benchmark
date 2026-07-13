# Paper Outline

## Neuro-Symbolic On-Device Navigation: Gemma 4 + A* for Efficient Pathfinding

**Target**: Efficient and On-Device AI Agents Workshop @ NeurIPS 2026
**Length**: 4 pages + references (short paper) or 9 pages + references (long paper)

---

## 1. Introduction (~0.5 page)

- Hook: LLMs fail at spatial navigation beyond small grids (cite Lost in Aggregation)
- Problem: Pure LLM planning is token-inefficient, unreliable, and requires cloud-scale models
- However: 2.3B SLMs (Gemma 4) now support native function calling + structured output on-device
- Opportunity: Restrict SLM to what it's good at (NL→structured extraction), delegate search to A*
- Contributions:
  1. First neuro-symbolic on-device pathfinding system with Gemma 4 + A*
  2. First SLM (<3B) evaluation on GridRoute and Lost in Aggregation benchmarks
  3. 10-100x token efficiency over pure SLM planners while matching optimal paths
  4. Ablation studies on function calling, thinking mode, and scaling

---

## 2. Related Work (~0.5 page)

### LLM + Classical Planner Hybrids
- LLM+P (2023), SayCan (2022), NSP (2024): Use large LLMs + symbolic planners
- Limitation: Cloud-scale models, not on-device

### SLMs for Planning
- SmallPlan (2025): SLMs via LLM distillation, but SLM is still THE planner
- Gideon (2025): 1.5B SLM for PDDL, but lower accuracy

### Neuro-Symbolic Architectures
- DUPLEX (2026): LLM as schema-guided IE → deterministic mapping → symbolic planner. 97.5% success. Our approach for spatial domain.
- Lost in Aggregation (2026): LLM + deterministic walker hybrid lifts success by 92 pts. We implement this insight.

### Benchmarks
- GridRoute (2025): Standard route planning benchmark
- Lost in Aggregation (2026): Multi-scale spatial reasoning benchmark

### Distinction
Grid2Guide (2025) is the inverse: A* → SLM for NL generation. We do SLM (NL parsing) → A*.

---

## 3. Method (~1 page)

### 3.1 Architecture
```
User NL -> Gemma 4 (function calling) -> JSON {start, goal, obstacles} -> A* -> Path
```

### 3.2 Gemma 4 NL Parser
- Model: Gemma 4 E2B (2.3B effective, 5.1B total with embeddings)
- Configuration: 4-bit quantization, thinking mode enabled
- Function calling with structured JSON schema for constraint extraction
- Fallback: prompt-based parsing when function calling unavailable

### 3.3 A* Solver
- Standard A* with 4-directional cardinal movement
- Manhattan heuristic
- Guaranteed optimal if grid connected

### 3.4 Replanning
- On constraint update, Gemma 4 parses diff → A* re-runs from current position
- Incremental: only re-parses the update, not the full problem

### 3.5 Baselines
1. Pure SLM: Gemma 4 generates path directly
2. AoP/A*: Gemma 4 with GridRoute Algorithm of Planning prompt
3. Published baselines: GPT-4 Turbo, Qwen2.5-7B–72B, DeepSeek-V3 (from GridRoute paper)

---

## 4. Experiments (~1.5 pages)

### 4.1 Setup
- Hardware: NVIDIA RTX 4050 Laptop (6GB VRAM)
- Datasets: GridRoute (500 tasks per size × 3 sizes), Lost in Aggregation (150 mazes × 7 sizes)
- Metrics: CR, FR, OR, GM, MSE (GridRoute); SR, VMR (Lost in Agg)

### 4.2 GridRoute Results
Table: 3 sizes × 3 methods × 6 metrics
Key result: Neuro-symbolic achieves near-perfect OR where pure SLM drops to <10% at size 30

### 4.3 Lost in Aggregation Results
Table: 7 sizes × 2 methods × SR/VMR
Key result: Neuro-symbolic achieves 100% SR at all sizes (A* is guaranteed optimal on connected grids)

### 4.4 Token Efficiency
Bar chart: tokens per query for each method
Key result: 10-100x fewer tokens than pure SLM planner

### 4.5 Latency and VRAM
Table: latency breakdown (SLM inference, A*, total), VRAM usage
Key result: <500ms end-to-end, <3.5GB VRAM

### 4.6 Ablations
- Function calling ON vs OFF: X% accuracy drop without function calling
- Thinking mode ON vs OFF: Y% accuracy drop without thinking
- Scaling: performance across grid sizes

---

## 5. Discussion and Limitations (~0.5 page)

### Key Takeaways
1. 2.3B SLM + A* matches/exceeds 70B+ pure LLM planners on path quality
2. Function calling is critical for reliable structured extraction
3. Validates Lost in Aggregation's finding that hybrid approaches are the winning strategy
4. On-device feasibility demonstrated (RTX 4050, <3.5GB VRAM)

### Limitations
- Grid environments only (no 3D, no dynamic obstacles beyond replanning)
- Single model tested (Gemma 4 E2B)
- A* requires full grid knowledge (not partially observable)
- Gemma 4 may have seen pathfinding examples in training

### Future Work
- Extend to partially observable settings (SLAM + A*)
- Test with other SLMs (Phi-4-mini, Qwen3-4B)
- Real-world drone/robot navigation
- Multi-agent coordination

---

## 6. Conclusion (~0.25 page)

Neuro-symbolic approach where a 2.3B SLM handles language understanding and A* handles optimal search demonstrates that on-device spatial reasoning agents are feasible today, using 10-100x fewer tokens than pure LLM alternatives while guaranteeing path optimality.

---

## Key Figures/Tables Needed

1. **Architecture diagram**: NL -> Gemma 4 -> JSON -> A* -> Path
2. **GridRoute results table**: 3 sizes × 3 methods, 6 metrics
3. **Lost in Aggregation results**: SR vs maze size, all methods
4. **Token efficiency bar chart**: tokens per query by method
5. **Latency breakdown chart**: SLM inference vs A* time
6. **VRAM usage chart**: memory by component
7. **Ablation results table**: FC on/off, thinking on/off, scaling
8. **Failure analysis**: types of errors per method

---

## Appendix

- Full prompt templates
- Implementation details
- Additional figures
