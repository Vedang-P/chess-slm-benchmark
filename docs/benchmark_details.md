# Benchmark Details

## 1. GridRoute (Li et al., 2025)
- **Paper**: arXiv:2505.24306
- **Code**: github.com/LinChance/GridRoute
- **License**: AGPL 3.0

### Configurations
| Param | Size 10 | Size 20 | Size 30 |
|-------|---------|---------|---------|
| Grid (N) | 10x10 | 20x20 | 30x30 |
| Obstacle size (s) | 3x3 | 4x4 | 5x5 |
| Num obstacles (n) | 2 | 3 | 4 |
| Maps per config | 100 | 100 | 100 |
| Start/end pairs | 5 | 5 | 5 |
| Total tasks | 500 | 500 | 500 |

### Metrics
- **CR**: Compliance Ratio (format correctness)
- **FR**: Feasibility Ratio (valid obstacle-free path)
- **OR**: Optimal Ratio (= Dijkstra optimal)
- **GM**: Geometric Mean (path length ratio)
- **MSE**: Mean Square Error
- **RT**: Run Time

### Baselines (from paper)
| Model | Size | Prompt | CR | FR | OR |
|-------|------|--------|----|----|-----|
| GPT-4 Turbo | — | Vanilla | ~90% | ~70% | ~30% |
| GPT-4 Turbo | — | CoT | ~95% | ~80% | ~40% |
| GPT-4 Turbo | — | AoP-Dijkstra | ~95% | ~85% | ~50% |
| Qwen2.5-7B | 7B | Vanilla | ~60% | ~30% | ~10% |
| Qwen2.5-72B | 72B | AoP-Dijkstra | ~90% | ~70% | ~35% |

### Failure Types
1. Invalid Step Distance (diagonal/illegal moves)
2. Path Through Obstacle
3. Out of Bounds
4. Empty Path (no output)
5. Start/End Mismatch

---

## 2. Lost in Aggregation (Jiang et al., 2026)
- **Paper**: arXiv:2606.22219
- **Data**: yuhanjiang415.github.io/lost-in-aggregation/ (GitHub Releases v0.1)
- **License**: CC BY 4.0

### Maze Corpus
| Effective size | Grid size | Mazes | File |
|---------------|-----------|-------|------|
| 3x3 | 7x7 | 150 | mazes_s3.json |
| 5x5 | 11x11 | 150 | mazes_s5.json |
| 7x7 | 15x15 | 150 | mazes_s7.json |
| 10x10 | 21x21 | 150 | mazes_s10.json |
| 15x15 | 31x31 | 150 | mazes_s15.json |
| 20x20 | 41x41 | 150 | mazes_s20.json |
| 30x30 | 61x61 | 150 | mazes_s30.json |

### Per-Maze Schema
```json
{
  "id": "maze_s7_medium_000",
  "effective_size": 7,
  "grid_size": 15,
  "algorithm": "dfs",
  "seed": 30042,
  "grid": [[1,1,1,...], [1,0,0,...], ...],  // 0=path, 1=wall
  "start": [5, 1],
  "goal": [1, 5],
  "difficulty": "medium",
  "metrics": {
    "junction_count_on_path": ...,
    "dead_end_density": ...,
    "confusion_ratio": ...,
    "shortest_path_length": ...
  },
  "topology": {
    "cell_types": {"r,c": "corridor|corner|junction|dead-end"},
    "passability": {"r,c": {up,down,left,right: bool}},
    "shortest_path": [[r,c], ...],
    "junction_choices": {...},
    "dead_ends": [[r,c], ...]
  }
}
```

### Three Modules
1. **Input Acquisition**: 4 formats (Words, Coordinates, ASCII Map, Image)
2. **Multi-Scale Representation**: Fine/Meso/Macro probes in isolation
3. **Hierarchical Route Planning**: 3 delegation regimes

### Baseline Results (from paper)
| Model | 3x3 SR | 5x5 SR | 7x7 SR | 10x10 SR | 15x15 SR |
|-------|--------|--------|--------|----------|----------|
| GPT-4o | ~100% | ~80% | ~2% | ~0% | ~0% |
| GPT-4o + Topology-aided | — | — | 94% | 80% | 70% |
| DeepSeek-V3 | ~100% | ~50% | ~28% | ~6% | ~0% |
| Llama-3.3-70B | ~60% | ~20% | ~0% | ~0% | ~0% |

---

## 3. LLM-BabyBench (Choukrani et al., 2025)
- **Paper**: arXiv:2505.12135
- **Code**: github.com/choukrani/llm-babybench
- **Data**: huggingface.co/datasets/salem-mbzuai/LLM-BabyBench
- **License**: MIT

### Three Tasks
1. **Predict** (8K rows): Predict final state from action sequence. 16 levels.
2. **Plan** (8K rows): Generate action sequence for navigation subgoal. Validated via environment.
3. **Decompose** (8K rows): Break high-level mission into subgoals.

### Decompose Task Metrics
- **CR** (Comprehension Rate): OmniBot executes LLM subgoals, with additions allowed
- **PR** (Precision Rate): OmniBot executes ONLY LLM subgoals (no additions)
- **ACI** (Assistance Curve Integral): Area under SR(k) curve for allowed additions

### BabyAI Environment
- MiniGrid-based grid world
- Object interactions: keys, doors, boxes, balls
- 16 levels: GoToObj -> GoToLocal -> PutNext -> Synth -> BossLevel
- Partially observable, compositional instructions
