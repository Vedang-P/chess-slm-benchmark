# Benchmark Details

## 1. GridRoute (Li et al., 2025)
- **Paper**: arXiv:2505.24306
- **Code**: github.com/LinChance/GridRoute
- **License**: AGPL 3.0

### Configurations
| Param | Size 5 (used here) | Size 10 | Size 20 | Size 30 |
|-------|---------------------|---------|---------|---------|
| Grid (N) | 5x5 | 10x10 | 20x20 | 30x30 |
| Obstacle size (s) | 1x1 | 3x3 | 4x4 | 5x5 |
| Num obstacles (n) | 1 | 2 | 3 | 4 |

This project doesn't use their published task files directly -- `src/grid_generator.py`'s
`generate_gridroute_maps()` regenerates tasks matching their config parameters
(obstacle size/count scaled per grid size, min start/goal distance = 30% of the grid
diagonal, connectivity verified via BFS, ground truth via Dijkstra), seeded and
reproducible. 5x5 is the size actually used everywhere in this project (matching
MazeBench's 5x5, for a fair cross-format comparison) -- 10x10 is used only as an
optional harder follow-up, not the primary evaluation size.

### Metrics (their names; this project's actual scoring)
- **CR**: Compliance Ratio (format correctness) -- this project's `valid`/`optimal`
  scoring (`src/evaluation.py`) folds compliance into feasibility: a response with no
  parseable path is `no_path`, not partially credited.
- **FR**: Feasibility Ratio (valid, obstacle-free path) -- `valid_rate` here.
- **OR**: Optimal Ratio (matches Dijkstra optimal length) -- `optimal_rate` here.
- GM/MSE/RT (geometric mean of length ratio, mean square error, run time) are not
  currently computed by this project's harness -- valid/optimal rate are the two
  numbers actually reported.

### Baselines (from their paper -- NOT reproduced by this project; see idea.md's Phase 1
notes on which baselines are actually replicated vs. cited as literature context)
| Model | Size | Prompt | CR | FR | OR |
|-------|------|--------|----|----|-----|
| GPT-4 Turbo | — | Vanilla | ~90% | ~70% | ~30% |
| GPT-4 Turbo | — | CoT | ~95% | ~80% | ~40% |
| GPT-4 Turbo | — | AoP-Dijkstra | ~95% | ~85% | ~50% |
| Qwen2.5-7B | 7B | Vanilla | ~60% | ~30% | ~10% |
| Qwen2.5-72B | 72B | AoP-Dijkstra | ~90% | ~70% | ~35% |

No SLM (<=~5B) has been tested on GridRoute anywhere in the literature found so far --
their own tested models are all >=7B. That gap is the actual open question this
project's baselines answer, not a reproduction of the table above (different model
scale entirely, not a fair like-for-like comparison).

### Failure Types
1. Invalid Step Distance (diagonal/illegal moves)
2. Path Through Obstacle
3. Out of Bounds
4. Empty Path (no output)
5. Start/End Mismatch

---

## 2. MazeBench / AlphaMaze (Dao & Vu et al., 2025)
- **Paper**: arXiv:2502.14669 ("AlphaMaze: Enhancing Large Language Models' Spatial
  Intelligence via GRPO")
- **Code**: github.com/menloresearch/visual-thinker (Apache 2.0) -- vendored here as
  the `alphamaze_reference` git submodule, run `git submodule update --init` to fetch it
- **Data**: `Menlo/Maze-Bench-v0.2` on HuggingFace (~100 mazes, `test` split)
- **Checkpoint**: `homebrewltd/AlphaMaze-v0.2-1.5B` (SFT+GRPO on DeepSeek-R1-Distill-Qwen-1.5B)

### Format
Token-based, not natural language -- this is the "other" surface format from
GridRoute's NL coordinates, and the whole point of testing both is that they're
genuinely different representations of the same underlying navigation problem.
Each maze is rendered as a grid of tokens: `<|row-col|>` coordinates (e.g. `<|0-0|>`),
wall tokens per cell (`<|no_wall|>`, `<|up_wall|>`, `<|up_down_wall|>`, etc.),
`<|origin|>`/`<|target|>` markers, and the expected output is a sequence of movement
tokens (`<|up|>`, `<|down|>`, `<|left|>`, `<|right|>`). `src/token_maze.py` implements
this project's own encoder/decoder for the *same* token vocabulary, used for Phase 2's
own training data (see that file's docstring for why it's not a byte-for-byte clone of
AlphaMaze's exact undocumented layout).

### Scoring -- two different notions, don't conflate them
- **Their published number (93%)**: presumed exact move-sequence match against a
  single stored reference solution.
- **Their real scoring code** (`benchmark_maze_solution` in the submodule, used by
  `eval.py` when the submodule is present): re-parses the maze's actual wall
  structure straight out of the prompt text and simulates the candidate's moves
  against those real walls, accepting *any* sequence that actually reaches the
  target -- not just the one stored reference path. This is what `eval.py` actually
  uses (falls back to exact-match, with a loud warning, only if the submodule isn't
  present) -- it's the more faithful metric, and can score a valid-but-different path
  as correct where naive exact-match would wrongly fail it.

### Published Result
AlphaMaze-v0.2-1.5B (DeepSeek-R1-Distill-Qwen-1.5B + their SFT+GRPO recipe): 93% on
MazeBench-v0.2. This project's Phase 1 replication check (`eval.py --model alphamaze
--benchmark mazebench`) exists specifically to confirm this harness reproduces that
number on their own checkpoint before trusting any of this project's own numbers.
