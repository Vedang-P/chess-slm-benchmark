# External resources: what exists to build on (2026-07-31)

Inventory of existing codebases/datasets used or considered by this project.
Lichess-derived data is CC0; third-party repos are MIT/Apache unless noted.

## Adopted (in the repo)

| Resource | What we use | Where |
|---|---|---|
| **python-chess** (niklasf, MIT) | Cross-validation of 8x8 legality + mate detection; FEN parsing | `scripts/test_engine.py` parity tests (runs on Kaggle; skipped if absent) |
| **Lichess puzzle database** (CC0, 6M puzzles) | Real mate-in-1 positions for `mate1-lichess` (288 positions, puzzle id + rating); standard positions for `cap-legal-8x8` | `scripts/build_lichess_mate1.py`, `scripts/build_cap_positions.py`, `data/positions/` |
| **kagisearch/llm-chess-puzzles** (MIT, 1000 lichess puzzles) | Source of standard 8x8 positions; their illegal-move table (gpt-4o 8.9%, claude-3-opus 46.4%) is baseline context for our legality metric | `data/external/kagi_puzzles.csv` |

## Considered, not adopted (why)

| Resource | Why it exists | Why not now |
|---|---|---|
| **ChessBench** (Ruoss et al., arXiv:2402.04494) | ~10M Stockfish-scored positions | Large; score-based rather than legality/mate-oriented; link not public on GitHub |
| **LLM Chess** (arXiv:2512.01992, `maxim-saplin/llm_chess`) | Multi-turn game-play harness vs engines | Full-game play is out of scope; natural base for a future rollout task |
| **Topsakal grid-game simulators** (arXiv:2407.07796) | Tic-Tac-Toe/Connect-4/Gomoku environments | Adding a second solvable game (Connect-4) is a follow-up |
| **OthelloGPT** (Li et al., arXiv:2210.13382) | Synthetic Othello games + world-model probes | Different game; only relevant if the paper grows an internal-state analysis |

## Key prior-work anchors (see the literature review)

- AlphaMaze (2502.14669): SFT+GRPO recipe for SLM game abilities.
- Inverse IFEval (2509.04292) + reversed-performance personas (2504.06460):
  the text-domain "can't follow do-badly instructions" evidence we extend.
- Specification gaming vs chess engine (2502.13295), sandbagging (2406.07358).
- Goal misgeneralization (2105.14111, 2210.01790); reversal curse (2309.12288).
- Gardner's minichess solved (1307.7118): exact-value oracle precedent.
