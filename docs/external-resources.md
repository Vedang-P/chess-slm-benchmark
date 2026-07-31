# External resources: what exists to build on (2026-07-31)

Inventory of existing codebases/datasets relevant to the anti-goal benchmark,
with adoption status. Everything lichess-derived is CC0; the repos below are
MIT/Apache unless noted -- check each before redistribution.

## Adopted (in the repo)

| Resource | What we use | Where |
|---|---|---|
| **python-chess** (niklasf, MIT) | Cross-validation of 8x8 legality + mate detection; FEN parsing | `scripts/test_engine.py` parity test (runs on Kaggle; skipped if absent) |
| **Lichess puzzle database** (CC0, `lichess_db_puzzle.csv.zst`, 6M puzzles) | Real mate-in-1 positions for the `mate1-lichess` task: stream-filtered `mateIn1` theme, presented-position convention verified, re-validated by our engine | `scripts/build_lichess_mate1.py`, `data/positions/mate1-lichess.json` (288 positions with puzzle id + rating provenance) |
| **kagisearch/llm-chess-puzzles** (MIT, 1000 lichess puzzles) | Reference: their illegal-move-rate table (gpt-4o 8.9%, claude-3-opus 46.4%) is the natural baseline context for our legality metric; the 1000-puzzle CSV is a potential alternative external set | `data/external/kagi_puzzles.csv` |

## Candidates (not adopted — documented for follow-up)

| Resource | Why it exists | Why not now |
|---|---|---|
| **ChessBench** (Ruoss et al., arXiv:2402.04494) | ~10M Stockfish-scored positions, planning/memorization-futile | Large dataset, score-based (not legality/mate-oriented); contact authors for the download link (repo not on GitHub under that name) |
| **LLM Chess** (Kolasani & Saplin, arXiv:2512.01992; `maxim-saplin/llm_chess`) | Multi-turn game-play harness vs engines, legality/completion metrics | Full-game play scoped OUT of this paper (SLM skill floor); the harness is the natural base for a future rollout task |
| **Topsakal grid-game simulators** (arXiv:2407.07796) | Tic-Tac-Toe / Connect-4 / Gomoku game environments | Adding a second game (Connect-4 is exactly solvable = another oracle) is a follow-up, not this paper |
| **OthelloGPT** (Li et al., arXiv:2210.13382) | Synthetic Othello games + world-model probing | Different game; relevant only if the paper grows the "internal state" analysis |
| **Full lichess DB** (puzzle + eval JSONL, CC0) | 394M Stockfish-evaluated positions; any-position sampling | Overkill for n=40 tasks; `data/external/lichess_mate1_raw.json` covers the mate task |

## Already used from earlier work (retained)

- **AlphaMaze / Maze-Bench** (arXiv:2502.14669): real scorer via the
  `alphamaze_reference` submodule + `Menlo/Maze-Bench-v0.2` dataset (maze track).
- **GridRoute** (arXiv:2505.24306): benchmark configs (maze track).
