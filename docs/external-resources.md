# External resources (datasets, code, papers)

Inventory of everything this project builds on or references. Lichess
data is CC0; MATE and third-party code have their own licenses — attribute
properly in the paper.

## Adopted

| Resource | What we use | Where |
|---|---|---|
| **Lichess puzzle database** (CC0, 6.1M puzzles) | `mate1-lichess` (250) and `mate2-lichess` (250): mate-in-1/2 tactics, stratified by rating, with full metadata (rating, themes, popularity, plays, opening tags, game URLs) | `scripts/build_lichess_mates.py` → `data/positions/` |
| **Lichess eval database** (CC0, 394M Stockfish evals) | `bestmove-8x8` (120): positions with Stockfish best move + cp/depth/knodes/PV | `scripts/build_bestmove_evals.py` → `data/positions/` |
| **MATE** (Wang et al., NAACL 2025; HF `OutFlankShu/MATE_DATASET`) | `mate-selection-test` (1000 held-out positions): FEN + two candidate moves + expert strategy/tactic explanations + expert truth. Also the fine-tuning source (50k planned). | `scripts/build_mate_evals.py` → `data/positions/` |
| **python-chess** (niklasf, MIT) | The rules engine for ALL scoring (standard chess); FEN parsing; SAN resolution | `src/benchmarks/games/tasks.py` |
| **DeepSeek V4 Flash** (opencode-go gateway) | Frontier reference model, thinking on/off arms | `src/models.py` |
| **Gemma 4 E2B/E4B** (Google, gated) | Small-model tier; fine-tuning targets (4-bit LoRA, planned) | `src/models.py` |

## Considered, not adopted

| Resource | Why it exists | Why not now |
|---|---|---|
| **ChessBench** (Ruoss et al., DeepMind, 2024) | 10M games with Stockfish-16 move+value labels; 270M-param transformers reach 2895 Elo | Not needed as eval data — our lichess-derived sets cover the same sources with richer metadata. But its *data recipe* is the strongest known chess-pretrained SLM base if we ever pre-train |
| **LLM CHESS** (Kolasani et al.) | Multi-turn game-play harness | Full-game play is future scope; game-play protocol borrowed for the planned 100-game match |
| **KinGPT / GAMBIT** (Tang) | Memorization-brittleness evaluation + LLM-Modulo | Borrowed as evaluation hygiene (theme-held-out splits, sanity metric, verifier-loop later) |
| **ChessQA** (Wen et al.) | 5-category chess-understanding benchmark | Category taxonomy borrowed as future scope (structural/motifs/semantic arms) |

## Key prior-work anchors

- MATE (2411.06655) — our base: expert-annotated move selection, fine-tune-then-compare.
- KinGPT / "Generalization or Memorization?" (2605.17565) — brittleness testing, LLM-Modulo.
- ChessQA (2510.23948) — capability taxonomy.
- LLM CHESS (2512.01992) — game-play protocol, Elo.
- ChessBench (2402.04494, Ruoss et al.) — dataset ancestry + chess-pretrained SLM recipe.
- ChessGPT (2306.09200, NeurIPS 2023) — policy+language bridging; KinGPT's main comparison target.
- Spec-gaming in reasoning models (2502.13295) — prompt hygiene against hacking.
- Easy2Hard (NeurIPS 2024 D&B) — difficulty stratification.

Full analysis: `docs/related-work.md`.
