# How Well Do Small Language Models Play Chess?

**A multi-task capability benchmark: legality, tactics, move strength, endgames, and
full-game play — plus simple games (tic-tac-toe, Connect-4) — for 1.5–4B models.**

**Target venue**: Efficient and On-Device AI Agents Workshop @ NeurIPS 2026 (Sydney)
— deadline **Aug 29, 2026 (AoE)** · Inference-only study, Kaggle free T4.

## The question

Every existing chess/game benchmark evaluates frontier models (LLM Chess, ChessArena,
ChessQA, ChessBench, Topsakal's grid games). A search for `"small language model" + chess`
returns exactly one paper. This project is the first systematic measurement of what small
open models (1.5–4B) can do in chess and simple games.

## Benchmark tasks

| Task | Board | Ground truth | What it measures |
|---|---|---|---|
| cap-legal-8x8 | 8x8, standard positions (lichess) | our engine + python-chess | rule understanding (legality) |
| mate1-lichess | 8x8, 262 real CC0 puzzles | exact checkmate detection | tactics (mate-in-1) |
| mate2-lichess | 8x8, 224 real CC0 puzzles | lichess solution ('only moves') | tactics (mate-in-2, first move) |
| bestmove-8x8 | 8x8, 40 positions | **Stockfish 18** (local engine) | move strength (top-1 accuracy) |
| sm-3x3-win / sm-5x5-win / sm-5x5-draw | 3x3/5x5, K+X vs K | exact retrograde values | endgame understanding |
| playout-5x5 | full chess games vs random | our engine | sustained play, completion, win rate |

**Representation control.** 8x8 tasks support four prompt representations — `grid`
(rendered board), `fen`, `bitboard` (64-bit bitboards per piece type), `list` (piece list).
Before the full sweep, a **representation pilot** (`configs/pilot.yaml` →
`kaggle_pilot.ipynb`) measures which representation small models actually understand;
`analyze_pilot.py` picks the winner, and the full sweep uses it (plus the runner-up for
the paper's ablation). If a model fails in every representation, the failure is
capability, not prompt format.

**No artificial thinking limits.** Position tasks run with a 2048-token generation
budget (effectively unlimited for these models) — truncating a model's chain of thought
mid-reasoning would confound the measurement. Full-game playouts use 1024 tokens per
move (per-ply generation dominates runtime there).

Every metric uses external oracles — no model self-judgment.

## Repo map

```
data/positions/     committed: 9 task sets with embedded oracle ground truth
data/external/      kagi 1000-puzzle CSV + raw lichess puzzle filter (CC0)
configs/
  models.yaml       model registry (6 SLMs; gated flags)
  suite.yaml        sweep definition (models x tasks x variants, check vs full)
src/
  models.py         4-bit HF inference loader + Ollama backend + registry
  report.py         per-sample JSONL, summaries, comparison_table.csv
  benchmarks/games/
    rules.py        simplified NxN chess engine (no castling/en-passant/double-step)
    oracles.py      exact retrograde solver + checkmate/mobility oracles
    positions.py    seeded generation with non-vacuity filters
    envs.py         playout environments: Chess5x5, TicTacToe, Connect4
    fen.py          FEN <-> board schema (+ python-chess validation)
    prompts.py      faithful prompts: grid/fen variants
    tasks.py        parsing + oracle-based scoring (cap/mate1/mate2/bestmove/sm/mob)
scripts/
  test_engine.py    engine + dataset + python-chess parity tests (the gate)
  generate_positions.py      one-time dataset generation (already run; committed)
  build_lichess_mates.py     fetch/build mate-in-1/2 sets (CC0 lichess DB)
  build_bestmove.py          Stockfish-verified best-move set (local engine)
  build_cap_positions.py     legality probe from standard positions
  run_chess.py      one model x one task x one prompt variant (incl. full games)
  run_suite.py      full matrix -> results/chess/comparison_table.csv
  analyze_results.py         writes docs/capability-analysis.md
  research_pipeline.py       end-to-end runner with reproducible transcript
notebooks/
  build_notebook.py  SINGLE SOURCE -> kaggle_check.ipynb + kaggle_run.ipynb
frontend/            live dashboard (GitHub Pages: vedang-p.github.io/chess-bench-live)
.opencode/           autoresearch skill (autonomous iteration loop)
docs/
  capability-analysis.md  auto-generated after a run
  external-resources.md   what exists to build on
  ai-authored.md          AI-authored-track disclosure + reproducibility
  terminal-gpu.md         Modal/Kaggle/Colab options
```

## Quick start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
brew install stockfish          # only needed to rebuild the best-move set

python scripts/test_engine.py --quick        # CPU-only gate (no torch needed)
python scripts/run_suite.py --smoke --models smollm2-1.7b   # pipeline smoke test
```

### On Kaggle (free T4)

1. Attach `HF_TOKEN` (Gemma models) and `GITHUB_TOKEN` (private-repo clone) secrets —
   attach in the notebook's Secrets panel, **save**, **restart kernel**.
2. Upload `notebooks/kaggle_check.ipynb` — engine tests + parity + tiny sweep.
3. Upload `notebooks/kaggle_pilot.ipynb` — **representation pilot** (2 models × 3 tasks
   × 4 representations, ~1-2h): picks the prompt format the full sweep should use.
4. Upload `notebooks/kaggle_run.ipynb` — full sweep; streams to the dashboard and
   backs up every cell to the public live repo (crash-safe: recovery cell + `--resume`).
5. Locally: `python scripts/analyze_pilot.py` (after step 3) and
   `python scripts/analyze_results.py` (after step 4).

## Live monitoring

The sweep publishes `monitor/state.json` + per-cell results to the public repo
`Vedang-P/chess-bench-live` (~every 2 min). Dashboard: **vedang-p.github.io/chess-bench-live**
(static; deploy from `frontend/` to Vercel/Cloudflare if preferred).

## Literature

`docs/external-resources.md` inventories what exists to build on; the lit review
(`lit-review.md`, 49 verified records) documents the gap: no multi-task chess/game
capability benchmark at SLM scale.
