# Chess Representation Study: SLMs vs a Frontier Model

**Do small multimodal models (Gemma 4 E2B/E4B, 2-4B) and a frontier text model
(DeepSeek V4 Flash) differ in *how* they need a chess position presented — and
how far apart are their actual chess abilities?**

**Target venue**: Efficient and On-Device AI Agents Workshop @ NeurIPS 2026 (Sydney)
— inference-only study, Kaggle free T4 + gateway API.

## The question

Chess is a clean probe for *representation sensitivity*: the same position can be
rendered as a grid, a FEN string, bitboards, a piece list, or a PGN move history.
Models trained mostly on FEN-heavy chess text (frontier models) may be format-
bound; small on-device models may be weaker regardless of format. This study
measures both axes on one benchmark: **model x representation x task**, with
external oracles only (engine-verified ground truth, no model self-judgment).

Prior work (LLM CHESS, ChessQA, ChessBench, Easy2Hard, geometric-stability
testing) evaluates frontier models on single formats. We add the representation
axis and the small-model tier.

## Models

| Model | Backend | Notes |
|---|---|---|
| gemma4-e2b | local, 4-bit, T4 | small VLM; thinking disabled |
| gemma4-e4b | local, 4-bit, T4 | heavier VLM; thinking disabled |
| deepseek-v4-flash | opencode-go gateway API | frontier text model; thinking disabled; ~2s/position |

DeepSeek V4 Flash is cheap, so it runs the full representation matrix at full n;
the T4-bound gemma models run the same cells at reduced n (incremental testing).

## Tasks (core, active)

| Task | Source | Ground truth | Metadata carried |
|---|---|---|---|
| mate1-lichess (250) | lichess puzzle DB (CC0) | engine-verified checkmating moves | rating, rating dev, popularity, plays, themes, opening tags, game URL |
| mate2-lichess (250) | lichess puzzle DB (CC0) | lichess 'only move' first move | same full metadata |
| bestmove-8x8 (120) | lichess eval DB (CC0) | Stockfish eval: PV best move | cp, mate, depth, knodes, full PV |

Positions are stratified by rating/theme/evals so the committed sets span
difficulty and tactic type (34+ distinct themes).

## Representations

**Phase 1 (standard chess, mate-in-1, n=5/rep) measured representation
sensitivity directly: FEN 5/5, grid 3/5, list 3/5, bitboard 0/5 (with
illegal moves). The study therefore runs the **fen** representation only**
— chess capability, not format adaptation. Full Phase-1 report:
`docs/phase1-results.md` (raw per-sample data in `docs/phase1/`).

- `grid`/`bitboard`/`list` — retained as a reported Phase-1 finding, not run further
- `pgn` — SAN move history **future scope** (needs game-history datasets)
- **Vision** — explicitly out of scope for now

## Metrics

Per sample: `parse_error / illegal / legal` → `compliant` (matches oracle).
Per cell: parse_rate, legal_rate, compliance rate. The `format` field separates
strict from-to compliance from lenient SAN parsing, so format-following and
chess ability are reported separately.

## Monitoring (batched — no GitHub hammering)

- state.json uploaded every 60s; one upload per completed cell (summary +
  samples + index) — never per-sample.
- Live dashboard: vedang-p.github.io/chess-bench-live (worker + public repo).
- Crash-safe: recovery cell pulls backed-up summaries + samples; `--resume`
  skips completed cells (schema-versioned).

## Future scope (README roadmap)

- Full game-play evaluation (LLM CHESS style: legality, win rate, Elo vs a
  fixed-strength engine)
- Legality probes at scale (cap-legal task, existing 40-position set)
- Position evaluation (centipawn prediction from the eval DB)
- Motif/structural questions (ChessQA-style categories: structural, motifs,
  short tactics, position judgment, semantic)
- The `pgn`/SAN-history representation (from lichess game PGNs)
- Board-image (vision) representation for multimodal models

## Repo map

```
data/positions/     committed task sets with oracles + full lichess metadata
data/raw/           gitignored: raw lichess DB downloads (1GB puzzle CSV, eval shards)
configs/
  models.yaml       3-model registry (2 HF 4-bit + gateway API)
  suite.yaml        sweep definition (models x tasks x reps, check vs full)
src/
  models.py         4-bit HF loader + OpenCodeGo gateway client + registry
  report.py         per-sample JSONL, summaries, comparison_table.csv
  benchmarks/games/
    rules.py        custom NxN engine for staged small-board tasks only; 8x8
                    scoring uses python-chess (standard chess)
    oracles.py      exact retrograde solver + checkmate/mobility oracles
    positions.py    seeded generation with non-vacuity filters
    fen.py          FEN <-> board schema (+ python-chess validation)
    prompts.py      prompts: grid/fen/bitboard/list variants
    tasks.py        parsing (strict + lenient SAN) + oracle-based scoring
scripts/
  test_engine.py    engine + dataset + python-chess parity tests (the gate)
  build_lichess_mates.py    mate1/mate2 from lichess puzzle DB (stratified, rich metadata)
  build_bestmove_evals.py   bestmove from lichess eval DB (cp/depth/knodes/PV)
  run_chess.py      one model x one task x one prompt variant
  run_suite.py      full matrix -> results/chess/comparison_table.csv
  analyze_results.py         writes docs/capability-analysis.md
notebooks/
  build_notebook.py  SINGLE SOURCE -> kaggle_check.ipynb + kaggle_run.ipynb
frontend/            live dashboard (GitHub Pages: vedang-p.github.io/chess-bench-live)
docs/
  related-work.md    detailed paper rundown + what we borrow
  external-resources.md
  ai-authored.md
```

## Quick start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python scripts/test_engine.py --quick        # CPU-only gate (no torch needed)
# deepseek smoke (needs OPENCODE_API_KEY in .env):
python scripts/run_chess.py --model deepseek-v4-flash --task mate1-lichess \
    --prompt-variant grid --n 1 --max_new_tokens 4096 --conditions win \
    --output_dir /tmp/ds_smoke
```

### On Kaggle (free T4)

1. Attach secrets: `GITHUB_TOKEN` (repo clone + live uploads), `HF_TOKEN`
   (gated gemma-4 checkpoints), `OPENCODE_API_KEY` (deepseek gateway) — attach,
   **save**, **restart kernel**.
2. Upload `notebooks/kaggle_check.ipynb` — engine tests + tiny sweep (n=1).
3. Upload `notebooks/kaggle_run.ipynb` — full sweep; batched monitoring; crash-safe.

## Live monitoring

Batched uploads (state every 60s, one upload per cell) to
`Vedang-P/chess-bench-live`. Dashboard: **vedang-p.github.io/chess-bench-live**.

## Literature

`docs/related-work.md` covers the benchmark landscape in detail and what we
borrow from each; `docs/external-resources.md` inventories the datasets.
