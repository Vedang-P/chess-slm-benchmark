# MATE-Grounded Chess Study: Fine-Tuned SLMs vs a Frontier Model

**Can a current on-device model, fine-tuned on expert-annotated chess data,
close the gap to a frontier reasoning model?**

**Target venue**: Efficient and On-Device AI Agents Workshop @ NeurIPS 2026 (Sydney)
— Kaggle free T4 fine-tuning/evaluation + gateway API.

## The question

Chess is a clean probe for reasoning because the state, legal actions, and
oracles are explicit. The study is grounded in MATE (Wang et al., NAACL 2025),
which fine-tuned LLaMA-3-8B on one million expert-annotated positions with
strategy/tactic explanations. We update that recipe to Gemma 4 E2B and compare
against DeepSeek V4 Flash, while retaining our standard-chess mate/best-move
battery for transfer evaluation.

Prior work (LLM CHESS, ChessQA, ChessBench, Easy2Hard, geometric-stability
testing) evaluates frontier models on single formats. We add the representation
axis and the small-model tier.

## Models

| Model | Backend | Notes |
|---|---|---|
| gemma4-e2b | local, 4-bit, T4 | first fine-tuning target; base + MATE-LoRA |
| gemma4-e4b | local, 4-bit, T4 | second fine-tuning target after E2B |
| deepseek-v4-flash | opencode-go gateway API | frontier reasoning reference; thinking enabled |

DeepSeek V4 Flash is the current frontier reference. Fine-tuning starts with
Gemma 4 E2B only, before E4B is attempted.

## Phase 2: MATE-only fine-tuning experiment

The immediate next experiment is **MATE only**, with a first 50,000-example
training sample and Gemma 4 E2B. The full chess benchmark is deliberately on
hold until this training/evaluation path is understood.

MATE is public at `OutFlankShu/MATE_DATASET`. It contains strategy and tactic
subsets, combined data, explanation/no-explanation variants, and a held-out
test set. A record is supervised candidate-move selection, not the same as
our open-ended FEN move-generation task. For example:

```text
FEN: 1r4k1/4nppp/8/4Pb2/8/1P5P/r1PR4/3R3K w - - 0 27
MoveA: d2d7  Place the piece more actively...
MoveB: d2d8  Switch the piece to a more advantageous place...
Answer: MoveB:d2d8
```

Tactic examples add a concrete line and motif:

```text
MoveA: d2d8, tactic d2d8 b8d8 d1d8 Checkmate!
MoveB: d2d7, tactic d2d7 f5d7 Trade the lower value piece...
Answer: MoveA:d2d8
```

The first fine-tune will use MATE only, 50k examples, one packed epoch, and
4-bit QLoRA on Gemma 4 E2B. The expected data volume is roughly 6-10M text
tokens; a realistic first estimate is **4-12 hours on a Kaggle T4**, but a
1,000-example calibration run will measure actual steps/second before the
50k run. The adapter, tokenizer/config, data manifest, loss/throughput logs,
and evaluation reports will be uploaded to Hugging Face.

This is not identical to the current FEN benchmark: MATE trains/evaluates
candidate selection (`MoveA` vs `MoveB`), while the tactical battery asks for
an open-ended move from a FEN. We will use MATE first, then test transfer to
mate-in-1, mate-in-2, and Stockfish best-move tasks.

## Planned 100-game match

After the MATE-only E2B experiment is validated, we will run a 100-game
head-to-head: 50 games with Gemma-LoRA as White and 50 as Black against
DeepSeek V4 Flash. Games will use standard python-chess rules, documented
opening seeds, a fixed ply cap, and explicit draw handling. We will save the
PGN plus every move's FEN, prompt, thinking, answer, legality, token usage,
latency, and termination reason. The report will include wins/losses/draws,
legality, average length, tokens per move, and color-balanced score/Elo
intervals. This match is later scope, not part of the first 50k run.

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

Every new sample also records a normalized `token_usage` object:

- input, output, reasoning, and total tokens
- cache reads, writes/misses, and hit tokens when the provider reports them
- time to first token, generation seconds, output tokens/second, reasoning
  tokens/second
- thinking enabled, max token setting, fallback type, and usage completeness

These fields are required for the paper figures, not optional diagnostics.

## Paper figures

`scripts/analyze_paper_figures.py` generates three publication figure families
from raw JSONL without hand-entered values:

1. **Performance versus tokens** — strict accuracy against average total
   tokens per sample; thinking-enabled models are marked separately.
2. **Response breakdown** — stacked overall/per-category bars for correct,
   legal-wrong, illegal, parse-error, and no-answer outcomes.
3. **Accuracy heatmap** — task category by model, with strict accuracy in each
   cell and counts in the companion tables.

The exact contract is documented in `docs/paper-figures.md`.

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
- MATE strategy/tactic explanation ablations beyond the first 50k E2B run
- 100-game Gemma-LoRA vs DeepSeek head-to-head with PGN and per-move telemetry
- KinGPT-style memorization/generalization checks: theme-held-out puzzles,
  transformed positions, and verifier-in-the-loop baselines

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
  token_usage.py    normalized provider/local token, cache, and latency schema
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
  analyze_paper_figures.py   three publication figures + figure-ready CSV/JSON
notebooks/
  build_notebook.py  SINGLE SOURCE -> kaggle_check.ipynb + kaggle_run.ipynb
frontend/            live dashboard (GitHub Pages: vedang-p.github.io/chess-bench-live)
docs/
  related-work.md    detailed paper rundown + what we borrow
  paper-figures.md   figure definitions + per-sample data contract
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
    --prompt-variant fen --n 1 --max_new_tokens 32768 --conditions win \
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
