# Chess SLM Benchmark: Beat DeepSeek at Chess by Reasoning (Gemma, Fine-Tuned)

**THE OBJECTIVE (see `docs/objective.md`):** benchmark Gemma 4 on chess, then
fine-tune it to play chess **better than DeepSeek V4 Flash using only natural
language reasoning** — no external search, no engine at inference time. *How*
we fine-tune is an open question (next step).

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

`scripts/build_lichess_mates.py` stratifies by **rating band** (equal-width
bands over 800-2900, equal quota per band, seeded output shuffle so any
`records[:n]` prefix is a fair spread) and `build_bestmove_evals.py` takes a
seeded random sample of the qualifying eval-DB positions.

> **The data currently committed under `data/positions/` predates that fix and
> is NOT stratified.** It was built by a version that bucketed on a hash of the
> puzzle id (not the rating) and had no per-band quota, so all 250 records came
> from one hash bucket: mate1 spans 801-1936 with 70% under rating 1200, and
> mate2 spans 801-2212. The best-move set was selected in FEN-lexicographic
> order, which is why **all 120 positions have an empty a8 square**. Re-running
> the builders produces mate1 805-2397 (median 1320) and mate2 801-2730
> (median 1654) with ~25 per band. Rebuilding replaces the sets the Phase-1
> numbers were measured on, so it is a deliberate step, not an automatic one.

## Representations

**Phase 1 (standard chess, mate-in-1, n=5/rep) measured representation
sensitivity directly: FEN 5/5, grid 3/5, list 3/5, bitboard 0/5 (with
illegal moves). The study therefore runs the **fen** representation only**
— chess capability, not format adaptation. Full Phase-1 report:
`docs/phase1-results.md` (raw per-sample data in `docs/phase1/`).

- `grid`/`bitboard`/`list` — retained only as the reported Phase-1 finding, not run further
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
- thinking enabled, max token setting, no-answer reason (truncated /
  gave_up / unparseable), and usage completeness

These fields are required for the paper figures, not optional diagnostics.

## Honesty guarantees (no fabricated answers)

- The recorded answer on every sample is the model's own text, verbatim.
- There is NO fallback: no forced second pass, no extraction from the
  reasoning text, no random move. If the model returns nothing parseable,
  the sample is a `parse_error`/`no_answer` — an honest failure.
- A gateway/transport failure is `api_error`, never model output: `content`
  stays empty, the reason is kept in a separate `error` field, and the sample
  is excluded from every rate (`n` counts scored samples, `n_attempted` counts
  rows). Previously the error string was returned *as* the model's answer and
  scored — which both counted infrastructure noise as model failure and could
  fabricate an answer (an error body containing `"code":"e4"` parsed as the
  legal SAN move `e3e4`).
- Extraction is strict and tested: the last `MOVE:`-prefixed token, then a
  line-anchored from-to move, then python-chess SAN (which only resolves to
  legal moves). Nothing is invented.
- Known model+gateway facts (measured, 2026-08-03): deepseek-v4-flash with
  thinking enabled does NOT answer within practical budgets — it consumes
  the whole budget on reasoning (0/10 native answers at 2048 total). It
  answers natively only with thinking disabled (10/10, 80% at n=10) or
  with multi-minute unbounded thinking. The gateway also ignores the
  thinking budget on many requests and serializes heavy generations per
  API key.

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
- Live dashboard: chess-bench-live.pages.dev (worker + public repo).
- Crash-safe: recovery cell pulls backed-up summaries + samples; `--resume`
  skips completed cells (schema-versioned).

## Future scope (README roadmap)

- Position evaluation (centipawn prediction from the eval DB)
- Full game-play evaluation (LLM CHESS style: legality, win rate, Elo vs a
  fixed-strength engine)
- Motif/structural questions (ChessQA-style categories: structural, motifs,
  short tactics, position judgment, semantic)
- The `pgn`/SAN-history representation (from lichess game PGNs)
- Board-image (vision) representation for multimodal models
- KinGPT-style memorization/generalization checks: theme-held-out puzzles,
  transformed positions, and verifier-in-the-loop baselines

## Repo map

```
data/positions/     committed task sets with oracles + full lichess metadata
data/raw/           gitignored: raw lichess DB downloads (1GB puzzle CSV, eval shards)
configs/
  suite.yaml        sweep definition (models x tasks, check vs full).
                    NOTE: full_n is 40, not the full task-set size — a "full"
                    sweep currently scores 40 positions per task, not 250/250/120.
src/
  models.py         4-bit HF loader (gemma) + OpenCodeGo gateway client (deepseek)
  token_usage.py    normalized provider/local token, cache, latency schema
  report.py         per-sample JSONL, summaries, comparison_table.csv
  hf_push.py        results archive to the public HF dataset repo
  live_push.py      GitHub contents-API uploads for the dashboard
  benchmarks/games/
    prompts.py      FEN prompts (one per task: mate-in-1, mate-in-2, best-move)
    tasks.py        strict answer extraction + oracle scoring (python-chess)
scripts/
  test_engine.py    dataset-consistency + extraction regression tests (the gate)
  build_lichess_mates.py    mate1/mate2 from lichess puzzle DB (stratified, rich metadata)
  build_bestmove_evals.py   bestmove from lichess eval DB (cp/depth/knodes/PV)
  build_mate_evals.py       MATE move-selection eval set (1000 held-out positions)
  run_chess.py      one model x one tactical task x FEN prompt
  run_mate_eval.py  MATE move-selection eval (single worker; gateway serializes)
  run_suite.py      full matrix -> results/chess/comparison_table.csv
  analyze_paper_figures.py   three publication figures + figure-ready CSV/JSON
  watch_run.py      detached watchdog (alive/progress/stall alerts)
  detach.sh         macOS-safe daemonizer for long local runs
notebooks/
  build_notebook.py        -> shared cell helpers for all notebook generators
  build_mate1000_variants_notebook.py -> deepseek thinking-run kernels per MATE subset
frontend/            live dashboard (Cloudflare Pages: chess-bench-live.pages.dev)
worker/              Cloudflare worker proxy (fresh GitHub contents feed)
docs/
  objective.md       THE PROJECT OBJECTIVE (read this first)
  related-work.md    detailed paper rundown + what we borrow
  memory-decisions.md  append-only log of every project decision
  paper-figures.md   figure definitions + per-sample data contract
  phase1-results.md  Phase-1 representation study
  external-resources.md
  ai-authored.md
```

## Quick start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python scripts/test_engine.py                # full gate (datasets + regressions)
python scripts/test_engine.py --quick        # regressions only, skips the dataset sweep
# deepseek smoke (needs OPENCODE_API_KEY in .env):
python scripts/run_chess.py --model deepseek-v4-flash --task mate1-lichess \
    --prompt-variant fen --n 1 --max_new_tokens 2048 --conditions win \
    --output_dir /tmp/ds_smoke
```

### On Kaggle (free T4)

1. Attach secrets: `GITHUB_TOKEN` (repo clone + live uploads), `HF_TOKEN`
   (gated gemma-4 checkpoints), `OPENCODE_API_KEY` (deepseek gateway) — attach,
   **save**, **restart kernel**.
2. Generate kernels: `python3 notebooks/build_mate1000_variants_notebook.py`
   — five parallel CPU workers per MATE subset (noexplain/tactic/both),
   then push one subset at a time with `kaggle kernels push -p notebooks/push_mate_<subset>_w<n>`.

## Live monitoring

Batched uploads (state every 60s, one upload per cell) to
`Vedang-P/chess-bench-live`. Dashboard: **chess-bench-live.pages.dev**.

## Literature

`docs/related-work.md` covers the benchmark landscape in detail and what we
borrow from each; `docs/external-resources.md` inventories the datasets.
