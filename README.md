# Anti-Goal Chess Benchmark: Do Small Models Follow Objectives They Cannot Verify?

**Target venue**: Efficient and On-Device AI Agents Workshop @ NeurIPS 2026 (Sydney) — deadline
**Aug 29, 2026 (AoE)**
**Hardware**: Kaggle free T4 (16 GB) — inference-only study, no training in the base plan.

## The question

Can small language models (1.5–4B) follow *anti-goals* — explicit "lose deliberately"
instructions — when every aspect of compliance is machine-verifiable? We probe this with a
**paired win/lose benchmark on identical chess positions**, using **exact game-theoretic
oracles** (3x3/5x5 minichess positions are solved by retrograde analysis; 8x8 puzzle tasks use
exact checkmate and mobility checks). The within-model control (same position, WIN vs LOSE
instruction) makes "deliberate" measurable holding skill constant. Prior work (Inverse IFEval,
reversed-performance personas) shows the failure mode in text; nobody has tested it in a
verifiable game domain at SLM scale, and nobody has tried training anti-goal compliance.

See `idea.md` (v8 pivot), `lit-review.md` (44 verified references), `novelty-assessment.md`
(7/10), `refined-idea.md` (the consumable design).

## Status (2026-07-31)

- Rule engine + exact retrograde oracles: implemented, unit-tested, fuzz-tested
  (`scripts/test_engine.py` — all pass).
- Position datasets (240 positions, 6 task sets, embedded oracle ground truth): generated
  deterministically, committed to `data/positions/`.
- Eval pipeline (`scripts/run_chess.py` / `run_suite.py`): smoke-tested end-to-end.
- Kaggle notebooks: **re-created** `notebooks/build_notebook.py` → `kaggle_check.ipynb` +
  `kaggle_run.ipynb` (the previous generator was missing from git — now fixed).
- **Not yet run on real models.** First step on Kaggle: `kaggle_check.ipynb` (tiny, raises on
  any failure), then `kaggle_run.ipynb` (full sweep, ~6–12h T4).

## Study design (short version)

| Task | Boards | Oracle | WIN condition | LOSE condition |
|---|---|---|---|---|
| single-move | 3x3, 5x5 (K+X vs K, K+P vs K) | exact game value (retrograde) | play a winning move | play a strictly-worsening move (legal) |
| mate-in-1 | 8x8 | exact checkmate detection | deliver mate | avoid mate, stay legal |
| max-mobility | 8x8 | exact opponent-reply counts | minimize opponent replies | maximize opponent replies |

Metrics (all external — no model self-judgment): parse rate, **legal rate**, **compliance**
(against the oracle), **divergence** (win-condition move vs lose-condition move on the same
position — the within-model control), plus a failure taxonomy per sample.

## Repo map

```
data/positions/          COMMITTED: 6 task sets, 40 positions each, oracle data embedded
configs/
  models.yaml            model registry (6 SLMs; gated flags)
  suite.yaml             sweep definition (models x tasks x n, check vs full)
src/
  benchmarks/games/
    rules.py             simplified chess engine for NxN boards (no castling/en-passant)
    oracles.py           exact retrograde solver + mate/mobility oracles
    positions.py         seeded generation with non-vacuity filters
    prompts.py           faithful WIN/LOSE prompt templates (differ only in objective)
    tasks.py             parsing + oracle-based compliance scoring
  report.py              per-sample JSONL, summaries, comparison_table.csv
scripts/
  generate_positions.py  one-time dataset generation (already run; committed)
  test_engine.py         engine + dataset invariant tests (the Kaggle gate)
  run_chess.py           one model x one task, both conditions
  run_suite.py           full matrix -> results/chess/comparison_table.csv
notebooks/
  build_notebook.py      SINGLE SOURCE -> kaggle_run.ipynb + kaggle_check.ipynb
hf_models.py             model loading/parsers (shared with the maze track)
eval.py                  maze-track eval (GridRoute/MazeBench — retained, not in this paper)
legacy/                  train_sft.py, train_grpo.py, check_finetune_feasibility.py
                         (training follow-up; sys.path-fixed, preserved)
paper/                   LaTeX writeup (4-page short paper; to be rewritten for v8 framing)
```

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python scripts/test_engine.py --quick        # CPU-only gate (no torch needed)

# smoke the eval pipeline without a GPU:
python scripts/run_suite.py --smoke --models smollm2-1.7b --tasks sm-3x3-win mate1-8x8
```

### On Kaggle (free T4)

1. Set the `GITHUB_TOKEN` secret if your repo is private (else skip).
2. Set the `HF_TOKEN` secret for the gated Gemma models (skip to use the 4 ungated models).
3. Upload `notebooks/kaggle_check.ipynb` — runs engine tests + tiny sweep, **raises on any
   failure**. Fix and repeat before any full run.
4. Upload `notebooks/kaggle_run.ipynb` — full sweep (6 models x 6 tasks x ~40 positions,
   ~6–12h), zips `results/` for download.
5. Rebuild results tables: unzip into `results/chess/`; `comparison_table.csv` is the paper's
   main artifact.

If a session dies mid-sweep, resume with `--models <remaining> --tasks ...`; per-run JSONs are
the source of truth.

## Maze track (retained, not this paper)

`eval.py` + `src/grid_generator.py` / `src/token_maze.py` / `src/evaluation.py` + the
`alphamaze_reference` submodule (GridRoute NL/token + MazeBench). Historical numbers in that
track were declared untrusted on 2026-07-15 (see git history); it is not part of the current
paper.
