# Do Small Models Follow Objectives They Cannot Verify?

**A controlled chess benchmark for SLM chess ability and anti-goal instruction following.**

**Target venue**: Efficient and On-Device AI Agents Workshop @ NeurIPS 2026 (Sydney)
— deadline **Aug 29, 2026 (AoE)** · Inference-only study, Kaggle free T4.

## The two questions

**Phase 0 — capability:** what can small language models (1.5–4B) actually do in chess?
We measure parse rate, **legal-move rate**, mate-in-1 solve rate, and exact winning-move
rate — across two input representations (**rendered grid** vs **FEN notation**) to find the
representation SLMs understand best before any anti-goal work.

**Phase 1 — anti-goal:** can those same models follow an explicit *"lose deliberately"*
instruction when every aspect of compliance is machine-verifiable? Paired **WIN/LOSE
conditions on identical positions**, exact game-theoretic oracles (3x3/5x5 solved by
retrograde analysis; exact checkmate/mobility on 8x8), and a within-model control
(divergence between WIN-condition and LOSE-condition moves).

## Benchmark tasks

| Task | Board | Oracle | WIN condition | LOSE condition |
|---|---|---|---|---|
| cap-legal-8x8 | 8x8, standard positions (lichess) | none (legality only) | any legal move | — |
| mate-in-1 (lichess) | 8x8, 288 real CC0 positions | exact checkmate detection | deliver mate | avoid mate, stay legal |
| mate-in-1 (synthetic) | 8x8 | exact checkmate detection | deliver mate | avoid mate, stay legal |
| single-move | 3x3, 5x5 (K+X vs K) | exact game value (retrograde) | play a winning move | play a strictly-worsening move |
| max-mobility | 8x8 | exact opponent-reply counts | minimize replies | maximize replies |

Metrics (all external oracles, no model self-judgment): parse rate, **legal rate**,
**compliance**, **divergence** (within-model WIN↔LOSE), plus a failure taxonomy.

## Repo map

```
data/positions/     committed: 8 task sets with embedded oracle ground truth
data/external/      kagi 1000-puzzle CSV + raw lichess mateIn1 filter (CC0)
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
    fen.py          FEN <-> board schema (+ python-chess validation)
    prompts.py      faithful prompts: grid/fen variants, WIN/LOSE objectives
    tasks.py        parsing + oracle-based compliance scoring
scripts/
  test_engine.py    engine + dataset + python-chess parity tests (the gate)
  generate_positions.py    one-time dataset generation (already run; committed)
  build_lichess_mate1.py   fetch/build real mate-in-1 positions (CC0 lichess DB)
  build_cap_positions.py   build the legality probe from standard positions
  run_chess.py      one model x one task x one prompt variant
  run_suite.py      full matrix -> results/chess/comparison_table.csv
  analyze_results.py      writes docs/capability-analysis.md + anti-goal-analysis.md
  research_pipeline.py    end-to-end runner (gate -> sweep -> analyze -> paper),
                          with a reproducible transcript (.rstack/transcript/)
notebooks/
  build_notebook.py  SINGLE SOURCE -> kaggle_check.ipynb + kaggle_run.ipynb
.opencode/
  skills/autoresearch/   autoresearch skill (autonomous iteration loop, v2.2.1)
  commands/ agents/ scripts/  (autoresearch subcommands + orchestrator)
docs/
  capability-analysis.md  auto-generated Phase-0 report (after a run)
  anti-goal-analysis.md   auto-generated Phase-1 report (after a run)
  external-resources.md   what exists to build on (adopted vs considered)
  ai-authored.md          AI-authored-track disclosure + reproducibility
```

## Quick start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python scripts/test_engine.py --quick        # CPU-only gate (no torch needed)
python scripts/run_suite.py --smoke --models smollm2-1.7b   # pipeline smoke test
python scripts/research_pipeline.py --smoke                 # end-to-end, stub models
```

### On Kaggle (free T4)

1. Set the `HF_TOKEN` secret for the gated Gemma models (else use the 4 ungated ones).
2. Upload `notebooks/kaggle_check.ipynb` — engine tests + python-chess parity + tiny
   sweep; **raises on any failure**.
3. Upload `notebooks/kaggle_run.ipynb` — full sweep (6 models x 8 tasks x variants,
   ~6–12h), zips `results/`.
4. Locally: unzip into `results/`, then `python scripts/analyze_results.py` to generate
   the capability + anti-goal reports.

Resume after a died session: re-run `run_suite.py` with `--models <remaining> --tasks ...`;
per-run JSONs under `results/chess/*.summary.json` are the source of truth.

## Literature & framing

`docs/external-resources.md` inventories the codebases/datasets we build on
(python-chess parity tests, lichess CC0 positions, kagisearch's 1000-puzzle CSV).
Key prior work: AlphaMaze (SLM game ability via SFT+GRPO), Inverse IFEval +
reversed-performance personas (text-domain anti-goal failure), specification gaming,
sandbagging, goal misgeneralization, reversal curse, solved minichess oracles.
