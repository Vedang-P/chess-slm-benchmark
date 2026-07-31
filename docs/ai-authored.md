# AI-Authored Research Declaration (AutoResearch 2026 Track B)

This document is the disclosure required by the AutoResearch workshop's
AI-authored track (and, more broadly, by any venue with agentic-research
disclosure requirements). It describes the agentic pipeline used to produce
this project's experiments and manuscript.

## Pipeline

- **Skills / orchestrators used**
  - RStack (lit-review, novelty-check): idea-to-experiment scaffolding;
    produced `lit-review.md` (44 verified references) and the novelty
    assessment behind the anti-goal framing.
  - **Autoresearch** (`.opencode/skills/autoresearch/`, v2.2.1): goal-directed
    iteration loop — modify → verify → keep/discard — used for benchmark
    development (engine correctness, oracle validation, eval stack).
  - `scripts/research_pipeline.py`: deterministic end-to-end runner:
    gate tests → capability sweep → anti-goal sweep → analysis → paper.

- **Humans in the loop**
  - Direction/decisions: a human researcher (Vedang Pandey) set the research
    question, chose the venue, scoped the benchmark (board sizes, tasks,
    models), and approved each phase boundary.
  - Code: written by an AI agent (opencode, deepseek-v4-flash) under human
    review; bugs found during agent-led testing were fixed by the agent.
  - Manuscript: drafted by the AI agent; reviewed/edited by the human.

- **Experiments**: executed by `scripts/run_suite.py` on Kaggle T4
  (inference-only), orchestrated by generated notebooks
  (`notebooks/build_notebook.py` → `kaggle_check.ipynb` / `kaggle_run.ipynb`).

## Reproducibility

- Every pipeline phase logs a transcript (commands, exit codes, timings) to
  `.rstack/transcript/` (see the latest `YYYYMMDD-HHMMSS/transcript.jsonl`).
- Position datasets + oracle ground truth are committed (`data/positions/`),
  generated deterministically (`scripts/generate_positions.py`).
- All results are committed as JSONL/CSV under `results/`; the comparison
  table is the manuscript's primary data artifact.

## Human involvement summary

| Activity | Human | AI |
|---|---|---|
| Research question / framing | final say | proposed |
| Literature review | review | executed (44 papers verified) |
| Benchmark + engine | review/approve | implemented + self-tested |
| Experiments | trigger | ran |
| Analysis | interpreted | computed |
| Manuscript | editor | drafter |
