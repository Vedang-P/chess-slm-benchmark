# Novelty Assessment: Phase-Segregated Chess Positions for SLM Trace Distillation & Benchmarking

Date: 2026-08-11. Based on 32 papers in `.rstack/lit-review.jsonl` (16 highly relevant).

## Paper-by-Paper Comparison

### Direct competitors (same method + same domain)

| Paper | Overlap | Relationship |
|---|---|---|
| **MATE** (2411.06655) | Position-level move selection on the same positions we'd use; expert annotations | `complementary` — our dataset uses MATE's positions but adds phase labels + teacher traces; MATE is the anchor baseline, not a competitor on data design |
| **Master Distillation / C1** (2603.20510) | Position-level trace distillation with engine-verified targets, teacher-hidden context | `partial_overlap` — **closest work**. Same mechanism (verified traces → SFT). C1 samples by balance (difficulty/theme); we add the **phase axis**. No phase segregation in C1's ablations. |
| **Complete Chess Games** (2501.17186) | Whole-game text training | `direct_competitor` (counter-position) — argues full games are the data; we argue phase-segregated verified traces are more efficient per token. |

### Partial overlap (method OR domain, not both)

| Paper | Shared | Missing vs our idea |
|---|---|---|
| **How Reasoning Evolves from Post-Training Data** (2604.05134) | Post-training data composition shapes chess reasoning | Doesn't propose a dataset; no phase axis, no benchmark release |
| **Strategic Reasoning post-training insights** (2507.11055) | Chess as post-training probe | No data design contribution |
| **ChessQA** (2510.23948) | Position-level evaluation | Stratifies by *ability* (structural/motifs/tactics/judgment), not by *game phase* |
| **LLM CHESS** (2512.01992) | Game-play benchmarking | Full-game protocol; no position subsets, no phases |
| **ChessArena** (2509.24239) | Strategic reasoning testbed | Game-level, not phase-stratified |
| **LEAP** (2310.20260) | Text-grounded move evaluation | Textbook corpus, not phase-labeled; different task (explanation grounding) |
| **Brittleness Testing** (2605.17565) + **Disentangling mem/gen** (2601.16823) | Evaluation hygiene (theme-held-out, OOD) | Inform our benchmark design; don't propose datasets |
| **Chess Transformer** (2008.04057), **Complete Chess Games**, **Learning Chess w/ LMs** (2209.11902) | Game-text training | No verification, no reasoning traces, no phases |
| **ChessGPT** (2306.09200) | Policy+language combined | Unreproducible; no dataset |
| **Emergent World Models** (2403.15498), **Chess-World-Model** (2605.30100), **Tracking World States** (2508.19851) | State tracking from sequences | Different capability, orthogonal |
| **UniMaia** (2605.27767), **Three-Body** (2607.21993), **Communicating Strategies** (2607.11486), **Hallucinations on Board** (2608.04240) | Language-steered/commentary chess | Not about training data design |

## Novelty Assessment

**What is genuinely novel:**
1. **The phase axis as a training-data design dimension.** No work we found trains trace-distilled SLMs on explicitly segregated opening/middlegame/endgame subsets. C1's ablations are balance/difficulty; MATE is category (explanation presence); nobody uses game phase. Since phases demand different reasoning (book memory vs tactics vs endgame technique), phase-stratified supervision is a principled, untested data lever.
2. **A phase-stratified benchmark.** No LLM chess benchmark stratifies by game phase. Existing axes: ability categories (ChessQA), gameplay (LLM CHESS/ChessArena), text grounding (LEAP). An endgame tier is nearly absent from LLM chess evals entirely.
3. **Phase-aware trace filtering.** The insight that ±100cp Stockfish filters behave differently per phase (open middlegame vs K+P endgame) — and that the teacher's (deepseek's) weaknesses are phase-dependent — is not discussed anywhere we found.

**What overlaps (be honest):**
- The *mechanism* (engine-verified teacher traces → SFT of a small model) is C1's (2603.20510), applied in the same domain. That is the biggest overlap.
- Position-level supervision is MATE's (2411.06655) idea.
- Benchmark hygiene ideas are from the memorization line (2601.16823, 2605.17565).

**Gaps addressed:**
- Sample-efficiency of trace data (phase-stratified subsets vs uniform sampling — measurable against C1's 39k result)
- Endgame reasoning measurement for LLMs (nearly absent in benchmarks)
- Reproducible phase classification for chess ML (no consensus definition published for ML use)

**Novelty score: 7/10.**
Genuine novelty in the *data axis* (phase segregation) and the *benchmark instrument* (phase-stratified release), built on a proven mechanism (C1 trace distillation). Not paradigm-shifting (the mechanism exists), but a defensible, clean contribution for a workshop — and it survives even a negative training result because the benchmark release stands alone.

**Main overlap concern:** if the training arm fails to beat C1-style uniform sampling, the paper's contribution collapses to the benchmark + classifier. The benchmark arm must be designed to be the *primary* contribution, not an afterthought.
