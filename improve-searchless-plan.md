# Improve searchless_chess (Ruoss) — DPO self-play pipeline at 136M/270M

Decision 2026-08-27. Direction: improve the open-source searchless_chess
models (google-deepmind/searchless_chess) toward GM-level / MATE-strong
play at smaller params. Precedent: dbest-isi/searchless-chess-9M-dpo
(+25 Elo, +1% puzzles from 1000 self-play games, 36k pairs, 50 steps).

## The gap we fill (vs the DPO precedent)
- Precedent: 9M only, 1000 games, depth-20/0.1s SF, 50 steps.
- We: 136M/270M, 10-50k self-play games, deeper SF (d24+), MATE-pool
  augmented pairs, multi-iteration DPO, EMA eval. The 270M→136M Elo gap
  is only +40 (2299 vs 2259); DPO gave +25 at 9M — closing 136M→270M
  parity is a plausible, measurable target.

## Evidence so far (MATE transfer — pending full-eval kernel)
- 270M MATE eval kernel RUNNING: 3 models × 4 MATE test sets.
- Transfer question: does ChessBench-trained play transfer to MATE's
  expert 2-choice? gemma base = 58.1%.

## Pipeline (per iteration, ~10-15h P100 per model)
1. SELF-PLAY: current model plays N games vs itself (N=10k; temp 0.3-0.7;
   ~270M at ~150 tok/s batch → 10k × 60 plies ≈ 1-2h gen on P100).
2. MISTAKE ID: Stockfish (depth 20-24, multi-threaded) scores every
   position; find plies where model move != SF best with |Δeval| > 0.3
   pawns → (chosen=SF move, rejected=model move) pairs.
   Add MATE-pool positions as extra preference pairs (expert labels).
3. DPO TRAIN: JAX/Haiku (reuse dbest-isi code layout), lr 1e-5, batch 32,
   50-500 steps, beta 0.1, EMA 0.999, ref frozen per iteration.
   Fits P100: 270M fp16 ≈ 540MB + Adam ≈ 3GB ≪ 16GB.
4. EVAL: MATE 4-set 2-choice + puzzle accuracy (official puzzles.py) +
   head-to-head vs base (BayesElo). Gate: any regression → keep prev iter.
5. REPEAT 2-3 iterations.

## Compute budget (2 accounts, 30h/wk each)
| Step | GPU-h (136M) | GPU-h (270M) |
|---|---|---|
| self-play 10k games | 1-2 | 2-3 |
| SF mistake ID | 0 (CPU) | 0 |
| DPO 200 steps | 2-3 | 4-6 |
| eval suite | 1 | 1 |
| per iter | ~5 | ~8 |
| 2-3 iters | 10-15 | 16-24 |

## Paper framing (workshop)
"Closing the gap to 270M at 1/2 the parameters: DPO self-play improvement
of open searchless chess models, evaluated on MATE (expert 2-choice) —
the first evaluation of searchless models on MATE + the first open
improvement of the 136M checkpoint."

## Gates
- G1: MATE transfer verdict (running kernel) — if 270M ≫ 58.1%, the
  MATE-eval contribution stands on its own; if ≈58%, transfer is the
  finding and improvement is harder.
- G2: pilot DPO on 136M (1 iter, 2k games) → +Elо vs base on MATE.
- G3: full 3-iter run.

## Repo layout to reuse
- /tmp/searchless_chess (official) — configs, tokenizer, engines, puzzles.py
- dbest-isi/searchless-chess-9M-dpo — DPO code (searchless_chess_code/),
  params.npz, hf_model.py loader
