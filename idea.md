# Research idea: Phase-segregated chess position dataset (opening/middlegame/endgame)

## Question
Can we build a refined chess training set for SLM trace distillation by
segregating positions into opening / middlegame / endgame subsets (instead
of playing full games), and can a subset of this dataset serve as a
phase-stratified benchmark for evaluating other models?

## Method sketch
- Source positions: MATE train zips (208k rows, already local) or lichess masters games.
- Segregation: deterministic phase detection (ply count, material, piece
  configuration, king safety) into opening/middlegame/endgame.
- Teacher: deepseek-v4-flash unbounded thinking traces per position, filtered
  by Stockfish eval stability.
- Student: gemma 4 E2B QLoRA SFT on the segregated traces.
- Benchmark: hold out a stratified subset per phase; evaluate other models
  (base gemma, MATE-LoRA, deepseek, frontier baselines) on it.

## Domain
Chess SLM research; MATE-grounded evaluation; phase-aware training data.
