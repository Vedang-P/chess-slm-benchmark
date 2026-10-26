# Project Status

## Best verified model

`ccgavn-5m-seed0-v2/checkpoint-3896658` is a 4,762,088-parameter, one-pass
CC-GAVN model. Its frozen evaluation is archived at
`eval-results/ccgavn-5m-seed0-v2-3896k-final/eval-summary.json` in the
[Hugging Face dataset](https://huggingface.co/datasets/vedangfake/chess-slm-benchmark).

| Evaluation | CC-GAVN | 9M reference |
| --- | ---: | ---: |
| MATE, 4,000 decisions | 3,710 (92.75%) | 3,949 (98.72%) |
| Official puzzles, 10,000 full sequences | 7,620 (76.20%) | 8,613 (86.13%) |

The benchmarks were inspected during development; they are fixed comparison
sets, not untouched holdouts. MATE rows repeat positions.

## Current experiment

The tactical continuation uses the verified CC-GAVN weights and the original
trainer, extending its global cosine schedule for 2,000,000 more updates.
Its source is the SHA256-pinned 6,157,341-row Lichess puzzle export. The
frozen manifest removes 9,995 verbatim official-puzzle matches, leaving
6,147,346 puzzle-player roots. Stockfish 16 supplies a 50 ms score for every
legal root action; the official 9M model supplies the teacher distribution.
All five Kaggle accounts handle resumable corpus preparation and T4 training.

The [monitor](https://chess-slm-monitor.vedangpandeyy.workers.dev/) reports
demo, corpus, training, and exact 50,000-step evaluation status. The final
success gate is **strictly above 86.13%** on all 10,000 official puzzles.
Partial or preview results do not establish that claim.
