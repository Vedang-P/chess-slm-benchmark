# Research idea: a 5M-parameter searchless chess model on the accuracy–efficiency frontier

Design and validate a genuinely novel searchless chess action-value model with
at most 5 million trainable parameters. The primary empirical goal is to beat
the released 9M Searchless Chess teacher on a frozen, leakage-safe evaluation
suite while using materially fewer parameters and lower inference cost. The
work must make a defensible research contribution suitable for an ICML or
NeurIPS workshop, not merely tune an existing baseline.

The immediate research questions are:

1. Why does the corrected legacy 5.30M GAVN reach 85.0% MATE but only 40.1%
   exact full-sequence puzzle accuracy, despite training to completion?
2. Which representation, objective, data curriculum, and decoding changes are
   most likely to close that gap without contaminating the frozen tests?
3. Can a compact student exploit chess symmetries, legal-move structure,
   teacher uncertainty, and decision-focused distillation more efficiently than
   a generic action-value transformer?
4. Does the resulting model improve the measured Pareto frontier in accuracy,
   parameter count, latency, calibration, and full-sequence puzzle solving?

All model selection will use a held-out development split derived only from
training positions. Frozen MATE and official puzzle sets remain final-test-only.
Every training run must periodically upload full resumable state to Hugging Face,
upload final artifacts and failure status, and support resume from Hugging Face.
Claims will be limited to results actually measured under this protocol.
