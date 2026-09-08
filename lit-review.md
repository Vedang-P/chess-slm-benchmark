# Literature Review — Compact Searchless Chess Action-Value Models

## Background

Searchless Chess shows that a transformer can amortize Stockfish-style
action-value evaluation: a model receives a FEN and a candidate action, then
predicts a return distribution. Ruoss et al. released ChessBench, including
10M games and 15.3B action-value labels, and found that scale matters strongly
but that perfect distillation remains unresolved. This project tests whether
the missing efficiency can come from chess-specific structure rather than from
more generic transformer parameters.

The relevant literature separates into three strands: chess-specific
representations, compact action-value/policy learning, and knowledge
distillation. The first strand is especially important here. Chessformer work
shows that square tokens, geometry-aware attention, and source-destination
action representations can outperform more generic alternatives. Separate
AlphaZero research likewise finds that feature and target design may matter
more than swapping in a transformer.

## Methods landscape

The released Searchless Chess models are board encoders conditioned on an
action token, optimized to predict a 128-bin return distribution. Chessformer
introduces square tokens, dynamic geometric attention bias, and a
source-destination policy head. The prior GAVN in this repository already
adopted the square-token and geometric-bias ideas, but it encoded the board
fully before introducing the candidate action in a small final MLP. Its action
representation therefore could not change earlier board-to-board interactions.

CC-GAVN makes the candidate action a token in every attention layer. The
candidate attends to the board and the board attends back to the candidate,
which gives an action-specific latent representation without inference-time
search. It additionally uses only an exact horizontal symmetry, including
castling, en-passant, and UCI-action remapping. This avoids treating invalid
rotations or color swaps as chess symmetries.

## Distillation implications

Direct output KL is a necessary baseline but does not necessarily preserve
the decision boundary of a compact student. Teacher-assistant, decoupled,
relational, contrastive, and ranking-distillation work all point to the same
risk: a small student can be harmed by ambiguous or poorly structured teacher
signals. The existing project also exposed a practical version of this issue:
the no-Q run trained a distribution head but was evaluated through an untrained
scalar head. The corrected, canonical action score is the expected value of the
128-bin predicted distribution.

CC-GAVN keeps the teacher distribution and the raw ChessBench Stockfish-return
bucket as complementary targets. A deterministic position-level development
split is excluded from all updates so that architecture and objective choices
are made without touching frozen MATE or official puzzle data.

## Open question and contribution target

No located paper establishes that a sub-5M, candidate-conditioned,
geometry-aware action-value model with exact rule-state symmetry can surpass a
released 9M searchless teacher under ChessBench's frozen puzzle protocol. That
is a testable gap, not a result. The paper contribution is viable only if the
development-selected CC-GAVN reproduces across seeds and improves the final
accuracy/latency Pareto frontier with confidence intervals. A corrected
evaluation of the prior GAVN is a required control, not a positive claim about
CC-GAVN.

## References

1. Ruoss et al. (2024), [Amortized Planning with Large-Scale Transformers: A Case Study on Chess](https://arxiv.org/abs/2402.04494).
2. Monroe and Chalmers (2024), [Mastering Chess with a Transformer Model](https://arxiv.org/abs/2409.12272).
3. Monroe et al. (2026), [Chessformer: A Unified Architecture for Chess Modeling](https://arxiv.org/abs/2605.19091).
4. Czech et al. (2023), [Representation Matters for Mastering Chess](https://arxiv.org/abs/2304.14918).
5. Jenner et al. (2024), [Evidence of Learned Look-Ahead in a Chess-Playing Neural Network](https://arxiv.org/abs/2406.00877).
6. Carroll and Beel (2020), [Finite Group Equivariant Neural Networks for Games](https://arxiv.org/abs/2009.05027).
7. Silver et al. (2017), [Mastering Chess and Shogi by Self-Play](https://arxiv.org/abs/1712.01815).
8. Schrittwieser et al. (2019), [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265).
9. Mirzadeh et al. (2019), [Improved Knowledge Distillation via Teacher Assistant](https://arxiv.org/abs/1902.03393).
10. Zhao et al. (2022), [Decoupled Knowledge Distillation](https://arxiv.org/abs/2203.08679).
11. Park et al. (2019), [Relational Knowledge Distillation](https://arxiv.org/abs/1904.05068).
12. Tian et al. (2019), [Contrastive Representation Distillation](https://arxiv.org/abs/1910.10699).
13. Busbridge et al. (2025), [Distillation Scaling Laws](https://arxiv.org/abs/2502.08606).
14. Tang and Wang (2018), [Ranking Distillation](https://arxiv.org/abs/1809.07428).
15. Furlanello et al. (2018), [Born Again Neural Networks](https://arxiv.org/abs/1805.04770).
