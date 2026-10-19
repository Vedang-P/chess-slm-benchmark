# Literature review: beating a searchless teacher below 5M parameters

Updated 2026-10-19. Primary sources checked through this date. This review
supersedes the earlier plateau addendum. Findings motivate experiments; their
reported gains are not predictions for our checkpoint.

## Conclusion

The highest-upside feasible route is **counterfactual refutation training**:
learn what defeats a candidate, ground every imagined board in exact rules,
relabel the student's own continuations, and distill the resulting planner
back into a one-pass scorer. Keep the trained CC-GAVN trunk and share its
weights across time. This changes the information available to the model
without breaching the user's strict 5M parameter cap.

Simply copying the 9M teacher more faithfully leaves its systematic mistakes
in the targets. Beating it requires additional information: stronger offline
engine analysis, alternative continuations, and supervision at future states.
Our inference model still uses no Stockfish. Learned bounded lookahead must
be reported separately from the original one-pass inference protocol.

## 1. Representations and the actual baseline

Ruoss et al. [1] establish ChessBench and action-value distillation from
Stockfish 16. They demonstrate substantial generalization and incomplete
transfer even at large scale. They do not establish that unlimited training
of a fixed tiny student closes the gap. Their dataset contains action-value
records; billions of records are not billions of distinct positions.

Chessformer [2,3] and Representation Matters [4] show that chess geometry,
square tokens, policy parameterization, and input features can be more
valuable than generic model scaling. CC-GAVN already incorporates much of
this architectural direction. Rebranding geometric attention would not be
novel. A factorized source/destination reply head is an appropriate compact
building block, with no claim that the head itself is new.

The local historical preview is 75.77% full puzzles and 92.425% MATE at
3.4M steps. The repository records the 9M reference at 86.13% and 98.725%.
The completed run's final artifacts remain unverified in this session.
`corrected-v2` names a recipe; its active architecture is `model_version=v1`.

## 2. Future-state learning and bounded planning

DiffuSearch [5] is the closest direct methodological competitor. Its study
uses a 7M network, four-step horizons, and 20 denoising steps. Its ablations
show that valid state transitions matter strongly. However, the reported
58.46% puzzle result for its larger training setting is below our recorded
preview, and its experiments used eight V100s. Its relative gains cannot be
added to our 75.77%. We borrow grounded future-state supervision, not its
absolute forecast or full diffusion training recipe.

MAV [6] explicitly compares external search and generated internal search
traces. Its 1B/2.7B models and search budgets are far outside our parameter
limit. It establishes that internal planning is prior art. AlphaZero [7],
MuZero [8], and Expert Iteration [9] likewise establish planning combined
with learned generalization. With an exact cheap chess transition available,
spending our small capacity on learning legal board dynamics is unnecessary.

Our proposed distinction is narrower: a sub-5M, checkpoint-compatible
candidate scorer, two learned opponent refutations, exact transitions,
root-perspective worst-line features, and distribution-preserving residual
fusion. The first opponent reply branches; later moves follow a learned
policy. This is bounded planning, not exhaustive minimax, proof search, or a
claim that both lines contain the true best defense.

## 3. Why on-policy counterfactual data is essential

DAgger [10] explains the distribution shift created when a learner's own
mistakes determine later observations. A planner trained only on oracle PVs
can see much worse boards when its reply predictor runs independently.
Therefore a later round must execute the learned continuation policy, query
the oracle on those exact states, and mix old and new examples. We implement
this aggregation rather than assuming teacher-forced accuracy transfers.

The October 2026 Queen paper [11] supplies a recent example of iterated
future-position consolidation and distillation. It uses a 4B chess-language
model and explanation supervision; it is not a tiny numeric Bellman learner.
The earlier notes incorrectly called it PAWN. Master Distillation [12] also
supports process-rich supervision and teacher-surpassing performance in its
own setting, but its 4B language model and puzzle protocol do not provide a
score forecast for CC-GAVN.

The 2025 strategic-RL study [13] uses a chess action-value model to reward
language-model actions. That is relevant to reward design, but introduces a
second language-model training problem we do not need. Distillation Scaling
Laws [14] reinforces the need to allocate compute between label quality and
student updates; its language-model scaling fits should not be numerically
extrapolated to our chess network.

## 4. Recurrence and the limits of learned lookahead

HRM [15] and Tiny Recursive Models [16] support reusing parameters to obtain
more computation. Their tasks and architectures differ from ours, and their
headline results are not evidence that a 5M chess agent will reach 95%.
We use explicit exact board transitions, not an unimplemented latent HRM.

Jenner et al. [17] provide evidence of future-move representations in Leela.
Cruz [18] finds strong context dependence. The later learned-priors study
[19], revised in 2026, cautions that an internal lookahead mechanism does not
guarantee that outputs actually follow it. Consequently our ablations must
compare against ranking-only and afterstate-only controls, and evaluate
predicted continuations rather than feeding oracle continuations at test time.
A planner that ignores future information is a failed mechanism even if an
auxiliary training loss improves.

## 5. Objectives, RL, and inexpensive controls

Stop Regressing [20] motivates classification-based value learning. Our
implementation uses two-bin interpolation, not the paper's HL-Gauss loss;
the existing 128-bin model should not be mislabeled as an HL-Gauss
implementation. Groupwise ranking complements value targets because many
winning actions have nearly identical WDL values but different tactical
continuations. Only proven oracle mate annotations supervise the mate head;
absence of a mate annotation is not proof of no mate.

Prior-Directed Exploration [21] is especially useful negative evidence
against making RL the entire plan. On its own 100k suite it reports roughly
93.9% to 94.9% puzzle accuracy; puzzle-only tuning can lose substantial playing
strength. Its anchor is the model's MCTS-derived policy prior, not a teacher's
return-bin distribution. Those distributions must not be conflated. Self-play
RL is a later experiment after reply/value competence, with game strength
measured alongside puzzle scores.

Consistency Checks [22] motivates exact-rule and symmetry diagnostics.
Reflection is valid here only without castling rights. Model Soups [23] is a
cheap control when compatible checkpoints share a basin; it is not our main
high-upside intervention. Mixing checkpoints with different architectures,
label transformations, or known training defects is not automatically safe.

## Novelty assessment and falsifiable claim

**Assessment: plausible combination novelty; individual components are known.**
The strongest claim is an accuracy/compute result: counterfactual refutation
aggregation improves the sub-5M frontier beyond both a frozen 9M teacher and
matched post-training controls. No located source establishes this exact
combination, but this search does not prove priority.

Reject the mechanism claim if ranking-only matches it at equal wall time,
if gains disappear on independent puzzles, if oracle-fed paths account for
the benefit, or if it only wins through much higher unreported inference
compute. See `POSTTRAINING-PLANNER-PLAN.md` for controls, costs, code, and targets.

## Sources inspected

1. Ruoss et al., 2024 — [Amortized Planning / ChessBench](https://arxiv.org/abs/2402.04494).
2. Monroe & Chalmers, 2024 — [Mastering Chess with a Transformer Model](https://arxiv.org/abs/2409.12272).
3. Monroe et al., 2026 — [Chessformer](https://arxiv.org/abs/2605.19091).
4. Czech et al., 2023/2024 — [Representation Matters](https://arxiv.org/abs/2304.14918).
5. Ye et al., ICLR 2025 — [DiffuSearch](https://arxiv.org/html/2502.19805v1).
6. Schultz et al., 2024/2025 — [Mastering Board Games by External and Internal Planning](https://arxiv.org/html/2412.12119v3).
7. Silver et al., 2017 — [AlphaZero](https://arxiv.org/abs/1712.01815).
8. Schrittwieser et al., 2019/2020 — [MuZero](https://arxiv.org/abs/1911.08265).
9. Anthony et al., 2017 — [Expert Iteration](https://arxiv.org/abs/1705.08439).
10. Ross, Gordon & Bagnell, 2010/2011 — [DAgger](https://arxiv.org/abs/1011.0686).
11. Bhaskar, Cheng & Chen, October 2026 — [Queen: Language Models that Play Chess and Explain Their Moves](https://arxiv.org/html/2610.03695v1).
12. Tang et al., 2026 — [Master Distillation](https://arxiv.org/abs/2603.20510).
13. Hwang et al., 2025 — [Post-training Insights from Learning Chess](https://arxiv.org/abs/2507.00726).
14. Busbridge et al., ICML 2025 — [Distillation Scaling Laws](https://arxiv.org/abs/2502.08606).
15. Wang et al., 2025, revised September 2026 — [HRM](https://arxiv.org/abs/2506.21734).
16. Jolicoeur-Martineau, 2025 — [Tiny Recursive Models](https://arxiv.org/html/2510.04871v1).
17. Jenner et al., 2024 — [Evidence of Learned Look-Ahead](https://arxiv.org/abs/2406.00877).
18. Cruz, 2025 — [Understanding Learned Look-Ahead](https://arxiv.org/abs/2505.21552).
19. [The Algorithm Is Not the Behavior](https://arxiv.org/abs/2508.21380), 2025, revised June 2026.
20. Farebrother et al., 2024 — [Stop Regressing](https://arxiv.org/abs/2403.03950).
21. Miłosz, Duch & Grabowski, August 2026 — [Prior-Directed Exploration](https://arxiv.org/html/2608.27757v1).
22. Fluri, Paleka & Tramèr, 2023 — [Consistency Checks](https://arxiv.org/abs/2306.09983).
23. Wortsman et al., 2022 — [Model Soups](https://arxiv.org/abs/2203.05482).
