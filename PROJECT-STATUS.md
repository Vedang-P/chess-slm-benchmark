# Project Status

Updated 2026-11-02.

## Best verified model

`ccgavn-5m-seed0-v2/checkpoint-3896658` is a 4,762,088-parameter, one-pass
CC-GAVN model (3,895,352 parameters after the verified function-preserving
transplant). Its frozen evaluation is archived at
`eval-results/ccgavn-5m-seed0-v2-3896k-final/eval-summary.json` in the
[Hugging Face dataset](https://huggingface.co/datasets/vedangfake/chess-slm-benchmark).

| Evaluation | CC-GAVN | 9M reference |
| --- | ---: | ---: |
| MATE, 4,000 decisions | 3,710 (92.75%) | 3,949 (98.72%) |
| Official puzzles, 10,000 full sequences | 7,620 (76.20%) | 8,613 (86.13%) |

The benchmarks were inspected during development; they are fixed comparison
sets, not untouched holdouts. MATE rows repeat positions. Model selection in
the current experiment uses a separate held-out validation set (below), and a
sealed test set is scored once at the end.

## What did not work

Two post-training attempts from this checkpoint lowered puzzle accuracy:
root-policy RL (v1/v4: 56.87% / 51.07%) and the tactical distillation
continuation (nine evaluations between 55.6% and 64.1%, last 55.89% at step
4,346,658). That continuation restarted at 26% of the original peak learning
rate, trained only on first-decision puzzle roots with no broad replay, and
overwrote general play. It was stopped on 2026-11-02; its checkpoints and the
labeled puzzle shards remain on Hugging Face.

## Current experiment: conservative continuation (cont1)

Same 3.9M architecture, restarted from checkpoint 3,896,658 with:

* constant 2e-5 learning rate after warmup, and an EMA of the weights (the EMA
  weights are what is evaluated);
* every batch mixes the original teacher-distilled corpus (replay) with whole
  tactical roots from the labeled puzzle shards;
* a listwise loss over each tactical root's legal moves;
* held-out validation: 4,000 Lichess puzzles disjoint from the official 10,000
  and rating-matched to them, scored with the exact harness rules during
  training, plus a regression guard that halts the run if validation stays more
  than 3 points below the starting model;
* sequential 11.5-hour learner sessions across the five Kaggle accounts; each
  session saves a complete checkpoint to Hugging Face and the next resumes it.

## Sprint results (30,000 steps each, one Kaggle session per arm)

All arms start from checkpoint 3,896,658 with the same trainer, seed and data
stream; only the listed change differs. Validation is the 4,000 held-out puzzles
(start: 77.98%); the official 10,000 puzzles and 4,000 MATE rows are logged once at
the end and were not used to choose anything.

| Arm | Change | Validation | Official puzzles | MATE | Broad dev loss |
| --- | --- | ---: | ---: | ---: | ---: |
| start | verified checkpoint | 77.98% | 7,620 (76.20%) | 3,710 | 2.878 |
| a0 | broad replay only (control) | 77.70% (-0.27) | 7,631 (76.31%) | 3,703 | 2.879 |
| a1 | 30% tactical roots, pointwise losses | 79.72% (+1.75) | 7,910 (79.10%) | 3,671 | 2.978 |
| a3 | 50% tactical roots + listwise loss | 87.15% (+9.18) | 8,619 (86.19%) | 3,804 | 3.078 |
| a4 | 30% tactical + listwise, lr 5e-5 | 86.75% (+8.78) | 8,625 (86.25%) | 3,764 | 2.996 |

What it shows:

* The low constant LR with EMA is safe: the control moved by -0.27 points, so the
  earlier regression came from the data and loss design, not from continuing.
* The listwise loss over each root's legal moves is what matters. The same tactical
  roots with a pointwise loss gain 1.75 points; with the listwise loss they gain 9.
* Both listwise arms are above the 86.13% reference (8,613 puzzles) at step 30,000,
  by 6 and 12 puzzles. That is inside the roughly 0.35-point sampling error of a
  10,000-puzzle score, so it is a tie, not yet a win. The sealed test puzzles and
  longer training decide it.
* The gain is paid for on the broad corpus (dev loss +4% for a4, +7% for a3) and MATE
  stays below the 9M reference (94.1% and 95.1% vs 98.72%). This is a tactics
  specialist trained on puzzle-distribution positions (benchmark puzzles and
  positions excluded), not a uniformly stronger model.
* 31% of the starting model's validation failures were at later plies, not the first
  decision; the labeled roots only cover the first decision.

The main run is arm a4 continued without a step limit (resumed from its own
checkpoint, optimizer and EMA). Stage 2 compares exact-expectation policy
optimisation (`--rl-weight 1.0`, KL anchor 0.05) against a control with the same
branch point and data (runs r1 and c1, both from the a4 EMA weights at step 30,000).

The [monitor](https://chess-slm-monitor.vedangpandeyy.workers.dev/) shows
validation curves per run, official evaluations, pool quota and supervisor
state.

## Later-ply roots (cont2 experiment)

Question: does dense Stockfish supervision of the *later* decisions on a puzzle's solution
line lift puzzle accuracy? In arm a4 at step 30,000 the 4,000 validation puzzles failed 273
times at the first model decision and 257 times at later plies (official 10,000: 687 vs 688).
The tactical pack holds only first-decision roots, and the listwise loss cut first-decision
failures by 55% but later-ply failures by only 7%, so later plies are now half of the errors.

Data (`puzzle-ply-v1` on the Hub). For every Lichess puzzle with a stored move at that ply,
the board after `Moves[0..ply-1]` for ply 3 and 5 (the solver's second and third decision),
labelled with every legal move by the frozen Stockfish 16, 50 ms root-move recipe. 5,227,036
ply-3 roots (1,046 shards) and 2,118,555 ply-5 roots (424 shards) of 5,000 puzzles each.
Official puzzles and the validation/test puzzles are excluded by id, and every position in
the benchmark exclusion set (the 104,092 reflection-orbit hashes of the cont-v1 pack) is
dropped by hash. Twenty-five CPU kernels (five per account, four Stockfish processes each)
label the shards and the cont tick keeps them alive; `cont-v2/ply-pack` is the training pack
built from the finished shards. The sealed test puzzles are never read.

Labeling fleet. Workers are Kaggle CPU kernels `ccgavn-ply-label-w00` to `w24` (five per
account, four Stockfish processes each, 10.5-hour sessions that resume partial shards from
the Hub). A shard is the next 5,000 eligible puzzles in source order; the benchmark-position
drop happens inside it, so a finished shard holds 4,980-5,000 roots (the drop rate was
0.3% on the first ply-5 shards, against 0.08% for first-decision roots). The cont tick
relaunches any worker that stopped while it still has unfinished shards, and the monitor
snapshot carries a `label_fleet` summary (phase, blocked workers, shards done). Each worker
writes `cont-pipeline/status/ply-label-wNN.txt`; one that failed is not relaunched on the
same commit. Create `puzzle-ply-v1/PAUSE` on the Hub to stop launching (running kernels
finish their session; delete the kernels to stop them now) and delete it to resume.
Four engines per kernel make each 50 ms search see about 39% fewer nodes than a lone engine
(490k against 799k nodes/s per engine), but on 60 roots the mean win-probability gap to
solo labels was 0.0100 and 0.0085 against 0.0088 between two four-engine passes, and the
best move agreed on every root: no shift measurable at that sample size. A larger check against
ground truth: on 15,000 roots each, the puzzle's stored solution is the Stockfish-best move in
99.85% of the solo-labeled first-decision roots (shards 0-2), 99.84% of the four-engine ply-3
roots and 99.86% of the ply-5 roots (mean regret of the stored solution 0.00012, 0.00006 and
0.00006).

Experiment. Two runs start from the EMA weights of the same `ccgavn-cont1-a4` checkpoint with
the main-run recipe (seed 0, 1,434 broad rows and 20 first-decision groups per step, lr 5e-5,
listwise loss), 25,000 steps each (about 4 GPU hours), validation every 2,500 steps:

* `ccgavn-cont1-p0` (control): `--ply-groups 0`;
* `ccgavn-cont1-p1`: `--ply-groups 10`, ten extra ply-3/5 roots per step drawn uniformly from
  the pack's training roots (about 70% ply 3, 30% ply 5) on their own RNG stream, so the
  broad and first-decision draws are identical to the control's.

As launched (2026-11-02, before either run started): both runs branch from
`ccgavn-cont1-a4/checkpoint-50000` (EMA weights; validation there 3,495 / 4,000 = 87.375%,
268 first-decision and 237 later-ply failures). The pack is `cont-v2/ply-pack` with manifest
digest `e3e04e11...bf2b4`, built from the 75 shards finished at 14:32 UTC: ply 3 shards
0-52 (264,801 roots) and ply 5 shards 0-21 (109,741 roots), 370,682 training roots, 3,860
held-out dev roots, 458 benchmark-position roots dropped by the labelers. Control on
`vedangpandeyyy`, treatment on `softmaxsimp`, launched with `--max-steps 25000 --val-every
2500 --ckpt-every 2500 --val-detail`. A replay of the weekly quota (r1 and c1 still running)
shows the main line keeps training until the Friday reset with both runs taken out.

Decision rule, fixed before either run starts. The only data used for the decision is the
4,000-puzzle validation set (EMA weights, the harness rules), compared puzzle by puzzle
(`scripts/compare_ply_runs.py`; the official 10,000 puzzles are not read until the verdict is
recorded below). With gain = treatment solved minus control solved:

| Check | Requirement (all four for ADOPT) |
| --- | --- |
| effect | gain >= +20 puzzles (0.5 point) at step 25,000 and exact McNemar p < 0.05 |
| consistent | mean gain over steps 20,000 / 22,500 / 25,000 >= +15 |
| mechanism | later-ply failures <= 90% of the control's and first-decision failures <= 105% of the control's at step 25,000 |
| broad corpus | broad dev loss <= 101% of the control's at step 25,000 |

REJECT when the final gain is <= +8 puzzles or later-ply failures did not drop; anything
else is INCONCLUSIVE (the next step would be a second seed pair, not a change to the main
run). ADOPT means "propose adding `--ply-groups` to the main run", which is the owner's call.
The 86.13% gate and the evaluation are unchanged and are judged only on the official 10,000
puzzles and the sealed test puzzles, never on this experiment.

What the design cannot show: a single pair does not measure run-to-run training noise (the
consistency and mechanism checks guard against a lucky final evaluation, not against a
different seed); the treatment also sees about 50% more tactical rows per step (17% more rows in all), so a gain
that is not concentrated in later-ply failures is evidence of generic tactical supervision
rather than of later-ply supervision; and validation is rating-matched to the official
puzzles but drawn from the same Lichess distribution as the training roots.

### Result on validation (recorded before the official 10,000 puzzles were read)

Both runs finished 25,000 steps (`ccgavn-cont1-p0` control, `ccgavn-cont1-p1` with
`--ply-groups 10`). The branch point scored 3,495 of 4,000 on validation (87.375%).
Puzzles solved on the 4,000 held-out validation puzzles, EMA weights:

| Step | Control | Treatment | Gain |
| ---: | ---: | ---: | ---: |
| 2,500 | 3,497 | 3,503 | +6 |
| 5,000 | 3,487 | 3,515 | +28 |
| 7,500 | 3,479 | 3,513 | +34 |
| 10,000 | 3,480 | 3,511 | +31 |
| 12,500 | 3,472 | 3,515 | +43 |
| 15,000 | 3,480 | 3,522 | +42 |
| 17,500 | 3,473 | 3,520 | +47 |
| 20,000 | 3,483 | 3,521 | +38 |
| 22,500 | 3,482 | 3,519 | +37 |
| 25,000 | 3,484 (87.10%) | 3,524 (88.10%) | +40 |

The rule's four checks, all computed by `scripts/compare_ply_runs.py` from the saved
per-puzzle results: **ADOPT**.

| Check | Observed |
| --- | --- |
| effect | +40 puzzles at step 25,000 (82 solved only by the treatment, 42 only by the control), exact McNemar p = 0.0004 |
| consistent | mean gain over steps 20,000 / 22,500 / 25,000 is +38.3 (+38, +37, +40) |
| mechanism | later-ply failures 217 vs 250 (86.8% of the control's); first-decision failures 259 vs 266 (97.4%) |
| broad corpus | broad dev loss 3.7916 vs 3.7693 (100.6% of the control's) |

Failures by ply of the first wrong decision (control / treatment): ply 1: 266 / 259,
ply 3: 157 / 132, ply 5: 70 / 58, ply 7 and later: 23 / 27. The gain is the intended one:
the second and third decisions, the plies that were labeled.

Reading it carefully. The control does not improve on the branch point (3,495 to 3,484;
the optimizer restart costs a few puzzles and the main run itself has been flat at 87.4% to
87.5% on validation since step 50,000), so the treatment's +29 over the branch point is the
whole improvement that either run shows. The gap opens within the first 5,000 steps and then
stays between +31 and +47 instead of growing, so more steps of the same data may not add more.
The treatment pays 0.6% on the broad dev loss. This is one seed pair on one validation set:
it supports proposing `--ply-groups` for the main run, which needs the owner's decision (the
recipe identity changes, so resuming needs `--allow-recipe-change`), and it says nothing yet
about the official puzzles or the sealed test puzzles.

### Official 10,000 puzzles (read after the verdict above was committed)

| Run | Puzzles | MATE (4,000) | Failures first / later |
| --- | ---: | ---: | ---: |
| control `p0` | 8,642 (86.42%) | 3,795 | 678 / 680 |
| treatment `p1` | 8,740 (87.40%) | 3,820 | 672 / 588 |

Paired: 189 puzzles solved only by the treatment and 91 only by the control (McNemar
p < 0.00001). Later-ply failures by ply (control / treatment): ply 3: 440 / 377, ply 5: 175 / 150.
The pattern matches validation: the first decision barely moves and the labeled plies carry the
gain. MATE rises a little (3,820 against 3,795) and stays below the 9M reference's 3,949.

Against the 8,613-puzzle gate the control is +29 and the treatment +127. The control already
clears 86.13% by a margin inside the roughly 0.35-point sampling error of a 10,000-puzzle
score, so the gate does not separate the runs; the paired difference does. The official puzzles
were inspected during development, so they are a fixed comparison set and not an untouched
holdout, and the sealed test puzzles have not been scored.

### Adoption into the main run (2026-11-02, on the owner's go-ahead)

The main run (`ccgavn-cont1-a4`) now trains with `--ply-groups 10`, the dose that was tested, and
`--val-detail`, from its own pack `cont-v2/ply-pack-main-v1` (manifest digest `4ae8541b...ec991`):
every shard finished at 19:55 UTC, 352 in all. Ply 3 contributes 250 shards (1,249,044 roots) and
ply 5 contributes 102 (508,792 roots); 1,740,113 roots are for training and 17,723 are held out
as a dev fold; 2,164 benchmark-position roots were dropped. At ten groups per step that is about
one pass over the pack per day, close to the first-decision pack's rate. The experiment's pack
stays untouched at `cont-v2/ply-pack`, so `p1` remains reproducible.

It takes effect at the next learner session: the running one started at 11:19 UTC with the old
recipe and ends about 22:49 UTC, and the tick then resumes the same checkpoint, optimizer and EMA
with the new flags. A run saved before ply support may take the pack's identity in on that first
resume and nothing else may change; afterwards the groups and the pack are pinned like the rest of
the recipe (changing them needs `--allow-recipe-change`, which is also the way back). Checked
before the change was pushed: the new trainer, with the groups off, continues a checkpoint
written by the previous trainer bit for bit, and with the groups on it resumes the same
checkpoint without any override.

What was not tested: the experiment restarted from the EMA weights with a fresh optimizer, while
the main run resumes with its optimizer state, so the restart dip should not recur but that exact
path is new. To watch: validation every 10,000 steps with failures by ply (`val_failed_by_ply`),
`ply_dev_hit`, and the broad dev loss (the treatment paid 0.6%). The main run's official
evaluation at step 100,000 still uses the old recipe and is the "before" number; the one at step
200,000 is the first "after". The sealed test puzzles stay untouched, and the 86.13% gate is
judged only on the official and sealed sets.
