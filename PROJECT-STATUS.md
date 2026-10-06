# Project Status — Chess SLM Benchmark (2026-08-30)

## 2026-10-06 — Independent review: corrections, measured audits, corrected-v2 (training DISARMED)

An independent code review produced 19 findings. Status of each, with the fix
or the measured number. Nothing here is retroactive: checkpoints already on HF
under `ccgavn-5m-seed0/` were produced by the legacy recipe and are kept as
the historical record.

**Training- and data-integrity fixes (landed 2026-10-06):**
1. LR schedule updated only optimizer group 0 (embeddings/biases/LayerNorm
   stayed at 5e-4 while logged LR decayed). Fixed: `set_learning_rate` applies
   every group (`scripts/data_hygiene.py`), tested.
2. Continuation built the full-budget shard schedule and indexed it by global
   step, so whole shard blocks were skipped. Reproduced on the live legacy
   schedule: **41 of 102 tags were never sampled at all after step 1.62M**.
   Fixed: stage-relative schedule (`--stage-start-step`), schedule digest
   pinned in checkpoints.
3. Horizontal reflection was invalid for positions with castling rights
   (e1g1 castle becomes e1b1). Fixed: reflect only positions with no castling
   rights, tested.
4. Development-fold exclusions applied only to the first shard. Fixed:
   `eligible_mask` applies the folded hash + benchmark exclusions to every
   shard.
5. Development split was not augmentation-aware. Fixed: `position_hashes`
   hashes the reflection orbit, tested for split invariance.
6. ChessBench training lacked MATE/puzzle exclusions. Fixed at the trainer
   level (`--exclusion-puzzles`); measured overlap below.
13. Malformed records became zero-filled training examples. Fixed: the parser
    raises and writes no dataset.
14. Dev sampling consumed the training RNG and checkpoints were wall-clock
    triggered, so checkpoint timing changed later batches. Fixed: separate
    seeded dev RNG; CUDA RNG saved/restored.
15. Deployments were not pinned. Fixed for the stage-2B kernel and eval
    kernel: the watcher embeds the recorded `source_commit` and the kernel
    fetches exactly that commit; the trainer refuses a resume whose checked-out
    commit differs from the checkpoint's recorded one unless
    `--allow-code-drift` is passed (which marks every later checkpoint).

**Model fixes (new `--model-version v2`; v1 stays for the warm start):**
16. Global metadata was embedded + positionally added and then averaged, which
    is permutation-invariant. v2 adds per-field gates (init 1.0), making field
    identity observable; tested.
17. Half of the dynamic attention bias (the query-side term) is constant along
    each softmax row and provably cancels. v2 keeps only the key-side term;
    v1 keeps its dead parameters for checkpoint compatibility.

**Measured data audit (`scripts/audit_train_contamination.py`; 4 shards,
8,000,000 sampled rows, same exclusion digest the trainer uses):**
- benchmark (MATE + puzzle) overlap in ChessBench: **0.0003%** (~2-8 rows per
  2M; ≈5k rows over the 1.75B-row union) — small but nonzero, and now excluded;
- development fold: **0.999%** (≈1/100 by construction), now excluded in every
  shard;
- duplicate positions: **0.241%** within a shard; **57,151** cross-shard
  duplicate positions across the four sampled shards (~0.95% of later shards'
  rows). MATE itself has 4,000 rows over **2,952 unique positions** (1,030
  positions in more than one subset, often with swapped candidates), so
  row-level binomial intervals overstate precision.

**Evaluation fixes:**
8. MATE non-independence: per-example JSONL + `scripts/analyze_mate.py`
   (position-clustered bootstrap CIs, paired McNemar on common rows).
9. Malformed MATE rows were silently skipped. Fixed: exact expected counts
   (4,000/10,000) and hard failure on any malformed row or wrong total.
10. Completion checks were weak (summary/DONE written even after failure).
    Fixed in both eval kernels: DONE only on returncode 0 + exact totals;
    `ensure_eval_preview.py` verifies the summary, not its existence.
11. No per-example persistence/resume. Fixed: `--examples-out` JSONL is
    appended as rows are scored and uploaded periodically; a killed session
    resumes instead of rescoring. Follow-up review finding (2026-10-06, fixed
    same day): the eval kernel shared one examples directory across targets,
    so a checkpoint without saved rows inherited the previous checkpoint's
    JSONL and could republish its scores. Each target now gets
    `examples-<step>`, and `eval_gavn.py` validates a checkpoint/dataset/score
    identity (including the state-file hash) before accepting saved rows;
    mismatch aborts and the kernel rescoring from scratch. Second follow-up:
    identity.json now uploads before the JSONL, both refusals (wrong identity
    and interrupted upload with rows but no marker) share the
    `eval examples unusable` token the kernel detects, and recovery deletes
    the remote artifacts before rescoring so a repeated crash cannot re-download
    the same incomplete state. Regression tests:
    `test_eval_resume_is_checkpoint_scoped`,
    `test_eval_kernel_scopes_examples_per_checkpoint`,
    `test_eval_interrupted_upload_recovers`.
12. Watcher/kernel milestone lists disagreed (watcher to 3,896,658, kernel to
    1,620,000). Fixed: one list in `ensure_eval_preview.py`, rendered into the
    pushed kernel, so they cannot drift.
7. Repeated preview evaluation of the frozen MATE/puzzle sets means they are
    **not an untouched holdout**. This cannot be undone; it is now stated
    explicitly. Model selection uses the development loss only; milestone
    evals are labeled monitoring/dev-diagnostic; the one-shot final at
    3,896,658 is labeled as a final protocol run on previously exposed sets.

**Documentation/provenance (18/19):** causal claims below are downgraded to
hypotheses; teacher-entropy homogeneity does not establish equivalent shard
content; "frozen" now means an archived protocol run at a fixed commit, not an
unseen holdout; the legacy 1.62M→1.715M trajectory (61/102 tags, no
exclusions) is disclosed wherever the continuation is described.

**Corrected-v2 staging (training DISARMED):** new prefix
`ccgavn-5m-seed0-v2`, warm start from the latest complete legacy checkpoint
(currently `checkpoint-1715000`), stage-relative schedule over all 102 tags,
exclusions active, all-group LR. `scripts/watch_2b.py` will not push a
training kernel unless armed (`configs/ccgavn-2b-ARMED` or `CCGAVN2B_ARMED=1`)
— by design, pushing code cannot start training.

## 2026-10-05 — 1B continuation COMPLETE; final frozen eval

The 1B continuation reached **1,620,000 steps** on 2026-09-30T21:51Z
(final train_loss 3.685, cosine lr → 4.7e-16, clean finish) on the frozen
blend (`configs/ccgavn-1b-shard-tags.json`: 61 tags = 10 original + 51 new,
929.0M new rows). All five `ccgavn-1b` kernels are stopped (one COMPLETE,
four CANCEL_ACKNOWLEDGED after the winner finished). The complete frozen
protocol was archived with returncode 0:

| checkpoint | MATE (4,000) | Puzzles (10,000) |
|---|---|---|
| CC-GAVN @320k | 89.05% (3,562) | 59.95% (5,995) |
| CC-GAVN @1620k (1B) | **91.70%** (3,668) | **74.31%** (7,431) |
| Ruoss 9M (target) | 98.72% | 86.13% |

1B adds **+2.65pp MATE / +14.36pp puzzles** over 320k at the same 4.76M
params. The 100k-step preview evals show the curve flattening: MATE was
91.42% at 1300k and 91.7% at 1620k; puzzles 74.08% → 74.31% (the 1600k
preview read 91.9% / 74.25%, so the final stretch moves within ±0.3pp).
Remaining gap to the 9M teacher: **−7.0pp MATE, −11.8pp puzzles**.

Artifacts: HF `ccgavn-5m-seed0/checkpoint-1620000/` (config, metrics,
state.pt) and `eval-results/ccgavn-5m-seed0-1620k-frozen/` (eval-full.log,
eval-summary.json, run-status.txt). Reporting cleanup same day: the stale
pre-fix crash traceback in the top-level `run-status.txt` was replaced
with a DONE status, and the monitor's `eval1b` stage now matches the
`1620k-frozen` archive (all five dashboard stages at 100%).

## 2026-10-05 — Stage 2B-A launched (102 tags); full-108 labeling restarted

User decisions (2026-10-05): (1) start stage 2B-A now on the **102 built
tags** (frozen 61 + 41 remaining built shards = 1.746B rows) and append the
last 16 (~268.7M rows) after they are labeled; (2) step budget = the
pre-registered 2.67-epoch proportion over the 723.0M added rows → new cosine
schedule to **step 3,896,658**; (3) **warm start** from checkpoint-1,620,000
(weights + optimizer state), not a fresh run; (4) keep the contiguous
P000/P001 schedule blocks — the dev spikes are transient interference with
full recovery, and changing the schedule would change the recipe that
produced +14.4pp puzzles.

- `configs/ccgavn-2b-shard-tags.json`: 102 tags, union 1,746,290,483 rows.
- `scripts/train_ccgavn.py --allow-shard-superset`: deliberate stage
  transitions (checkpoint tag set must be a subset of the new set).
- `kernels/ccgavn-2b/train_2b.py` + `scripts/watch_2b.py` (cloud keep-alive,
  crash backoff): launched on `vedangpandeyyy/ccgavn-2b` 2026-10-05.
- Labeling fleet re-pushed to finish the full 108-shard plan (16 shards left,
  across 4 accounts); `ensure_build_2b.py` target is now the full 1.921B rows.
- Preview-eval milestones extended to 3.8M + a `3896k-frozen` final.

First shard-content insights (curve + teacher sampling, 2026-10-05):
ChessBench shards look statistically similar (teacher entropy 2.59–2.62
nats, top-prob ~0.30; per-block dev effects within ±0.02, Spearman vs
entropy/rows 0.17 / −0.12). This is a hypothesis, not an equivalence result:
similar teacher entropy does not establish equivalent shard content, and the
2026-10-06 audit found only ~0.0003% benchmark overlap and ~0.24% duplicate
positions, so the shards are not simply repeats either. The puzzle curriculum
is the only qualitatively different type: softer teacher (2.88 nats) and the
only blocks that move dev (+0.17 to +0.22 during the block, fully recovered
after). That every large MATE/puzzle jump so far is *downstream* of those
blocks is a timing observation, not evidence of causation; data quantity,
training duration, curriculum, and schedule all changed together.

## 2026-09-20 — 1B continuation: push-only gate bug found & fixed; frozen corpus

The 1B continuation had not started since `checkpoint-320000` landed
(2026-09-19T18:51Z). Root cause: `kaggle kernels push` uploads only the
`code_file`, so the sibling `shard_rows.json` written by `scripts/watch_1b.py`
never reached the kernel; `train_1b.py` silently fell back to an empty map,
counted **0M / 920M** rows, waited 900s and exited ("prerequisites not ready
within the wait window") on every push (kernel log confirmed, v17-v20 loop,
~0.25 GPU-h per ~20 min burned). The corpus side was fine: 92/108 planned
shards (1.652B new rows) are on HF.

Fix (engineering only): `train_1b.py` now takes the frozen shard-tag list as a
push-time placeholder (hard-fails if not embedded), gates exactly on all
frozen tags + `checkpoint-320000`; `watch_1b.py` injects the list, gained
`--check-only`; `train_ccgavn.py` gained `--shard-tags-file`, stores
`shard_tags` in checkpoint configs, and refuses a resume with a different
frozen set; `ShardManager` accepts an explicit `tags` list.

Frozen 1B-first corpus (user decision 2026-09-20): `configs/ccgavn-1b-shard-tags.json`
= 10 original tags + 51 new planned shards in slice order (built-only) =
**929.0M new rows**. The label fleet is stopped (sessions ended; the CI stop
rule prevents re-pushes); the extra ~720M built rows remain on HF for the
deferred 2B stage. Details and the epoch arithmetic: `SCALING-PLAN-2B.md`.

## 2026-09-19 — CC-GAVN 320k run + 2B scaling plan

**Current run (automatic):** `ccgavn-5m-seed0` continued from checkpoint-160000
to 320k steps on the 10-shard mix (8 ChessBench shards + 2 puzzle-curriculum
shards, 14.9% of samples). Watchers are fully cloud-side (Cloudflare cron -> GitHub Actions; no local machine). the auto-eval kernel (plus `ensure-eval-320k` CI safety net)
runs the frozen protocol when checkpoint-320000 lands.

**Frozen results so far (full sets, same protocol):**

| model | MATE (4,000) | Puzzles (10K) |
|---|---|---|
| CC-GAVN @160k | 87.38% | 51.86% |
| CC-GAVN @260k (preview) | 88.78% | 57.86% |
| Ruoss 9M (target) | 98.72% | 86.13% |

The +6.0pp puzzle jump at 260k coincides with the tactic-curriculum blocks
(downgraded from "is the effect": curriculum, data, and schedule changed
together, so this is a hypothesis, not an isolated control). MATE +1.4pp.

**Stockfish-anchored ladder (CC-GAVN @160k):** 1W 20D 19L vs UCI_Elo=1400
(score 0.275, implied ~1230 anchored Elo); the ladder stopped early by the
user's rule (do not climb when the low anchor is already lost). Artifacts:
HF `elo-results/ccgavn-160k-stockfish-ladder/` (40 PGNs + results.json).

**1B-first scaling plan (user decision 2026-09-19):** see `SCALING-PLAN-2B.md`.
The labeling fleet runs on the largest 108 shards but stops once ~920M new
rows (~1B total) are on HF; the CI watcher enforces the stop and never
re-pushes after that. The 2B stage is deferred until 1B results justify it.
Largest 108 remaining shards (~1.92B rows) are being teacher-labeled by four
Kaggle accounts (2 slices each) with the same 9M-teacher recipe, then the same
CC-GAVN recipe is retrained at 2.67 epochs. Rule: the fundamental recipe never
changes to save compute; only engineering (streaming/parallelism) changes.


Current working status. Keep updated as the direction changes.

## The objective
Find the best searchless chess action-value model below 9M parameters, with a
primary target of matching or exceeding Ruoss 9M on MATE and official puzzles,
then test whether the same method can approach the 136M/270M accuracy frontier.
Target: Efficient and On-Device AI Agents Workshop @ NeurIPS 2026. Compute:
Kaggle free tier only (T4/P100, ~30h/week/account, 3 accounts). Honesty
protocol (corrected 2026-10-06): MATE and puzzle positions are excluded from
training, and model selection uses the development split only. The MATE/puzzle
sets have been scored repeatedly during development (milestone monitoring), so
they are **not an untouched holdout**; final numbers must state that exposure,
and no claim of a clean holdout may be made for them.

## Direction history

### 1. RLVR / GRPO (ABANDONED — measured infeasible)
GRPO on gemma-4-E2B-it with Stockfish rewards, uncapped thinking.
- v9 (256 cap): trained but all rollouts clipped, reward 0 (model never
  emits EOS on MoveA/B prompt; `MoveMoveMove` repetition).
- v11 (G8 uncapped): 2h timeout, 0 rollouts completed.
- v12 (G2 uncapped): step 0 never completed in 3h+; killed.
- Built + verified live-thinking infra (token stream -> stdout + HF
  live-thinking.txt every 15s; demo-2: real reasoning, terminates with
  `MoveA:...<turn|>`, 4695 tok/163s).
- Verdict: uncapped thinking on base 2B never terminates on P100. RLVR
  infeasible at this compute. User decision: pivot.

### 2. EGSD / SIL (PROPOSED, PARKED)
- EGSD (engine-graded self-distillation): user rejected ("caveman SFT with
  a filter + a loop"). Script + plan exist, not active.
- SIL (Search-in-Language): verbalized engine search trees + self-consistency
  voting. 5000 traces built (`results/searchlang-traces.jsonl` + _sft).
  Parked; traces remain reusable data.

### 3. Improve searchless_chess (ACTIVE DIRECTION)
Open-source DeepMind chess models (Ruoss et al., NeurIPS 2024):
9M/136M/270M transformers, searchless, trained on ChessBench (10M games,
15.3B Stockfish-16 action-values). 270M = 2895 Lichess Elo vs humans (GM).

**Measured result (exact MATE subsets, local CPU, official ActionValueEngine):**
- 9M: **98.2%** (982/1000) — complete, full 1000 rows
- 9M: **98.9%** on tactic, both, and strategy/full (989/1000 each)
- 136M: **99.4%** on all four subsets (3976/4000 combined)
- 270M: **99.4%**, **99.4%**, **99.4%**, **99.5%** on noexplain, tactic, both, strategy/full (3977/4000 combined)
- 9M official puzzle harness: **86.13%** (8613/10000 full solution sequences)
- gemma-4-E2B base baseline: 58.1%


## Key files
- `searchlang-plan.md` — parked SIL plan
- `egsd-plan.md` — parked EGSD plan
- `scripts/eval_searchless_mate.py` — MATE eval for searchless models
- `scripts/build_search_traces.py` — SIL trace builder (5000 done)
- `scripts/egsd_sample.py` — EGSD sampler (parked)
- `scripts/train_mate_grpo.py` — GRPO trainer (abandoned; keep for infra:
  live traces, HF checkpoints, _ThinkingProcessor)
- `results/searchlang-traces.jsonl` (+ _sft) — 5000 verbalized search traces

## External resources (searchless_chess)
- Paper: https://arxiv.org/abs/2402.04494 (Ruoss et al., NeurIPS 2024)
- Repo (open, Apache-2.0): https://github.com/google-deepmind/searchless_chess
- Weights: storage.googleapis.com/searchless_chess/checkpoints/{9M,136M,270M}.zip
- Dataset ChessBench: data/download.sh (10M games, 15.3B labels)
- MAV successor (weights NOT released): https://arxiv.org/abs/2412.12119

## Environment gotchas (measured)
- 2024 checkpoints are orbax-ocdbt: need jax 0.4.35 + orbax 0.5.5 era stack.
  Modern jax can't read them; Kaggle preinstalls jax 0.11 (pip
  ResolutionImpossible on era pins). Local venv `/tmp/slvenv` WORKS — do
  local CPU eval, not Kaggle, for inference.
- Local CPU speed: 9M ~0.28s/position, 270M ~0.85s/fwd (jit-less).
- Background bash jobs with output pipes can appear hung (low CPU) — run
  sweeps foreground/streaming.


## Baselines (paper comparison data)
- **Gemma-4-E2B baselines — SAFE**: HF dataset `eval-results/caveman-sft-{a1,a2,b1,b2,pretest}/` (5 variants, noexplain samples+summary), restored locally to `results/baselines/`. Note: these are 250-row win-condition slices (examples), NOT full-1000 accuracy.
- **Full clean-1000 (gemma 58.1% + DeepSeek V4 Flash samples) — LOST 2026-08-27** (deleted during cleanup; never git-tracked, not on HF). MUST re-run on the exact noexplain-1000 via `scripts/run_mate_eval.py` (gemma local, DeepSeek API) and store on HF + a non-gitignored location.

## Wave-2 results: completed GAVN arms (2026-09-07; corrected 2026-09-09)

Three GAVN arms reached the full 160k-step target before the Sept-12 quota
exhaustion. Frozen noexplain-1000 MATE (`scripts/eval_gavn.py`, local CPU):

| arm | config | final loss | historical MATE /1000 |
|---|---|---|---|
| gavn-5m-seed0 | 5M, bias both, full loss | 3.893 | 634 = 63.4% via scalar-q |
| gavn-5m-loss | 5M, no Q-loss | 3.883* | **INVALID: 679 = 67.9% via untrained q head** |
| gavn-5m-geometry | 5M, fixed bias | ~3.9 | 589 = 58.9% via scalar-q |

*metrics.json at 160k for the loss arm was read from its 105k sample; losses
all converged to ~3.88-3.92.

### Evaluation correction (2026-09-09)

The no-Q arm's `q_head` received **zero gradient** (`--w-q 0`), but both the
historical MATE and puzzle jobs forced `--score q`. The 67.9% MATE and 4.8%
puzzle figures therefore scored an untrained random head and are invalid. The
same preserved HF checkpoint (`account3-gavn-5m-loss/checkpoint-160000`),
rescored through its trained 128-bin return distribution, gives:

| checkpoint | canonical score | result |
|---|---|---|
| gavn-5m-seed0 (legacy 5.30M params) | distribution expectation | **859/1000 = 85.9%** noexplain MATE |
| gavn-5m-loss (legacy 5.30M params) | distribution expectation | **850/1000 = 85.0%** noexplain MATE |
| gavn-5m-geometry (legacy 5.30M params) | distribution expectation | **858/1000 = 85.8%** noexplain MATE |
| gavn-5m-loss (legacy 5.30M params) | distribution expectation | **401/1000 = 40.1%** exact full-sequence puzzles (same historical slice) |

### Full frozen protocol complete (2026-09-09): all three 5M arms

All three completed 5M arms then ran the complete frozen protocol on local
CPU (commit `c69aed1`): all four MATE sets (4,000 rows) plus the official
10,000-puzzle full-solution-sequence protocol, canonical distribution score.
Full logs, commands, and checkpoint configs:
`results/frozen-evals-2026-09-09/` (local) and HF
`eval-results/gavn-5m-full-frozen-2026-09-09/`.

| arm | MATE 4,000 rows | puzzles 10,000 |
|---|---|---|
| gavn-5m-seed0 | **3,414/4,000 = 85.35%** | **4,383/10,000 = 43.83%** |
| gavn-5m-loss (no Q-loss) | 3,410/4,000 = 85.25% | 4,159/10,000 = 41.59% |
| gavn-5m-geometry (fixed bias) | 3,389/4,000 = 84.72% | **4,388/10,000 = 43.88%** |

References (same protocol where measured): Ruoss 9M = 3,949/4,000 = 98.725%
MATE, 8,613/10,000 = 86.13% puzzles; 136M/270M = 99.4% MATE; paper puzzles:
9M 88.9%, 136M 94.5%, 270M 95.4%.

Reading: the three arms are a statistical tie on both measurements (≤0.6pp
MATE, ≤2.3pp puzzles; MATE and puzzle orderings disagree), so neither the
Q-loss ablation nor the geometry ablation separates at this scale. Against
the 9M teacher the gap is ~13pp MATE and ~42pp puzzles — a decisive negative
for legacy-GAVN at 5M params on this compute. The corrected numbers replace
both the invalid scalar-q figures (63.4/67.9/58.9%) and the first-1,000-puzzle
40.1% slice.

`scripts/eval_gavn.py` now defaults to the distribution score, rejects
untrained scalar heads, uses the official legal-move order, and applies the
official repetition rule. The evaluation notebook no longer hard-codes q mode.

### CC-GAVN follow-up (ready, not yet measured)

`scripts/train_ccgavn.py` implements Candidate-Conditioned GAVN v1: a
4,762,088-parameter model in which the candidate move is a token in every
geometric attention layer, with exact horizontal FEN/action reflection, a
position-disjoint development split, and full HF-resume checkpoints. It must
beat the corrected GAVN by development selection and then pass the full frozen
protocol before any frontier claim is made.

## Two-model endgame + fixed-slim geometry arm (2026-09-09, user decision)

Focus narrows to exactly two models: CC-GAVN (superseded text: it became the
active direction and is now under corrected-v2; this line previously said
"parked, do not modify", which no longer reflects the run history) and the
fixed-bias geometry arm trimmed to its true trained size. The ablation matrix
justified this: the no-Q and dynamic-bias ingredients changed nothing, and the
geometry arm's `bias_mode="fixed"` forward never reads the dynamic projection
— those ~1.84M parameters sit at random init with zero gradient.

- `scripts/train_gavn.py` gains `--bias-mode fixed-slim` (physically omits the
  dynamic module; plain `fixed` still allocates it so legacy checkpoints load
  strictly) and `--relation-schema {v2,legacy-v1}` for fresh runs (resume
  auto-detects as before). Config checkpoints record both.
- **Verified equivalence** on the trained geometry checkpoint
  (`account3-gavn-5m-geometry/checkpoint-160000`): slim = 3,461,377 vs
  full = 5,304,577 params; dropping exactly the 16 dead dynamic tensors and
  loading strictly, the slim forward is **bitwise identical** (max |Δlogit| =
  0.0 over 200 MATE positions × all legal moves, 6,447 pairs). The slim
  trainer also passed an end-to-end CPU training smoke with real HF upload
  and evals through `scripts/eval_gavn.py` unchanged.
- `notebooks/08_kaggle_train_gavn_slim.ipynb` trains the slim arm with the
  identical recipe (dim 224, 8 layers, legacy-v1 relations, w-q 0.5, seed 0,
  160k steps, HF-resumable) as RUN_ID `account3-gavn-5m-geometry-slim`;
  registered in `scripts/launch_trainers.py`
  (`--only gavn-5m-geometry-slim`). Its smoke gate asserts the exact slim
  parameter count before spending GPU.
- **Replication gate:** the slim arm must first match the 5.30M arm (84.72%
  MATE / 43.88% puzzles) on the frozen protocol before any improvement
  variant (e.g. corrected v2 relation schema, which fixes the two dead
  relation categories) is trained. All GPU accounts are quota-blocked until
  2026-09-12T00:00Z.

RETIRED (2026-09-09, user decision): the old arms and runs are no longer
needed. The stranded checkpoints (gavn-3m-seed0 @ 105k, gavn-3m-seed1 @ 110k,
baseline @ 55k/120k) will NOT be resumed; the completed arms' results are
final. `launch_trainers.py` / `launch_evals.py` now register only the slim
geometry arm. Old HF checkpoints are kept untouched for provenance (they
underpin the reported results and the slim-equivalence verification); the
stale local watchers (monitor_overnight, watch_runs, hf_poll) were stopped —
its trainer auto-push was already permanently disarmed by
`logs/trainer_push_state.json` (`pushed: true`). One monitor instance is
supervised by the user's omp daemon and may respawn; it only polls and
cannot relaunch anything.

## Wave-1 training status (2026-09-03)

All 6 sharded trainers (GAVN-3M seed0/seed1, GAVN-5M, geometry ablation,
loss ablation, 5M JAX baseline) reached HF `checkpoint-5000` on 80.27M-row
ChessBench, then stopped on the 2026-09-03 GPU quota exhaustion (all 3
accounts 0.0h; reset 2026-09-05T00:00Z). Auto-relaunch armed via
`scripts/quota_relaunch.py` (resumes from step 5000).

Checkpoint-5000 eval probes (`scripts/eval_gavn.py`, local CPU):
- noexplain-1000, GAVN-3M seed0: **538/1000 = 53.8%**
- tactic 200-row probe, GAVN-3M seed0: 104/200 = 52.0%
- noexplain 100-row probes: GAVN-5M 63%, loss-ablation 60%,
  geometry-ablation 59%, GAVN-3M seed0 58%, GAVN-3M seed1 55%.
  All arms above chance at ~3% training; train losses 4.49-4.55 clustered.
- Step 5000/160000 (~3%), train loss 5.64 -> ~4.4. At chance as expected;
  the number that matters is the same eval at checkpoint-50000+.

## Remaining work
1. Repair the controlled 5M student: fix the double-log-softmax bug, use the
   real training distribution rather than a test-bag derivative, and add full
   HF-resumable state.
2. Implement the 3–6M square-token Geometric Action-Value Network (GAVN),
   with chess relation bias, source/destination action factorization, and
   distribution + scalar-Q + ranking objectives.
3. Run a frozen-protocol ablation matrix across the three Kaggle accounts.
4. Select the Pareto frontier by held-out MATE accuracy, 10K puzzle accuracy,
   parameters, FLOPs, latency, calibration, and error overlap.
5. Convert the result into a reproducible A* workshop paper with uncertainty
   estimates and negative-result documentation.
