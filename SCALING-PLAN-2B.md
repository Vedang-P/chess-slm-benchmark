# 2B-Pair Scaling Plan (kickoff 2026-09-19)

## 2026-10-06 — Correction: the launched 2B-A segment is a legacy 61-tag run; corrected-v2 replaces it

The 2026-10-05 launch below was executed with two defects that were only found
by the 2026-10-06 review:

- the schedule was built for the full step budget and then indexed by the
  *global* step, so the first 1,620,000 slots of the 102-tag schedule were
  skipped: **41 of the 102 declared tags were never sampled**. The 1.62M→
  ~1.715M checkpoints on HF therefore trained on the 61-tag subset that
  happened to fall after the skip point, without MATE/puzzle exclusions, with
  the LR schedule applied to optimizer group 0 only, and with the
  castling-reflection / dev-RNG issues listed in `PROJECT-STATUS.md`;
- the "~2.67 passes over the union" arithmetic is not what the executed
  schedule did.

Corrected-v2 (training disarmed until explicitly armed):

- new HF prefix `ccgavn-5m-seed0-v2`; warm start from the latest complete
  legacy checkpoint (weights + optimizer state, disclosed as a
  `historical_training_caveat` in every checkpoint);
- stage-relative schedule over **all 102 tags** (`--stage-start-step`), so no
  block is skipped; from stage start `s0` the stage length is
  `3,896,658 − s0` steps and the schedule digest is pinned in checkpoint
  configs;
- MATE (4,000 rows / 2,952 unique positions) and all official puzzle decision
  positions excluded from training; exclusions also remove the development
  fold globally;
- one list of eval milestones in `scripts/ensure_eval_preview.py` is rendered
  into the pushed eval kernel (no watcher/kernel drift);
- `scripts/watch_2b.py` refuses to push any training kernel unless armed via
  `configs/ccgavn-2b-ARMED` or `CCGAVN2B_ARMED=1`.

## 2026-10-05 — Stage 2B-A: user decisions, exact arithmetic, launch

This plan was written for a single ~2B / 2.67-epoch stage. On 2026-10-05 the
user chose a two-segment execution that starts training immediately:

- **Segment 2B-A (running):** resume `ccgavn-5m-seed0` from
  checkpoint-1,620,000 on the **102 built tags** (frozen 61-tag 1B corpus +
  41 remaining built shards). Union = 94,277,038 original + 1,652,013,445 new
  = **1,746,290,483 rows**. New cosine schedule to step **3,896,658**
  (= 1,620,000 + round(2.67 × 1,746,290,483 / 2048) = +2,276,658 steps), i.e.
  ~2.67 passes over the added 722,968,579 rows because the schedule is
  proportional over the union. Warm start (weights + optimizer state); the
  LR continues on the new cosine (~3.1e-4 at resume, no re-warmup).
- **Segment 2B-B (pending):** the last 16 planned shards (~268.7M rows) are
  being labeled now; append them with a further warm-start continuation once
  they land.
- Recipe unchanged: distillation signal, buckets, loss, model (4.76M params),
  batch 2048, and the contiguous P000/P001 blocks all stay as-is. The
  proposal to "fix" the puzzle-block dev spikes by rescheduling was rejected
  by the user on 2026-10-05 (transient interference, fully recovered;
  rescheduling would change the recipe that produced the puzzle gains).
- Engineering: `train_ccgavn.py --allow-shard-superset` (stage transition,
  checkpoint tags must be a subset), `kernels/ccgavn-2b/train_2b.py`,
  `scripts/watch_2b.py`, CI tick updated, monitor extended to 3,896,658.

**First shard-content insights (2026-10-05; sampled teacher_logp.npy via HTTP
Range, no full downloads):**

- The 92 built ChessBench shards are homogeneous: teacher entropy 2.59–2.62
  nats (top-prob ~0.30), no shard with a distinct content signature; per-block
  dev-loss effects are within ±0.02 and uncorrelated with entropy/rows
  (Spearman 0.17 / −0.12). Adding the remaining ChessBench shards is therefore
  a *data-quantity* intervention, not a *data-diversity* one — expect smooth
  diminishing returns; there is no "golden shard" to chase.
- The only content class that moves the model is the puzzle curriculum
  (P000/P001, 7.0M rows each): softer teacher (2.88 nats) and the only blocks
  that move dev (+0.17/+0.22 excursion, fully recovered). Every large
  MATE/puzzle jump in the log is downstream of puzzle blocks. After 2B, the
  higher-leverage axes are curriculum (more/harder tactics) or capacity — not
  more of the same ChessBench distribution.
- Crude scaling extrapolation: unique ChessBench data 80M (at 320k) → 1.023B
  (at 1.62M) bought +2.65pp MATE / +14.36pp puzzles; a log-linear fit over
  that interval predicts roughly **+0.7pp MATE / +3.8pp puzzles** for the next
  2x of unique data. The flat 1B tail makes this an upper bound, so 2B-A is a
  direct test of whether the data axis still pays.

## 2026-09-20 audit: 1B continuation gate bug, frozen corpus, fleet stop

**Why training never started (fixed):** `kaggle kernels push` uploads *only*
the `code_file` (verified in the Kaggle CLI source; no other files are sent).
`kernels/ccgavn-1b/train_1b.py` read its shard-row map from a sibling
`shard_rows.json` that `scripts/watch_1b.py` wrote into the temp push folder —
so inside the kernel the file never existed, the map silently fell back to
`{}`, the gate counted **0M / 920M rows**, waited 900s and exited with
"prerequisites not ready within the wait window". The CI watcher saw no active
kernel ~10-20 min later and re-pushed (v17 06:22Z, v18 06:42Z, v19 07:02Z, …),
burning ~0.25 GPU-h per cycle on `vedangpandeyyy` (15.12h → 13.86h between
05:22Z and 07:02Z) while `checkpoint-320000` (landed 2026-09-19T18:51Z) sat
ready. The monitor's `train1b = 0%` was correct: no checkpoint beyond 320k
existed because the continuation never ran.

**Fix (engineering only, no recipe change):**
- `train_1b.py` now carries a placeholder for the frozen shard-tag list; the
  watcher injects the JSON at push time (same pattern the `build-2b` kernels
  already use). If the list is missing the kernel exits immediately instead of
  silently waiting 15 minutes.
- The gate is now exact: every frozen tag must be on HF plus
  `checkpoint-320000` (`config.json` + `state.pt`).
- `scripts/watch_1b.py` gained `--check-only` (never pushes; used locally).
- `scripts/train_ccgavn.py` gained `--shard-tags-file`; `ShardManager` gained
  a `tags` filter; checkpoints record the frozen `shard_tags` and a resume
  with a different set is refused. This also removes the previous silent
  hazard where a mid-run resume re-snapshotted the (then still growing) HF
  shard listing and changed the schedule order.

**Fleet stop:** the 920M stop rule only prevented *re-pushes*; the four
in-flight `build-2b-slice` sessions kept labeling until their sessions ended
(last upload 07:58Z), overshooting to **92/108 planned shards = 1.652B new
rows**. The extra ~730M rows stay on HF untouched for the deferred 2B stage;
`ensure_build_2b.py` will not re-push (target already exceeded).

**Frozen 1B-first corpus (user decision 2026-09-20):**
`configs/ccgavn-1b-shard-tags.json` freezes **61 tags** = the 10 original tags
(8 ChessBench + 2 puzzle, already used by the 320k run) + **51 new planned
shards, taken in plan slice order, built-only, accumulating to 929.0M new
rows** (≥ 920M target). The 2B stage remains deferred.

**Epoch arithmetic to keep in view:** the continuation runs steps 320,000 →
1,620,000 of a fresh schedule over the frozen corpus (1.023B rows total =
929.0M new + 94.28M original). The schedule allocates steps ∝ rows, so the
post-320k phase (1.3M steps × 2048 = 2.662B samples) covers ~80% of the
corpus, ~821M rows → **≈ 3.2 passes** in that phase (≈ 3.6 over the new rows
alone). Flagged, not changed: the step count and the data cap are the user's
scientific decisions.

Goal (revised 2026-09-19, user decision): run the CC-GAVN recipe on **~1B
unique** ChessBench action-value pairs first (Google's recipe proportion:
~2.67 epochs over the unique data), keeping the distillation recipe **exactly**
as-is (9M teacher distributions + raw Stockfish bucket soft targets, 128 bins).
The 2B stage is deferred until the 1B results justify it. The labeling fleet
stops once ~920M new rows (~1B total with the existing 80.27M) are uploaded;
the CI watcher enforces this stop rule.

## Rules (user decision, 2026-09-19)

1. **The fundamental recipe never changes** to save compute or time. In
   particular the 9M-teacher distillation signal is not dropped, the loss is
   unchanged, and no test-time tricks are added to the training recipe.
2. Optimization is limited to *engineering* (throughput, parallelism,
   streaming, storage), never to the scientific method.
3. Everything is documented here honestly as it happens.

## Data selection

- Source: ChessBench action-value train shards (`action_value-*-of-02148`).
- Sizes scanned for all 2,148 shards (2026-09-18); full ranking saved locally
  as `shard_sizes.json` (and reproducible with a HEAD scan; ~27 s).
- Selected the **largest 108 unbuilt shards** (indices excluding 00000-00007,
  which are already labeled): ~1.921B rows available. For the 1B-first stage
  the fleet stops when ~920M new rows are labeled (about half the slices);
  the remainder stays available for the deferred 2B stage.
- Split into 8 slices of 13-14 shards (~230-252M rows each); see
  `shard_slices_2b.json` (saved at kickoff).

## Labeling (Phase A)

- Pipeline: existing `scripts/build_full_dataset.py` (download raw bag ->
  parse -> 9M teacher label -> upload to HF under `chessbench-full-build/`),
  now with `--shard-list` for explicit shard indices.
- Parallelism: 4 accounts x 2 labeler processes (one per T4), each process
  owning one slice. Kaggle kernels re-pushed by the `watch-build-2b` GitHub
  workflow until every shard of the slice exists on HF.
- Measured rates: raw download ~1.3-1.9GB/shard; parse ~5-15 min CPU;
  teacher ~4,700 rows/s -> ~70 min per ~20M-row shard; upload ~6.5GB/shard.
  Estimate ~1.5-2h wall per shard, ~20-28h per slice.

## Trainer engineering (Phase B, required before 2B training)

- `shard_data.py` currently downloads **all** shards up front; at 2B rows that
  is ~670GB and impossible in a Kaggle session.
- Change: download a shard when the schedule first needs it, delete the
  previous shard afterwards (schedule blocks are contiguous). Target overhead
  <= ~1 min per ~13k-step block.
- This is a pure data-plumbing change: the model, loss, data content, and
  training recipe stay identical.

## Training (Phase C)

- **Continuation, not a fresh run (user decision 2026-09-19):** resume
  `ccgavn-5m-seed0` from `checkpoint-320000` and train a further ~1.3M steps
  (to ~1.62M total) with the ~60-shard corpus (8 original + 2 puzzle + ~54 new)
  in the schedule. The post-320k samples (~1.3M x 2048 = 2.66B) work out to
  ~2.6-2.7 epochs over the new ~920M rows, matching the Google proportion.
  ~180 GPU-h at ~2 steps/s on T4x2.
- Launch gate: the training kernel waits for BOTH `checkpoint-320000` on HF and
  the 1B corpus to be complete, then resumes. It never starts on a partial
  corpus (the shard list is snapshotted at process start).
- Evaluate after the continuation; only then decide on the deferred 2B stage.
- Stop rule: development loss plateau plus the final protocol at the end.
  (Historical note: milestone preview evals *did* repeatedly score the frozen
  MATE/puzzle sets, so they are not an untouched holdout; that exposure is now
  disclosed, and selection uses development loss only.)

## Evaluation (Phase D)

- One-shot final protocol per completed stage: 4 x 1,000 MATE + the official
  10K puzzles, `--score auto`, archived on HF under `eval-results/`. Milestone
  evals are monitoring/dev diagnostics and are labeled as such; they are not a
  holdout, and final numbers must state that these sets were scored during
  development. Per-example JSONL is persisted so analyses cluster by position
  (4,000 MATE rows cover 2,952 unique positions).

## Artifacts and where things live

- Labeled shards: HF `vedangfake/chess-slm-benchmark` ->
  `chessbench-full-build/shard-XXXXX/{train_set.npz,teacher_logp.npy}`.
- Training runs: HF `ccgavn-5m-seed0` (current) and the future 2B run prefix.
- Evaluations: HF `eval-results/...`; Stockfish ladder: `elo-results/...`.
- Shard size ranking / slice assignment: `shard_sizes.json`,
  `shard_slices_2b.json` (workspace copies), regenerable by HEAD scan.

## Timeline (kickoff)

- Phase A starts 2026-09-19 (~2 days wall, ~120-150 GPU-h of labeling).
- Phase B while A runs.
- Phase C starts when enough shards exist; ~2.5-3 weeks to 2B at 2.67 epochs.
- Phase D automatic (polling eval kernel + CI safety net).
