# 2B-Pair Scaling Plan (kickoff 2026-09-19)

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

- 2.67 epochs over ~1.0B rows = ~2.67B samples = ~1.3M steps at batch 2048
  (~180 GPU-h at the measured ~2 steps/s on T4x2). Evaluate; only then decide
  whether to extend to the 2B stage (~2.6M steps total).
- Stop rule: development loss plateau plus the frozen protocol at the end.
  No probing against frozen sets.

## Evaluation (Phase D)

- Frozen protocol only, once per completed stage: 4 x 1,000 MATE + the
  official 10K puzzles, `--score auto`, on Kaggle CPU, archived on HF under
  `eval-results/`.

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
