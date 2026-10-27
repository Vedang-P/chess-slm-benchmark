"""Single frozen contract for the puzzle continuation and its supervisor."""

HF_REPO = "vedangfake/chess-slm-benchmark"
SOURCE_PREFIX = "puzzle-corpus-v1"
DATA_PREFIX = "puzzle-action-values-v1"
# Freeze the teacher-complete shards available when the learner is first
# launched. This lets training begin while CPU labeling and teacher generation
# continue; the resulting checkpoint is explicitly an interim cohort result.
RUN = "ccgavn-3.9m-puzzles-teacher-ready-20261028-seed0"
COHORT_MANIFEST = f"{DATA_PREFIX}/cohorts/teacher-ready-20261028-v1.json"
COHORT_ID = "teacher-ready-20261028-v1"
FUNCTIONAL_RUN = "ccgavn-5m-v2-functional-seed0"
DEMO_RUN = "ccgavn-puzzle-demo-3.9m-v1"
BASE_RUN = "ccgavn-5m-seed0-v2"
BASE_STEP = 3_896_658
ADDITIONAL_STEPS = 2_000_000
FINAL_STEP = BASE_STEP + ADDITIONAL_STEPS
MODEL_VERSION = "v2"
MODEL_PARAMETERS = 3_895_352
REMOVED_SOFTMAX_NULL_PARAMETERS = 869_440
MILESTONE_EVERY = 50_000
ROOTS_PER_SHARD = 5_000
INCLUDED_ROOTS = 6_147_346
SHARD_COUNT = (INCLUDED_ROOTS + ROOTS_PER_SHARD - 1) // ROOTS_PER_SHARD
CPU_BATCH_SESSIONS_PER_ACCOUNT = 5
ACCOUNTS = ("vedanggggg", "vedangpandeyyy", "softmaxsimp",
            "samaltmannnn", "shoumikmitra")
CREDENTIAL_DATASETS = {
    "vedanggggg": "vedanggggg/chess-creds",
    "vedangpandeyyy": "vedangpandeyyy/chess-creds",
    "softmaxsimp": "softmaxsimp/chess-creds",
    "samaltmannnn": "samaltmannnn/hf-creds",
    "shoumikmitra": "shoumikmitra/hf-creds",
}
SOURCE_SHA256 = "76335bfa7d7c4a7f93c1366d81549e53951ebb79dd43d34904cab8f22d962f8d"
SEARCHLESS_COMMIT = "90ae0e6b121673fc3079aaeffa047580bb600c0a"
STOCKFISH16_COMMIT = "68e1e9b3811e16cad014b590d7443b9063b3eb52"


def milestones() -> list[int]:
    return [BASE_STEP + step for step in range(MILESTONE_EVERY,
                                               ADDITIONAL_STEPS + 1,
                                               MILESTONE_EVERY)]
