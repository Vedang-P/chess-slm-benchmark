"""Single frozen contract for the puzzle continuation and its supervisor."""

HF_REPO = "vedangfake/chess-slm-benchmark"
SOURCE_PREFIX = "puzzle-corpus-v1"
DATA_PREFIX = "puzzle-action-values-v1"
RUN = "ccgavn-5m-puzzles-v1-seed0"
BASE_RUN = "ccgavn-5m-seed0-v2"
BASE_STEP = 3_896_658
ADDITIONAL_STEPS = 2_000_000
FINAL_STEP = BASE_STEP + ADDITIONAL_STEPS
MILESTONE_EVERY = 50_000
ROOTS_PER_SHARD = 5_000
INCLUDED_ROOTS = 6_147_346
SHARD_COUNT = (INCLUDED_ROOTS + ROOTS_PER_SHARD - 1) // ROOTS_PER_SHARD
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
STOCKFISH16_COMMIT = "68e1e9b3811e16cad014b590d7443b9063b52"


def milestones() -> list[int]:
    return [BASE_STEP + step for step in range(MILESTONE_EVERY,
                                               ADDITIONAL_STEPS + 1,
                                               MILESTONE_EVERY)]
