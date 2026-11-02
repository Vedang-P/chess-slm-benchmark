"""Frozen contract for the cont1 continuation and its sequential-pool supervisor."""

HF_REPO = "vedangfake/chess-slm-benchmark"
BASE_RUN = "ccgavn-5m-v2-functional-seed0"
BASE_STEP = 3_896_658
MODEL_PARAMETERS = 3_895_352
SEARCHLESS_COMMIT = "90ae0e6b121673fc3079aaeffa047580bb600c0a"

ACCOUNTS = ("vedanggggg", "vedangpandeyyy", "softmaxsimp", "samaltmannnn", "shoumikmitra")
CREDENTIAL_DATASETS = {
    "vedanggggg": "vedanggggg/chess-creds",
    "vedangpandeyyy": "vedangpandeyyy/chess-creds",
    "softmaxsimp": "softmaxsimp/chess-creds",
    "samaltmannnn": "samaltmannnn/hf-creds",
    "shoumikmitra": "shoumikmitra/hf-creds",
}

# One learner session per Kaggle account in turn. Kaggle kills a session at 12 h
# of wall clock; the trainer measures this budget from the kernel's own start, so
# setup time counts and the final checkpoint is on the Hub by ~11.5 h.
SESSION_SECONDS = 41_400
MIN_SESSION_SECONDS = 3_600
QUOTA_RESERVE_H = 0.4
LEARNER_SLUG = "ccgavn-cont-learner"
STATUS_PREFIX = "cont-pipeline/status"

# The conservative recipe: low constant LR with EMA, a 70/30 broad/tactical mix
# and a listwise loss over each tactical root. The sprint arms vary one thing at
# a time around this; the main run uses the arm that validates best.
COMMON_ARGS = [
    "--lr", "2e-5", "--warmup", "1000", "--ema-decay", "0.9998",
    "--weight-decay", "0.01", "--reflect-prob", "0.5", "--block-steps", "6000",
    "--tau", "0.05", "--w-ce-tac", "1.0", "--seed", "0",
]
MIX_30 = ["--broad-rows", "1434", "--tactical-groups", "20"]
MIX_50 = ["--broad-rows", "1024", "--tactical-groups", "34"]

ARMS = {
    "a0": {"note": "control: broad replay only",
           "args": ["--broad-rows", "2048", "--tactical-groups", "0", "--lambda-list", "0"]},
    "a1": {"note": "30% tactical roots, pointwise losses only",
           "args": MIX_30 + ["--lambda-list", "0"]},
    "a2": {"note": "30% tactical roots + listwise loss",
           "args": MIX_30 + ["--lambda-list", "1.0"]},
    "a3": {"note": "50% tactical roots + listwise loss",
           "args": MIX_50 + ["--lambda-list", "1.0"]},
    "a4": {"note": "a2 at 2.5x the learning rate",
           "args": MIX_30 + ["--lambda-list", "1.0", "--lr", "5e-5"]},
}
ARM_STEPS = 30_000
ARM_RUN_PREFIX = "ccgavn-cont1-"
