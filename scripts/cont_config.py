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
# Stage 2: both branches start from the EMA weights of arm a4 at step 30,000 and
# differ only in the exact-expectation RL term (the control keeps --rl-weight 0).
STAGE2_INIT = ["--init-hf-run", "ccgavn-cont1-a4", "--init-checkpoint", "checkpoint-30000",
               "--init-weights", "ema"]
A4_RECIPE = MIX_30 + ["--lambda-list", "1.0", "--lr", "5e-5"]
ARMS["r1"] = {"note": "stage 2: exact-expectation RL (beta 0.05) from a4@30k EMA",
              "args": A4_RECIPE + STAGE2_INIT + ["--rl-weight", "1.0", "--rl-beta", "0.05"]}
ARMS["c1"] = {"note": "stage 2 control: same branch point and data, no RL term",
              "args": A4_RECIPE + STAGE2_INIT + ["--rl-weight", "0"]}
# Later-ply experiment: both branches start from the EMA weights of one main-run checkpoint
# (--init-checkpoint is given at launch) with the main-run recipe, and differ only in
# --ply-groups. Length and logging flags (--max-steps 15000 --val-every 2500 --ckpt-every
# 2500 --val-detail) go on the launch command so the launch log records them.
PLY_INIT = ["--init-hf-run", "ccgavn-cont1-a4", "--init-weights", "ema"]
ARMS["p0"] = {"note": "later-ply control: main-run recipe from a main-run checkpoint, no ply roots",
              "args": A4_RECIPE + PLY_INIT + ["--ply-groups", "0"]}
ARMS["p1"] = {"note": "later-ply roots: the control plus 10 ply-3/5 groups per step",
              "args": A4_RECIPE + PLY_INIT + ["--ply-groups", "10"]}
ARM_STEPS = 30_000
ARM_RUN_PREFIX = "ccgavn-cont1-"
