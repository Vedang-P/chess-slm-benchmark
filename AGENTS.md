# Project Memory

## P0 rule: no unilateral engineering decisions — report, don't decide

When something isn't going as planned (API behavior, yields, check
failures, speed), **stop and report to the user with the evidence instead
of silently working around it**. Never change the experiment's design to
make the tooling behave (e.g. disabling a thinking model, shrinking a
budget, weakening a filter, changing what gets scored). Cheap shortcuts
and reward-hacking are forbidden. If in doubt, surface the tradeoffs and
let the user choose.

## Persistent requirement: Hugging Face checkpoint persistence

**Every training/eval run that could lose progress (Kaggle kernels die at
~12h) MUST upload model weights + training state to Hugging Face
periodically and support resume from HF.**

- The proven pattern lives in `scripts/train_mate_lora.py`:
  `HfCheckpointCallback` uploads the latest checkpoint dir (adapter +
  optimizer + scheduler + trainer_state) to the `--hf-repo` dataset every
  `--hf-upload-every` seconds, uploads the final adapter at train end, and
  `download_hf_checkpoint()` resumes from the latest remote checkpoint.
- Token: `HF_WRITE_TOKEN` (env or `.env`). Default repo:
  `vedangfake/chess-slm-benchmark`.
- Any NEW trainer (e.g. `scripts/train_mate_grpo.py`) must replicate this:
  periodic checkpoint upload during the run, final artifact upload at the
  end, and `--resume-from-hf` support.
- Failure status is also uploaded to HF (`run-status.txt` in the same repo)
  so a killed kernel's error is readable without downloading the multi-GB
  working dir.
