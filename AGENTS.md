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

- The shared implementation lives in `scripts/kaggle_checkpoint.py`; the
  baseline trainers in `scripts/train_ccgavn.py` and `scripts/train_gavn.py`
  show how to use it. It uploads complete checkpoint directories to the
  `--hf-repo` dataset on an interval, uploads final artifacts, and resumes
  from the latest complete remote checkpoint.
- Token: `HF_WRITE_TOKEN` (env or `.env`). Default repo:
  `vedangfake/chess-slm-benchmark`.
- Every new trainer must meet this contract.
- Failure status is also uploaded to HF (`run-status.txt` in the same repo)
  so a killed kernel's error is readable without downloading the multi-GB
  working dir.
