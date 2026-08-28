#!/usr/bin/env bash
# Run the searchless_chess MATE sweep: 3 models x 4 subsets.
# The 4 subsets are the EXACT MATE test sets gemma and deepseek were
# scored on (strategy, noexplain, tactic, both). Protocol: official
# ActionValueEngine semantics (see scripts/sl_mate_eval.py), the same
# protocol that produced the Mac's 9M=98.2% noexplain-1000 result.
# Checkpoint layout (official zips): <root>/9M/6400000/params,
# <root>/136M/6400000/params, <root>/270M/6400000/params (ocdbt DB root is
# the inner "params" dir; "params_ema" is the EMA variant) — override per
# model with CKPT_9M / CKPT_136M / CKPT_270M.
# Outputs: results/sl-mate-<model>.json (per-row + per-set accuracy).
# Usage:
#   MODEL=9M ./scripts/run_searchless_mate_subsets.sh        # one model
#   ./scripts/run_searchless_mate_subsets.sh                 # all 3 models
#   MODEL=9M MAX_ROWS=50 ./scripts/run_searchless_mate_subsets.sh  # smoke
set -euo pipefail

VENV_PY="${VENV_PY:-C:/Users/vedang/AppData/Local/Temp/slvenv/Scripts/python.exe}"
SL_CODE="${SL_CODE:-C:/tmp/sl_code}"
CKPT_ROOT="${CKPT_ROOT:-C:/tmp/sl9m}"
EVAL="${EVAL:-data/positions/mate-selection-test.json,data/positions/mate-selection-test-noexplain.json,data/positions/mate-selection-test-tactic.json,data/positions/mate-selection-test-both.json}"
MAX_ROWS="${MAX_ROWS:-0}"

declare -A CKPTS=( ["9M"]="$CKPT_ROOT/9M/6400000/params" ["136M"]="$CKPT_ROOT/136M/6400000/params" ["270M"]="$CKPT_ROOT/270M/6400000/params" )
MODELS=("${MODEL:-9M 136M 270M}")

for M in $MODELS; do
  CKPT="${CKPTS[$M]}"
  if [ ! -d "$CKPT" ]; then
    echo "!! checkpoint dir missing for $M: $CKPT" >&2
    continue
  fi
  echo "== $M ($CKPT) =="
  "$VENV_PY" scripts/sl_mate_eval.py \
    --model "$M" --checkpoint "$CKPT" --sl-code "$SL_CODE" \
    --eval "$EVAL" --max-rows "$MAX_ROWS" --save "results/sl-mate-$M.json"
done
