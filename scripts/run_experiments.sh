#!/bin/bash
set -euo pipefail

# Main experiment runner - executes all stages

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
VENV="$REPO_DIR/venv"
source "$VENV/bin/activate"

# Config
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="neuro-symbolic-pathfinding"
export HF_HOME="$REPO_DIR/data/hf_cache"
export WANDB_DIR="$REPO_DIR/data/wandb"

mkdir -p "$REPO_DIR/data/results" "$REPO_DIR/data/gridroute" "$REPO_DIR/data/hf_cache" "$REPO_DIR/data/wandb"

STAGE=${1:-all}
METHODS=${2:-"pure_slm,aop_astar,neuro_symbolic"}
NL_VARIANT=${3:-"direct"}

echo "=========================================="
echo "Neuro-Symbolic Pathfinding Experiments"
echo "Stage: $STAGE"
echo "Methods: $METHODS"
echo "NL Variant: $NL_VARIANT"
echo "=========================================="
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo "VRAM: $(python -c 'import torch; print(f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB" if torch.cuda.is_available() else "N/A")')"
echo "=========================================="

run_gridroute() {
    echo ""
    echo "=== GridRoute Benchmark ==="
    python -m src.gridroute_runner
}

run_lost_in_agg() {
    echo ""
    echo "=== Lost in Aggregation Benchmark ==="
    python -m src.lost_in_agg_runner
}

run_ablation_function_calling() {
    echo ""
    echo "=== Ablation: No Function Calling ==="
    python -c "
from src.grid_generator import generate_all_gridroute
from src.gemma4_env import Gemma4Env
from src.neuro_symbolic_pipeline import neuro_symbolic_plan
from src.evaluation import compute_metrics, print_report
import numpy as np

env = Gemma4Env(load_in_4bit=True, enable_thinking=True, use_function_calling=False)
env.load()
tasks = generate_all_gridroute(seed=42)
results = []
for t in tasks:
    r = neuro_symbolic_plan(env, t.grid, t.nl_variants['direct'])
    r.optimal_path = t.optimal_path
    r.optimal_length = t.optimal_length
    results.append(r)
report = compute_metrics(results, tasks[0].grid)
print_report(report, 'No FC GridRoute')
"
}

echo ""
echo "Starting experiments at $(date)"

case "$STAGE" in
    gridroute)
        run_gridroute
        ;;
    lost_in_agg)
        run_lost_in_agg
        ;;
    ablation_fc)
        run_ablation_function_calling
        ;;
    all)
        run_gridroute
        run_lost_in_agg
        ;;
    *)
        echo "Usage: $0 {all|gridroute|lost_in_agg|ablation_fc} [methods] [nl_variant]"
        exit 1
        ;;
esac

echo ""
echo "Experiments complete at $(date)"
