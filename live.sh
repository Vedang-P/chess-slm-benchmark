#!/bin/bash
# Live progress monitor — run with: bash live.sh
LOG="/home/vedang/Desktop/Research/neuro-symbolic-pathfinding/benchmark_run.log"
RESULTS="/home/vedang/Desktop/Research/neuro-symbolic-pathfinding/data/results/gemma4-e2b_q3_k_s"

while true; do
  clear
  echo "⏱  $(date '+%H:%M:%S')"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━"
  
  # Latest progress
  tail -1 "$LOG" 2>/dev/null || echo "No log"
  
  # Count tasks done per config
  echo ""
  for f in "$RESULTS"/gridroute_*.json; do
    [ -f "$f" ] && echo "✅ DONE: $(basename "$f" .json)" || true
  done
  
  # Estimate remaining
  last=$(grep "Progress:" "$LOG" 2>/dev/null | tail -1 | grep -oP '\d+(?=/\d+)')
  total=$(grep "Progress:" "$LOG" 2>/dev/null | tail -1 | grep -oP '\d+$' || echo "?")
  if [ -n "$last" ] && [ "$last" -gt 0 ]; then
    pct=$(( last * 100 / total ))
    remaining=$(( (total - last) / 10 ))  # ~10 tasks/min
    echo "  $last/$total ($pct%) ~${remaining}min left"
  fi
  
  # GPU
  echo ""
  nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null || echo "GPU: N/A"
  
  sleep 15
done