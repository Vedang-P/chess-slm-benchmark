#!/bin/bash
cd /home/vedang/Desktop/Research/neuro-symbolic-pathfinding
source /home/vedang/Desktop/Research/neuro-symbolic-pathfinding/venv/bin/activate
export PYTHONUNBUFFERED=1
exec python3 -u run_benchmarks.py --verbose
