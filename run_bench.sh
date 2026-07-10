#!/bin/bash
cd /home/vedang/Desktop/Research/neuro-symbolic-pathfinding
source venv/bin/activate
exec nice -n 19 python3 -u run_benchmarks.py
