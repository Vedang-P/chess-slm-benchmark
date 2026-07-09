# Makefile for Neuro-Symbolic Pathfinding Project

.PHONY: setup data baselines eval clean all

# Setup virtual environment and install dependencies
setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

# Download benchmark datasets
data:
	mkdir -p data/mazes
	wget -P data/mazes/ https://github.com/YuhanJiang415/lost-in-aggregation/releases/download/v0.1/mazes_s3.json
	wget -P data/mazes/ https://github.com/YuhanJiang415/lost-in-aggregation/releases/download/v0.1/mazes_s5.json
	wget -P data/mazes/ https://github.com/YuhanJiang415/lost-in-aggregation/releases/download/v0.1/mazes_s7.json
	wget -P data/mazes/ https://github.com/YuhanJiang415/lost-in-aggregation/releases/download/v0.1/mazes_s10.json
	wget -P data/mazes/ https://github.com/YuhanJiang415/lost-in-aggregation/releases/download/v0.1/mazes_s15.json
	python -m src.grid_generator

# Test Gemma 4 setup
test-model:
	python -m src.gemma4_env

# Generate grid data
generate:
	python -m src.grid_generator

# Run baselines
baselines:
	python -m src.baselines

# Run full evaluation
eval:
	python -m src.gridroute_runner

# Run all: generate data, baselines, evaluation
all: data baselines eval

# Clean generated files
clean:
	rm -rf data/results/*
	rm -rf data/gridroute/*.csv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
