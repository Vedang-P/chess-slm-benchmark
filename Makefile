# Makefile for Neuro-Symbolic Pathfinding Project

.PHONY: setup data generate clean all

# Setup virtual environment and install dependencies
setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

# Download benchmark datasets
data:
	./scripts/download_mazes.sh
	python -m src.grid_generator

# Generate grid data
generate:
	python -m src.grid_generator

# Run all: generate data
all: data

# Clean generated files
clean:
	rm -rf data/results/*
	rm -rf data/gridroute/*.csv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
