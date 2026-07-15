# Makefile for Neuro-Symbolic Pathfinding Project
#
# GridRoute/MazeBench tasks are generated on the fly by eval.py/train_sft.py/
# train_grpo.py directly (src/grid_generator.py, no static data file) -- the
# actual entry points are those scripts and notebooks/kaggle_train.ipynb, not
# this Makefile. Kept for environment setup only.

.PHONY: setup clean

# Setup virtual environment and install dependencies
setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

# Clean generated files
clean:
	rm -rf results/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
