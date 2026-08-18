.PHONY: setup test clean

setup:
	python3 -m venv venv && venv/bin/pip install -r requirements.txt

test:
	python3 scripts/test_engine.py

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
