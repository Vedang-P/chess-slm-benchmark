.PHONY: setup test check notebooks clean

setup:
	python3 -m venv venv && venv/bin/pip install -r requirements.txt

test:
	python3 scripts/test_engine.py

check: test
	python3 scripts/run_suite.py --smoke --models smollm2-1.7b

notebooks:
	python3 notebooks/build_notebook.py && python3 notebooks/build_notebook.py --check

clean:
	rm -rf results results_check* __pycache__ src/__pycache__
