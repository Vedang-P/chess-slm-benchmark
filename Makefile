.PHONY: setup test check notebooks clean

setup:
	python3 -m venv venv && venv/bin/pip install -r requirements.txt

test:
	python3 scripts/test_engine.py

check: test
	python3 scripts/run_chess.py --model deepseek-v4-flash --task mate1-lichess \
		--prompt-variant fen --n 1 --max_new_tokens 2048 --conditions win --smoke

notebooks:
	python3 notebooks/build_mate1000_variants_notebook.py

clean:
	rm -rf results results_check* __pycache__ src/__pycache__
