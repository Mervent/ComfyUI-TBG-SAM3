.PHONY: venv lint format test

venv:
	uv venv .venv --python 3.10
	uv pip install -r requirements.txt

lint:
	uv run --no-project ruff check lib/ tests/ nodes.py

format:
	uv run --no-project ruff format lib/ tests/ nodes.py

test:
	uv run --no-project pytest tests/ -v
