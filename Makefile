.PHONY: install test lint smoke

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

smoke:
	python scripts/smoke_train.py --steps 10
