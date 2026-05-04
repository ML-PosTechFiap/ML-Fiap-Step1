.PHONY: install lint test run train up down clean

install:
	uv pip install torch --index-url https://download.pytorch.org/whl/cpu
	uv pip install -r pyproject.toml --no-sources

lint:
	ruff check src tests
	ruff format --check src tests

lint-fix:
	ruff check --fix src tests
	ruff format src tests

test:
	PYTHONPATH=src pytest tests/ -v

test-cov:
	PYTHONPATH=src pytest tests/ -v --tb=short

run:
	PYTHONPATH=src uvicorn api.main:app --reload --port 8000

train:
	PYTHONPATH=src python src/main.py

up:
	docker compose up --build

down:
	docker compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
