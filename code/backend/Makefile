.PHONY: install dev test test-cov lint format check clean run migrate

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

dev:
	pip install -r requirements.txt
	pip install ruff mypy

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=app --cov-report=html --cov-report=term-missing

# ── Code Quality ──────────────────────────────────────────────────────────────
lint:
	ruff check app/ tests/

format:
	ruff format app/ tests/

check: lint
	python -m mypy app/ --ignore-missing-imports

# ── Database ──────────────────────────────────────────────────────────────────
migrate:
	alembic upgrade head

migrate-create:
	alembic revision --autogenerate -m "$(msg)"

# ── Run ───────────────────────────────────────────────────────────────────────
run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

run-prod:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage
