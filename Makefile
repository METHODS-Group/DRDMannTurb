PYTEST = uv run pytest
PYTEST_OPTS = -v --tb=short

.PHONY: test test-unit test-integration test-slow test-cov lint format typecheck check clean help

# Default: fast unit tests
test: test-unit

test-unit:
	$(PYTEST) $(PYTEST_OPTS) test/unit_tests

test-integration:
	$(PYTEST) $(PYTEST_OPTS) test/integration_tests

test-slow:
	$(PYTEST) $(PYTEST_OPTS) --runslow test

test-cov:
	$(PYTEST) $(PYTEST_OPTS) --cov=drdmannturb --cov-report=term-missing --cov-report=html test

lint:
	uv run ruff check drdmannturb test examples
	uv run ruff format --check drdmannturb test examples

format:
	uv run ruff check --fix drdmannturb test examples
	uv run ruff format drdmannturb test examples

typecheck:
	uv run ty check drdmannturb

# Everything CI runs, locally
check: lint typecheck test-unit

clean:
	rm -rf .pytest_cache/ htmlcov/ .coverage dist/ build/ *.egg-info/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

help:
	@echo "Targets:"
	@echo "  test             - Run unit tests (default)"
	@echo "  test-unit        - Run unit tests"
	@echo "  test-integration - Run integration tests"
	@echo "  test-slow        - Run full suite including slow tests"
	@echo "  test-cov         - Run full suite with coverage"
	@echo "  lint             - ruff check + format check"
	@echo "  format           - ruff autofix + format"
	@echo "  typecheck        - ty check"
	@echo "  check            - lint + typecheck + unit tests"
	@echo "  clean            - Remove build/test artifacts"
