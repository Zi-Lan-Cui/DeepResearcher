.PHONY: quality lint typecheck test coverage check

quality: lint typecheck test coverage

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

typecheck:
	uv run pyright

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q

coverage:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run coverage run -m pytest -q
	uv run coverage report

check:
	uv run python -m compileall -q src/deepsearch_agent
	uv run ruff check src tests
	uv run pyright
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q
