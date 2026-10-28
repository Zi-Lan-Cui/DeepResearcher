.PHONY: quality lint typecheck test coverage check verify-service

quality: lint typecheck test coverage

lint:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

typecheck:
	uv run pyright

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q -p pytest_asyncio.plugin

coverage:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run coverage run -m pytest -q -p pytest_asyncio.plugin
	uv run coverage report

check:
	uv run python -m compileall -q src/deepresearcher
	uv run ruff check src tests scripts
	uv run pyright
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q -p pytest_asyncio.plugin

# 启动真实 API/Worker 子进程，但使用无 LLM 费用的可控 Graph。
verify-service:
	uv run python scripts/verify_service_processes.py
