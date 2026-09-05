"""Static dependency guards for the service/control-plane boundary."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ENGINE_ROOTS = (
    Path("src/deepsearch_agent/agents"),
    Path("src/deepsearch_agent/llm"),
    Path("src/deepsearch_agent/orchestration"),
    Path("src/deepsearch_agent/tools"),
)
SERVICE_PREFIX = "deepsearch_agent.service"


def test_agent_engine_does_not_import_service_delivery_layer():
    violations: list[str] = []
    for root in ENGINE_ROOTS:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if module == SERVICE_PREFIX or module.startswith(f"{SERVICE_PREFIX}."):
                        violations.append(f"{path}:{node.lineno} imports {module}")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == SERVICE_PREFIX or alias.name.startswith(
                            f"{SERVICE_PREFIX}."
                        ):
                            violations.append(f"{path}:{node.lineno} imports {alias.name}")

    assert violations == [], "engine must not depend on service delivery modules:\n" + "\n".join(
        violations
    )


def test_execution_runtime_can_be_imported_before_run_manager():
    """Package compatibility exports must not create an import-order cycle."""

    code = (
        "from deepsearch_agent.service.execution.runtime import worker_lifespan; "
        "from deepsearch_agent.service.runs import RunManager; "
        "assert worker_lifespan and RunManager"
    )
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603
