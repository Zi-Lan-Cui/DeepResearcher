"""Static dependency guards for the service/control-plane boundary."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ENGINE_ROOTS = (
    Path("src/deepresearcher/agents"),
    Path("src/deepresearcher/nodes"),
    Path("src/deepresearcher/tools"),
)
# 图装配与模型出入口是根级单文件；编排解散后它们接替 orchestration/ 受同一条纪律。
ENGINE_FILES = (
    Path("src/deepresearcher/llm.py"),
    Path("src/deepresearcher/graph.py"),
    Path("src/deepresearcher/execution_boundary.py"),
)
SERVICE_PREFIX = "deepresearcher.service"
EXECUTION_PREFIX = "deepresearcher.service.execution"


def test_agent_engine_does_not_import_service_delivery_layer():
    violations: list[str] = []
    for root in (*ENGINE_ROOTS, *ENGINE_FILES):
        for path in [root] if root.is_file() else root.rglob("*.py"):
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


def test_run_control_plane_does_not_import_execution_plane():
    violations: list[str] = []
    for path in Path("src/deepresearcher/service/runs").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == EXECUTION_PREFIX or module.startswith(f"{EXECUTION_PREFIX}."):
                    violations.append(f"{path}:{node.lineno} imports {module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == EXECUTION_PREFIX or alias.name.startswith(
                        f"{EXECUTION_PREFIX}."
                    ):
                        violations.append(f"{path}:{node.lineno} imports {alias.name}")

    assert violations == [], "run control plane must not import execution modules:\n" + "\n".join(
        violations
    )


def test_execution_runtime_can_be_imported_before_run_manager():
    """Canonical package modules must remain safe in either import order."""

    code = (
        "from deepresearcher.service.execution.runtime import worker_lifespan; "
        "from deepresearcher.service.runs.manager import RunManager; "
        "assert worker_lifespan and RunManager"
    )
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603
