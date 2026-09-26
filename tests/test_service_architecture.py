"""内核/service 边界的静态依赖守卫。"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

SRC_ROOT = Path("src")
KERNEL_ROOT = SRC_ROOT / "deepresearcher"
SERVICE_PREFIX = "deepresearcher.service"
EXECUTION_PREFIX = "deepresearcher.service.execution"

# worker.py 是执行平面进程入口（python -m deepresearcher.worker），属于外壳；
# 它放在包根只是为了模块路径短，不改变归属。
SHELL_ENTRYPOINTS = frozenset({Path("src/deepresearcher/worker.py")})

# 棘轮白名单：service 现存的每一条跨界 import（模块名与属性名都登记，按前缀
# 匹配）。新增跨界 import 必须显式加名单并说明它为何是内核对外端口；
# 方向是收窄（graph 与 protocol 类端口），已有边不会自行消失。
ALLOWED_KERNEL_IMPORTS = frozenset(
    {
        # 配置与模型出入口
        "deepresearcher.config",  # Settings / get_settings / LLMConfig / project_path_env
        "deepresearcher.graph",  # build_graph
        "deepresearcher.llm",  # classify_llm_error
        # 观测与追踪
        "deepresearcher.observability",  # JsonlSink（包口导出）
        "deepresearcher.observability.logging_config",  # get_logger
        "deepresearcher.observability.tracing",  # TraceRecorder
        "deepresearcher.observability.tracing.context",  # current_span_context / new_id
        "deepresearcher.observability.usage_runtime",  # bind/current/enforce/record/reset + 异常与类型
        # 契约与词汇
        "deepresearcher.routing",  # NodeName
        "deepresearcher.schemas",  # StopReason 等
        "deepresearcher.schemas.limits",  # EVENT_CONTENT_PREVIEW_CHARS
        "deepresearcher.vocab",  # RETRIEVAL_* 常量
        # 资料与传输端口（tools 侧 protocol/store 抽象，外壳注入实现）
        "deepresearcher.tools.transport",  # HttpClient
        "deepresearcher.tools.web.documents",  # DocumentRef / DocumentReadRange / 大纲与 Grep 结果
        "deepresearcher.tools.web.materials",  # ResearchMaterialStore / Memory 实现 / StoredDocument 等
        "deepresearcher.tools.web.materials.models",  # build_stored_document
        "deepresearcher.tools.web.materials.store",  # grep_lines / read_lines
    }
)


def _containing_package_parts(path: Path) -> list[str]:
    parts = list(path.relative_to(SRC_ROOT).with_suffix("").parts)
    parts.pop()  # 模块文件去掉自身名；__init__ 去掉后恰好得到所在包
    return parts


def _resolved_imports(path: Path) -> list[tuple[int, str]]:
    """返回 (行号, 解析后的点分名)；相对导入按所在包还原，`from pkg import x`
    同时产出 pkg 与 pkg.x（静态无法区分子模块与属性，一律从严）。"""
    pkg = _containing_package_parts(path)
    resolved: list[tuple[int, str]] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                if node.level - 1 > len(pkg):
                    continue  # 越过包根，非法 import，交由解释器报错
                parts = pkg[: len(pkg) - (node.level - 1)] + (node.module or "").split(".")
                base = ".".join(p for p in parts if p)
            else:
                base = node.module or ""
            candidates = [base, *(f"{base}.{a.name}" for a in node.names)]
            resolved.extend((node.lineno, candidate) for candidate in candidates)
        elif isinstance(node, ast.Import):
            resolved.extend((node.lineno, alias.name) for alias in node.names)
    return resolved


def _under(prefix: str, resolved: str) -> bool:
    return resolved == prefix or resolved.startswith(f"{prefix}.")


def _walk(root: Path):
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def test_kernel_does_not_import_service_layer():
    """内核（deepresearcher 除 service/ 与外壳入口外的一切）不得依赖投递层。

    evidence/reporting/observability 等新内核模块无需登记即自动受同一纪律约束。
    """
    violations: list[str] = []
    for path in _walk(KERNEL_ROOT):
        if _under("deepresearcher.service", ".".join(path.relative_to(SRC_ROOT).parts)) or (
            path in SHELL_ENTRYPOINTS
        ):
            continue
        for lineno, resolved in _resolved_imports(path):
            if _under(SERVICE_PREFIX, resolved):
                violations.append(f"{path}:{lineno} imports {resolved}")

    assert violations == [], "kernel must not depend on service delivery modules:\n" + "\n".join(
        violations
    )


def test_service_only_touches_kernel_through_declared_ports():
    violations: list[str] = []
    for path in _walk(KERNEL_ROOT / "service"):
        for lineno, resolved in _resolved_imports(path):
            if not resolved.startswith("deepresearcher.") or _under(SERVICE_PREFIX, resolved):
                continue
            # 前缀语义：属性名（config.Settings）跟随其模块登记，棘轮按被依赖的模块收紧。
            if not any(_under(m, resolved) for m in ALLOWED_KERNEL_IMPORTS):
                violations.append(f"{path}:{lineno} imports {resolved}")

    assert violations == [], (
        "service imports kernel modules outside the declared port allowlist:\n"
        + "\n".join(violations)
    )


def test_run_control_plane_does_not_import_execution_plane():
    violations: list[str] = []
    for path in _walk(KERNEL_ROOT / "service" / "runs"):
        for lineno, resolved in _resolved_imports(path):
            if _under(EXECUTION_PREFIX, resolved):
                violations.append(f"{path}:{lineno} imports {resolved}")

    assert violations == [], "run control plane must not import execution modules:\n" + "\n".join(
        violations
    )


def test_event_and_preview_planes_do_not_import_each_other():
    """持久事件面与可丢弃预览面互不依赖;跨面接线只发生在执行器与 SSE 路由。"""
    preview_prefix = "deepresearcher.service.preview"
    events_prefix = "deepresearcher.service.events"
    violations: list[str] = []
    for path in _walk(KERNEL_ROOT / "service" / "events"):
        for lineno, resolved in _resolved_imports(path):
            if _under(preview_prefix, resolved):
                violations.append(f"{path}:{lineno} imports {resolved}")
    for path in _walk(KERNEL_ROOT / "service" / "preview"):
        for lineno, resolved in _resolved_imports(path):
            if _under(events_prefix, resolved):
                violations.append(f"{path}:{lineno} imports {resolved}")

    assert violations == [], "event and preview planes must stay independent:\n" + "\n".join(
        violations
    )


def test_store_append_has_single_caller():
    """生产内唯一合法的 RunEventStore.append 调用点是 RunEventHub.flush。

    绕过 hub 直写账本 = 消费端收不到唤醒、seq 编排与 _requeue 契约全部失效。
    """
    allowed = Path("src/deepresearcher/service/events/hub.py")
    violations: list[str] = []
    for path in _walk(KERNEL_ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "append"
            ):
                base = node.func.value
                receiver = getattr(base, "attr", None) or getattr(base, "id", None)
                if receiver in {"store", "event_store"} and path != allowed:
                    violations.append(f"{path}:{node.lineno} calls {receiver}.append")

    assert violations == [], "only RunEventHub.flush may append to the event store:\n" + "\n".join(
        violations
    )


def test_web_control_plane_does_not_import_execution_plane():
    """API 进程结构上无法执行:web→execution 的任何 import 都等于重新启用 embedded。"""
    violations: list[str] = []
    for path in _walk(KERNEL_ROOT / "service" / "web"):
        for lineno, resolved in _resolved_imports(path):
            if _under(EXECUTION_PREFIX, resolved):
                violations.append(f"{path}:{lineno} imports {resolved}")

    assert violations == [], "web control plane must not import execution modules:\n" + "\n".join(
        violations
    )


def test_no_production_module_imports_local_preview_bus():
    """LocalPreviewBus 只由单栈 harness(测试)构造。

    生产预览路径只经 EphemeralEventBus 协议装配(Redis 总线,或退化为 None);
    若把本地总线 import 进 src,等于恢复 embedded 模式的进程内直投。
    """
    preview_prefix = "deepresearcher.service.preview.local"
    violations: list[str] = []
    for path in _walk(KERNEL_ROOT):
        for lineno, resolved in _resolved_imports(path):
            if _under(preview_prefix, resolved):
                violations.append(f"{path}:{lineno} imports {resolved}")

    assert violations == [], "production code must not import LocalPreviewBus:\n" + "\n".join(
        violations
    )


def test_execution_runtime_can_be_imported_before_run_manager():
    """包的正规模块必须任意导入顺序都安全。"""

    code = (
        "from deepresearcher.service.execution.runtime import worker_lifespan; "
        "from deepresearcher.service.runs.manager import RunManager; "
        "assert worker_lifespan and RunManager"
    )
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603
