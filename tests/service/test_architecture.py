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

# worker.py 是执行平面进程入口 python -m deepresearcher.service.bootstrap.worker，属外壳；
# 放在包根只是为了模块路径短，不改变归属。
SHELL_ENTRYPOINTS = frozenset({Path("src/deepresearcher/worker.py")})

# 棘轮白名单：service 现存的每一条跨界 import，模块名与属性名都登记，按前缀
# 匹配。新增跨界 import 必须显式加名单并说明它为何是内核对外端口；名单只收
# 不放，收窄方向是 graph 与 protocol 类端口。
ALLOWED_KERNEL_IMPORTS = frozenset(
    {
        # 配置与模型出入口
        "deepresearcher.config",  # Settings / get_settings / LLMConfig / project_path_env
        "deepresearcher.graph",  # build_graph
        "deepresearcher.llm",  # classify_llm_error
        # 观测与追踪
        "deepresearcher.observability",  # JsonlSink（包口导出）
        "deepresearcher.observability.logging_config",  # get_logger
        "deepresearcher.observability.tracing",  # spans / ledger
        "deepresearcher.observability.span",  # current_span_context / new_id
        "deepresearcher.usage_runtime",  # bind/current/enforce/record/reset + 异常与类型
        # 契约与词汇
        "deepresearcher.env",  # env_str/int/float/bool：两套配置的解析语义单源
        "deepresearcher.routing",  # NodeName
        "deepresearcher.schemas",  # StopReason 等
        "deepresearcher.schemas.limits",  # EVENT_CONTENT_PREVIEW_CHARS
        "deepresearcher.pricing",  # LLM 单价表与成本公式（记账与重算共用）
        "deepresearcher.vocab",  # RETRIEVAL_* / INCOMPLETE_REPORT_HEADING 等共享字面量
        # 依赖健康端口：协议在 kernel 声明，持久化实现在 service，消费方在 tools
        "deepresearcher.health",  # HealthLedger / Cooldown / Unavailable
        # 资料与传输端口（tools 侧 protocol/store 抽象，外壳注入实现）
        "deepresearcher.tools.transport",  # HttpClient
        "deepresearcher.tools.web.documents",  # DocumentRef / DocumentReadRange / 大纲与 Grep 结果
        "deepresearcher.tools.web.materials",  # ResearchMaterialStore 等
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
    """内核,即 deepresearcher 除 service/ 与外壳入口外的一切,不得依赖 service 交付层。

    evidence/reporting/observability 等新内核模块无需登记,自动受同一约束。
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


def test_run_lifecycle_does_not_import_execution_plane():
    """runs 定义并改写 Run 的生命周期，不含执行；依赖方向只能是 execution → runs。"""
    violations: list[str] = []
    for path in _walk(KERNEL_ROOT / "service" / "runs"):
        for lineno, resolved in _resolved_imports(path):
            if _under(EXECUTION_PREFIX, resolved):
                violations.append(f"{path}:{lineno} imports {resolved}")

    assert violations == [], "runs must not import execution modules:\n" + "\n".join(violations)


def test_event_and_preview_planes_do_not_import_each_other():
    """持久事件层与可丢弃的预览通道互不依赖；跨层接线只发生在执行器与 SSE 路由。"""
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


def test_asyncpg_raw_connections_live_only_in_signals():
    """PG 连接统一由 sqlalchemy 引擎管理,登记在案的例外只有 signals 的长驻订阅连接。

    再开一处 raw 连接必须先有书面理由,见 signals.py 文件头;否则 NullPool、
    方言、池策略这些集中决定会被各模块陆续绕过。
    """
    allowed = Path("src/deepresearcher/service/coordination/signals.py")
    violations: list[str] = []
    for path in _walk(KERNEL_ROOT):
        for lineno, resolved in _resolved_imports(path):
            if _under("asyncpg", resolved) and path != allowed:
                violations.append(f"{path}:{lineno} imports {resolved}")

    assert violations == [], (
        "asyncpg may only be imported by service.coordination.signals:\n" + "\n".join(violations)
    )


def test_advisory_lock_sql_lives_in_coordination_helper():
    """pg_advisory_* 只许出现在 coordination/locks.py。

    锁的释放时机、是否跨 yield 等语义散写到各平面后,容易出现互不一致的副本。
    """
    allowed = Path("src/deepresearcher/service/coordination/locks.py")
    violations: list[str] = []
    for path in _walk(KERNEL_ROOT):
        if path == allowed:
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if "pg_advisory" in line:
                violations.append(f"{path}:{lineno} contains pg_advisory")

    assert violations == [], "advisory lock SQL must stay inside coordination.locks:\n" + "\n".join(
        violations
    )


def test_store_append_has_single_caller():
    """生产内唯一合法的 RunEventStore.append 调用点是 RunEventBuffer.flush。

    绕过 buffer 直接写事件表,消费端收不到唤醒,seq 编排与 _requeue 契约随之失效。
    """
    allowed = Path("src/deepresearcher/service/events/buffer.py")
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

    assert violations == [], (
        "only RunEventBuffer.flush may append to the event store:\n" + "\n".join(violations)
    )


def test_web_control_plane_does_not_import_execution_plane():
    """API 进程结构上不具备执行能力:web 引用 execution 会重新打开进程内执行的路径。"""
    violations: list[str] = []
    for path in _walk(KERNEL_ROOT / "service" / "web"):
        for lineno, resolved in _resolved_imports(path):
            if _under(EXECUTION_PREFIX, resolved):
                violations.append(f"{path}:{lineno} imports {resolved}")

    assert violations == [], "web control plane must not import execution modules:\n" + "\n".join(
        violations
    )


def test_no_production_module_imports_local_preview_bus():
    """LocalPreviewBus 只由测试 harness 构造。

    生产预览路径只经 EphemeralEventBus 协议装配,取 Redis 总线或退化为 None;
    把本地总线 import 进 src 会重新打开进程内直投的路径。
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
    """包模块之间的 import 顺序必须任意组合都安全,这里先 execution 后 runs。"""

    code = (
        "from deepresearcher.service.execution.runtime import worker_lifespan; "
        "from deepresearcher.service.runs.manager import RunManager; "
        "assert worker_lifespan and RunManager"
    )
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603


def test_adapters_do_not_import_orchestration_layer():
    """tools/ 是出站适配层：只接收数据类型，不得反向依赖编排状态、图装配与 agent 包。

    Task 曾住在 state.py，fetch/search/reader 为了一个入参类型反向依赖 LangGraph
    顶层状态；迁移到 schemas.tasks 后由本测试守住这条边，避免再长回去。
    """
    forbidden = (
        "deepresearcher.state",
        "deepresearcher.graph",
        "deepresearcher.nodes",
        "deepresearcher.agents",
    )
    violations: list[str] = []
    for path in _walk(KERNEL_ROOT / "tools"):
        for lineno, resolved in _resolved_imports(path):
            if any(_under(prefix, resolved) for prefix in forbidden):
                violations.append(f"{path}:{lineno} imports {resolved}")

    assert violations == [], (
        "outbound adapters must not depend on the orchestration layer:\n" + "\n".join(violations)
    )


def test_shared_middleware_does_not_import_concrete_agents():
    """agents/middleware 是被多个 agent 共用的横切层，不得反向依赖某个 agent 的实现。

    绑定单一 agent 且需读它 context 槽的中间件（方向结果接收）住在其 agent 包内；
    放进共用包会让每个 agent 都随那一个 agent 的业务层一起加载。
    """
    agents = (
        "deepresearcher.agents.supervisor",
        "deepresearcher.agents.researcher",
        "deepresearcher.agents.writer",
        "deepresearcher.agents.clarifier",
    )
    violations: list[str] = []
    for path in _walk(KERNEL_ROOT / "agents" / "middleware"):
        for lineno, resolved in _resolved_imports(path):
            if any(_under(prefix, resolved) for prefix in agents):
                violations.append(f"{path}:{lineno} imports {resolved}")

    assert violations == [], "shared middleware must not depend on a concrete agent:\n" + "\n".join(
        violations
    )


def test_event_layer_does_not_depend_on_tracing():
    """事件层只依赖 observability.span 的业务标签，不得反向依赖 tracing 包。

    tracing.ledger 把 span 写成事件（tracing → events.sink）是合理方向；反过来
    events 为了 SpanContext 去 import tracing 会形成包级循环，两者将无法单独加载。
    """
    violations: list[str] = []
    for path in _walk(KERNEL_ROOT / "observability" / "events"):
        for lineno, resolved in _resolved_imports(path):
            if _under("deepresearcher.observability.tracing", resolved):
                violations.append(f"{path}:{lineno} imports {resolved}")

    assert violations == [], (
        "event layer must not depend on the tracing package (use observability.span):\n"
        + "\n".join(violations)
    )
