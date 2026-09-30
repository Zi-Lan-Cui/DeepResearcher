"""OTel TracerProvider 的服务侧装配:全局 provider 一次性注册,langfuse 优先接管。

必须在构造 RunExecutor(即首次 ``Langfuse()``)之前调用 :func:`configure_tracer_provider`:
langfuse SDK 检测到已存在的全局 SDK provider 会直接接管并挂上它自己的导出 processor,
两端共享同一棵树;反过来先建 langfuse 再接管则只能复用其 provider,配置面不一致。
观测面装配失败不得影响启动,整体 fail-open。
"""

from __future__ import annotations

import sys

from opentelemetry import trace as opentelemetry_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

from deepresearcher.observability.logging_config import get_logger
from deepresearcher.observability.tracing import ledger
from deepresearcher.service.settings import ServiceConfig

logger = get_logger("deepresearcher.service.execution.telemetry")


def configure_tracer_provider(config: ServiceConfig) -> None:
    """按 ``otel_exporter`` 装配全局 TracerProvider;非法值告警后按 none 处理。"""
    try:
        provider = opentelemetry_trace.get_tracer_provider()
        if not isinstance(provider, TracerProvider):
            created = TracerProvider()
            opentelemetry_trace.set_tracer_provider(created)
            # set_tracer_provider 是一次性的;竞态下以先装上的那份为准。
            current = opentelemetry_trace.get_tracer_provider()
            provider = current if isinstance(current, TracerProvider) else created
        # 账本处理器常驻:span 是唯一的执行记录引擎,run 是否落账由注册表决定。
        ledger.attach_to_provider(provider)
        if config.otel_exporter == "console":
            # 显式传 out:SDK 默认参数在模块导入时就绑定了 sys.stdout,
            # 测试内的流重定向与装配时刻的终端都要用当下的 stdout。
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter(out=sys.stdout)))
        elif config.otel_exporter != "none":
            logger.warning(
                "SERVICE_OTEL_EXPORTER=%r 不可识别(可用 none|console),按 none 处理。",
                config.otel_exporter,
            )
    except Exception:
        logger.warning("otel_provider_setup_failed", exc_info=True)


def shutdown_tracer_provider() -> None:
    """关闭全局 provider(冲刷批处理队列);未装配或已关闭时静默。"""
    try:
        provider = opentelemetry_trace.get_tracer_provider()
        if isinstance(provider, TracerProvider):
            provider.shutdown()
    except Exception:
        logger.warning("otel_provider_shutdown_failed", exc_info=True)
