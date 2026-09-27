"""OTel 桥:自研 span 开合的同时创建真实 OpenTelemetry span 并置为当前上下文。

kernel 只依赖 opentelemetry-api;TracerProvider 由 service 装配根注册,
尚未配置 provider 时整体静默降级(NonRecordingSpan,不产生任何 otel_* 记录键)。
langfuse.* 等观测后端的属性词汇刻意收拢在本文件,换后端时其余模块不动。
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from opentelemetry import context as opentelemetry_context
from opentelemetry import trace as opentelemetry_trace

# OTel 语义惯例:出站依赖调用(工具触网)用 CLIENT,图内编排节点用 INTERNAL。
_SPAN_KIND_MAP = {
    "tool": opentelemetry_trace.SpanKind.CLIENT,
    "node": opentelemetry_trace.SpanKind.INTERNAL,
    "trace": opentelemetry_trace.SpanKind.INTERNAL,
}
# langfuse 把 observation.type 为 tool 的 span 渲染为工具叶;其余按普通 span。
_LANGFUSE_OBSERVATION_TYPE = {"tool": "tool"}
_INSTRUMENTATION_SCOPE = "deepresearcher.tracing"

_enabled = True


@contextmanager
def override_enabled(enabled: bool) -> Iterator[None]:
    """测试缝:临时关闭桥,验证降级路径与无桥记录逐字节一致。"""
    global _enabled
    previous, _enabled = _enabled, enabled
    try:
        yield
    finally:
        _enabled = previous


class BridgeSpan:
    """一次 begin_span 的句柄:暴露 OTel 双 id,由调用方 finish 定三态。

    id 字段仅在 OTel 给出有效 SpanContext 时为 32/16 位 hex 字符串,否则为
    None——调用方据此决定记录里是否写入 otel_* 键。
    """

    def __init__(self, span: Any, otel_trace_id: str | None, otel_span_id: str | None):
        self._span = span
        self._finished = False
        self.otel_trace_id = otel_trace_id
        self.otel_span_id = otel_span_id

    def finish(self, status: str, error: str | None = None) -> None:
        """按 completed|failed|cancelled 定终态;span.end() 由桥的退出统一执行。"""
        if self._span is None or self._finished:
            return
        self._finished = True
        if status == "failed":
            self._span.set_status(opentelemetry_trace.StatusCode.ERROR, error or "")
            self._span.set_attribute("langfuse.observation.level", "ERROR")
            if error:
                self._span.set_attribute("langfuse.observation.status_message", error)
        elif status == "cancelled":
            # 取消不是错误:不标 ERROR,只留属性供面板过滤。
            self._span.set_attribute("deepresearcher.status", "cancelled")


_NULL_BRIDGE_SPAN = BridgeSpan(None, None, None)


def _sanitized_attributes(attributes: Mapping[str, object]) -> dict[str, str | int | float | bool]:
    result: dict[str, str | int | float | bool] = {}
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            result[key] = value
        else:
            result[key] = str(value)
    return result


@contextmanager
def begin_span(
    name: str,
    *,
    kind: str = "node",
    attributes: Mapping[str, object] | None = None,
) -> Iterator[BridgeSpan]:
    """创建并激活一个 OTel span,体内 current context 即它,子 span 自动认父。

    参数 name 为 span 名;kind 取 trace/node/tool 之一,决定 SpanKind 与
    langfuse 渲染类型;attributes 的非标量值会被 str 化。桥关闭时产出无 id 的
    空句柄。桥自身不制造新异常:体内异常经兜底记账后原样重抛。
    """
    if not _enabled:
        yield _NULL_BRIDGE_SPAN
        return
    tracer = opentelemetry_trace.get_tracer(_INSTRUMENTATION_SCOPE)
    span = tracer.start_span(
        name, kind=_SPAN_KIND_MAP.get(kind, opentelemetry_trace.SpanKind.INTERNAL)
    )
    span.set_attribute("langfuse.observation.type", _LANGFUSE_OBSERVATION_TYPE.get(kind, "span"))
    for key, value in _sanitized_attributes(attributes or {}).items():
        span.set_attribute(key, value)
    token = opentelemetry_context.attach(opentelemetry_trace.set_span_in_context(span))
    span_context = span.get_span_context()
    handle = BridgeSpan(
        span,
        format(span_context.trace_id, "032x") if span_context.is_valid else None,
        format(span_context.span_id, "016x") if span_context.is_valid else None,
    )
    try:
        yield handle
    except BaseException as exc:
        # 调用方未及 finish(异常发生在其记账之前)时的兜底,保证 span 有终态。
        handle.finish("failed", str(exc).strip() or type(exc).__name__)
        raise
    finally:
        opentelemetry_context.detach(token)
        span.end()
