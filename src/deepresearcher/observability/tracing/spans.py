"""span/trace 上下文管理器:OTel span 是唯一的执行记录引擎。

span 身份与父子嵌套由 OpenTelemetry 当前上下文承载;业务标签(run/session/node)
仍走 context.py 的 ContextVar,建 span 时写入 creation-time 属性,供账本处理器
(ledger.py)按 run 路由。langfuse.* 等观测后端的属性词汇刻意收拢在本文件与
ledger 的映射处,换后端时其余模块不动。
"""

import asyncio
import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

from opentelemetry import context as opentelemetry_context
from opentelemetry import trace as opentelemetry_trace

from deepresearcher.errors import clip_text
from deepresearcher.observability.tracing.context import bind_context, current_context

# OTel 语义惯例:出站依赖调用(工具触网)用 CLIENT,图内编排节点用 INTERNAL。
_SPAN_KIND_MAP = {
    "tool": opentelemetry_trace.SpanKind.CLIENT,
    "node": opentelemetry_trace.SpanKind.INTERNAL,
    "trace": opentelemetry_trace.SpanKind.INTERNAL,
}
# langfuse 把 observation.type 为 tool 的 span 渲染为工具叶;其余按普通 span。
_LANGFUSE_OBSERVATION_TYPE = {"tool": "tool"}
INSTRUMENTATION_SCOPE = "deepresearcher.tracing"


def _label_attributes(kind: str) -> dict[str, object]:
    labels = current_context()
    return {
        "deepresearcher.kind": kind,
        "deepresearcher.run_id": labels.run_id if labels else None,
        "deepresearcher.session_id": labels.session_id if labels else None,
        "deepresearcher.node_id": labels.node_id if labels else None,
    }


def _sanitized(attributes: Mapping[str, object]) -> dict[str, str | int | float | bool]:
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
def _recording(
    name: str,
    *,
    kind: str,
    attributes: Mapping[str, object],
    input: object = None,
) -> Iterator[str]:
    """创建并激活 OTel span,体内 current context 即它,子 span 自动认父。

    返回:
        str: 本 span 的 OTel trace id(32 位 hex);无 provider 时为空串。
    异常按三态归类后原样重抛:CancelledError/KeyboardInterrupt 记为 cancelled
    且不标 ERROR,其余记 ERROR 并携带错误文本。
    """
    tracer = opentelemetry_trace.get_tracer(INSTRUMENTATION_SCOPE)
    span = tracer.start_span(
        name,
        kind=_SPAN_KIND_MAP.get(kind, opentelemetry_trace.SpanKind.INTERNAL),
        attributes=_sanitized(attributes),
    )
    span.set_attribute("langfuse.observation.type", _LANGFUSE_OBSERVATION_TYPE.get(kind, "span"))
    if input is not None:
        span.set_attribute(
            "langfuse.observation.input",
            input if isinstance(input, str) else json.dumps(input, ensure_ascii=False, default=str),
        )
    token = opentelemetry_context.attach(opentelemetry_trace.set_span_in_context(span))
    try:
        yield format(span.get_span_context().trace_id, "032x")
    except BaseException as exc:
        if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)):
            # 取消不是错误(不标 ERROR),但原因随属性留档,账本终态记录照旧携带。
            span.set_attribute("deepresearcher.status", "cancelled")
            span.set_attribute("deepresearcher.error", type(exc).__name__)
        else:
            error = clip_text(str(exc).strip() or type(exc).__name__)
            span.set_status(opentelemetry_trace.StatusCode.ERROR, error)
            span.set_attribute("langfuse.observation.level", "ERROR")
            span.set_attribute("langfuse.observation.status_message", error)
        raise
    finally:
        opentelemetry_context.detach(token)
        span.end()


def record_output(value: object) -> None:
    """把输出写到当前 span;面板 Traces 列表的 Output 列取根 observation 的输出。"""
    current = opentelemetry_trace.get_current_span()
    if not current.is_recording():
        return
    current.set_attribute(
        "langfuse.observation.output",
        value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str),
    )


@contextmanager
def span(name: str, *, kind: str = "node", input: object = None) -> Iterator[str]:
    """开一个非 trace 根的 span(node/tool);身份与父子关系由 OTel 上下文决定。"""
    with _recording(name, kind=kind, attributes=_label_attributes(kind), input=input) as trace_id:
        yield trace_id


@contextmanager
def trace(
    name: str = "research",
    *,
    run_id: str | None = None,
    session_id: str | None = None,
    metadata: Mapping[str, object] | None = None,
    input: object = None,
) -> Iterator[str]:
    """开一次 run 的根 span;同时把 run/session 绑定为业务标签向体内传播。

    参数 metadata 双写:deepresearcher.metadata(JSON,供账本还原记录形状)与
    langfuse.observation.metadata.*(供面板展示)。
    """
    attributes = _label_attributes("trace")
    explicit_labels: dict[str, str | None] = {}
    if run_id is not None:
        attributes["deepresearcher.run_id"] = run_id
        explicit_labels["run_id"] = run_id
    if session_id is not None:
        attributes["deepresearcher.session_id"] = session_id
        explicit_labels["session_id"] = session_id
    if metadata:
        attributes["deepresearcher.metadata"] = json.dumps(
            dict(metadata), ensure_ascii=False, default=str
        )
        for key, value in metadata.items():
            attributes[f"langfuse.observation.metadata.{key}"] = value
    with (
        bind_context(**explicit_labels),
        _recording(name, kind="trace", attributes=attributes, input=input) as trace_id,
    ):
        yield trace_id
