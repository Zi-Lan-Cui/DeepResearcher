from datetime import datetime, timezone
from typing import Literal, TypedDict
from uuid import uuid4

from pydantic import BaseModel, Field

from deepresearcher.observability.tracing.context import SpanContext, current_span_context


def _link_or(link: SpanContext | None) -> SpanContext:
    """显式 link 优先；缺省自动读取当前上下文。"""
    return link if link is not None else current_span_context()


class Event(TypedDict, total=False):
    record_type: str
    event_id: str
    event_type: str
    timestamp: str
    trace_id: str
    span_id: str
    run_id: str
    session_id: str
    node_id: str
    node: str
    component: str
    status: Literal["started", "completed", "failed", "skipped", "cancelled"]
    duration_ms: float
    error: str
    payload: dict


class NodeEvent(BaseModel):
    """节点生命周期事件的状态内模型。"""

    record_type: Literal["event"] = "event"
    event_id: str
    event_type: str
    timestamp: str
    trace_id: str | None = None
    span_id: str | None = None
    run_id: str | None = None
    session_id: str | None = None
    node_id: str | None = None
    node: str
    component: str | None = None
    status: Literal["started", "completed", "failed", "skipped", "cancelled"]
    duration_ms: float | None = None
    error: str = ""
    payload: dict[str, object] = Field(default_factory=dict)


def make_audit_event(
    event_type: str,
    *,
    link: SpanContext | None = None,
    trace_id: str | None = None,
    span_id: str | None = None,
    run_id: str | None = None,
    session_id: str | None = None,
    node_id: str | None = None,
    node_id_fallback: str | None = None,
    component: str | None = None,
    payload: dict | None = None,
) -> Event:
    """创建领域审计事件；关联字段按 显式参数 > link > 当前上下文 解析。

    ``node_id_fallback`` 只在完全没有上下文时生效（如单元直接调用），
    使调用方无需自己展开关联身份。
    """
    resolved = _link_or(link)
    trace_id = trace_id or resolved.trace_id
    span_id = span_id or resolved.span_id
    run_id = run_id or resolved.run_id
    session_id = session_id or resolved.session_id
    node_id = node_id or resolved.node_id or node_id_fallback
    event: Event = {
        "record_type": "event",
        "event_id": f"evt-{uuid4().hex}",
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if trace_id:
        event["trace_id"] = trace_id
    if span_id:
        event["span_id"] = span_id
    if run_id:
        event["run_id"] = run_id
    if session_id:
        event["session_id"] = session_id
    if node_id:
        event["node_id"] = node_id
    if component:
        event["component"] = component
    if payload:
        event["payload"] = payload
    return event


def make_tool_event(
    tool: str,
    status: Literal["started", "completed", "failed", "skipped", "cancelled"],
    *,
    link: SpanContext | None = None,
    trace_id: str | None = None,
    span_id: str | None = None,
    run_id: str | None = None,
    session_id: str | None = None,
    node_id: str | None = None,
    component: str | None = None,
    duration_ms: float | None = None,
    error: str | None = None,
    payload: dict | None = None,
    event_name: str | None = None,
) -> NodeEvent:
    component = component or {
        "search": "search_tool",
        "fetch": "source_reader",
    }.get(tool)
    event = make_node_event(
        tool,
        status,
        link=link,
        trace_id=trace_id,
        span_id=span_id,
        run_id=run_id,
        session_id=session_id,
        node_id=node_id,
        duration_ms=duration_ms,
        error=error,
        payload=payload,
        component=component,
    )
    return event.model_copy(update={"event_type": event_name or "tool_" + status, "node": tool})


def make_node_event(
    node: str,
    status: Literal["started", "completed", "failed", "skipped", "cancelled"],
    *,
    link: SpanContext | None = None,
    trace_id: str | None = None,
    span_id: str | None = None,
    run_id: str | None = None,
    session_id: str | None = None,
    node_id: str | None = None,
    component: str | None = None,
    duration_ms: float | None = None,
    error: str | None = None,
    payload: dict | None = None,
) -> NodeEvent:
    resolved = _link_or(link)
    trace_id = trace_id or resolved.trace_id
    span_id = span_id or resolved.span_id
    run_id = run_id or resolved.run_id
    session_id = session_id or resolved.session_id
    node_id = node_id or resolved.node_id
    return NodeEvent(
        event_id=f"evt-{uuid4().hex}",
        event_type="node_" + status,
        timestamp=datetime.now(timezone.utc).isoformat(),
        trace_id=trace_id,
        span_id=span_id,
        run_id=run_id,
        session_id=session_id,
        node_id=node_id or node,
        node=node,
        component=component,
        status=status,
        duration_ms=round(duration_ms, 2) if duration_ms is not None else None,
        error=_clip(error, _EVENT_ERROR_MAX_CHARS),
        payload=payload or {},
    )


_EVENT_ERROR_MAX_CHARS = 500


def _clip(text: str | None, limit: int) -> str:
    """事件里的错误文本带截断标记,与 RunError 钳位同一口径留痕。"""
    if not text:
        return ""
    return text if len(text) <= limit else text[: limit - 8] + "…[截断]"


def make_artifact_event(
    artifact_type: Literal["prompt", "output"],
    content: str,
    *,
    name: str,
    link: SpanContext | None = None,
    trace_id: str | None = None,
    span_id: str | None = None,
    run_id: str | None = None,
    session_id: str | None = None,
    node_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """创建与运行事件分离的完整内容记录；关联字段按 显式参数 > link > 当前上下文 解析。"""
    resolved = _link_or(link)
    return {
        "record_type": "artifact",
        "artifact_id": f"artifact-{uuid4().hex}",
        "artifact_type": artifact_type,
        "name": name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id or resolved.trace_id,
        "span_id": span_id or resolved.span_id,
        "run_id": run_id or resolved.run_id,
        "session_id": session_id or resolved.session_id,
        "node_id": node_id or resolved.node_id,
        "content_chars": len(content),
        "content": content,
        "metadata": metadata or {},
    }
