"""状态内事件模型(NodeEvent 家族)与构造函数;失败字段单源在 schemas.sections.failure_event_fields。"""

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from deepresearcher.errors import clip_text
from deepresearcher.observability.tracing.context import SpanContext, current_span_context


def _link_or(link: SpanContext | None) -> SpanContext:
    """显式 link 优先；缺省自动读取当前上下文。"""
    return link if link is not None else current_span_context()


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
) -> dict[str, object]:
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
    event: dict[str, object] = {
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
        error=clip_text(error),
        payload=payload or {},
    )
