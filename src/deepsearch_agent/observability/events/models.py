from datetime import datetime, timezone
from typing import Literal, TypedDict
from uuid import uuid4

from pydantic import BaseModel, Field

from deepsearch_agent.observability.tracing.context import current_context


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
    trace_id: str | None = None,
    span_id: str | None = None,
    run_id: str | None = None,
    session_id: str | None = None,
    node_id: str | None = None,
    component: str | None = None,
    payload: dict | None = None,
) -> Event:
    """创建领域审计事件；调用方可为可复现性记录受控的草稿或最终产物。"""
    context = current_context()
    trace_id = trace_id or (context.trace_id if context else None)
    span_id = span_id or (context.span_id if context else None)
    run_id = run_id or (context.run_id if context else None)
    session_id = session_id or (context.session_id if context else None)
    node_id = node_id or (context.node_id if context else None)
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
        "evidence_extract": "source_reader",
    }.get(tool)
    event = make_node_event(
        tool,
        status,
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
    return event.model_copy(
        update={"event_type": event_name or "tool_" + status, "node": tool}
    )


def make_node_event(
    node: str,
    status: Literal["started", "completed", "failed", "skipped", "cancelled"],
    *,
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
    context = current_context()
    trace_id = trace_id or (context.trace_id if context else None)
    span_id = span_id or (context.span_id if context else None)
    run_id = run_id or (context.run_id if context else None)
    session_id = session_id or (context.session_id if context else None)
    node_id = node_id or (context.node_id if context else node)
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
        error=error[:500] if error else "",
        payload=payload or {},
    )


def make_artifact_event(
    artifact_type: Literal["prompt", "output"],
    content: str,
    *,
    name: str,
    trace_id: str | None = None,
    span_id: str | None = None,
    run_id: str | None = None,
    session_id: str | None = None,
    node_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """创建与运行事件分离的完整内容记录。"""
    return {
        "record_type": "artifact",
        "artifact_id": f"artifact-{uuid4().hex}",
        "artifact_type": artifact_type,
        "name": name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id,
        "span_id": span_id,
        "run_id": run_id,
        "session_id": session_id,
        "node_id": node_id,
        "content_chars": len(content),
        "content": content,
        "metadata": metadata or {},
    }
