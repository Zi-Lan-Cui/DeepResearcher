"""业务关联标签的 ContextVar 载体：new_id 生成、current_span_context 冻结快照。"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from uuid import uuid4

from opentelemetry import trace as opentelemetry_trace


@dataclass(frozen=True)
class TraceContext:
    """绑定在 contextvars 上的业务标签快照。"""

    run_id: str | None = None
    session_id: str | None = None
    node_id: str | None = None


@dataclass(frozen=True)
class SpanContext:
    """从活跃上下文冻结出的可携带关联身份。"""

    trace_id: str | None = None
    span_id: str | None = None
    run_id: str | None = None
    session_id: str | None = None
    node_id: str | None = None


_context: ContextVar[TraceContext | None] = ContextVar("deepresearcher_trace_context", default=None)


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def current_context() -> TraceContext | None:
    return _context.get()


def current_span_context() -> SpanContext:
    """冻结当前关联身份：trace/span 两个 id 取自 OTel 当前 span，业务标签取自 ContextVar。"""
    span_context = opentelemetry_trace.get_current_span().get_span_context()
    labels = _context.get()
    return SpanContext(
        trace_id=format(span_context.trace_id, "032x") if span_context.is_valid else None,
        span_id=format(span_context.span_id, "016x") if span_context.is_valid else None,
        run_id=labels.run_id if labels else None,
        session_id=labels.session_id if labels else None,
        node_id=labels.node_id if labels else None,
    )


def set_context(context: TraceContext | None):
    return _context.set(context)


def reset_context(token) -> None:
    _context.reset(token)


@contextmanager
def bind_context(
    *, run_id: str | None = None, session_id: str | None = None, node_id: str | None = None
):
    """在当前异步上下文绑定/合并业务标签；未传入的字段沿用已有值。"""
    current = current_context()
    context = TraceContext(
        run_id=run_id if run_id is not None else (current.run_id if current else None),
        session_id=session_id
        if session_id is not None
        else (current.session_id if current else None),
        node_id=node_id if node_id is not None else (current.node_id if current else None),
    )
    token = set_context(context)
    try:
        yield context
    finally:
        reset_context(token)
