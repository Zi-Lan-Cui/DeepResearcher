from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from uuid import uuid4


@dataclass(frozen=True)
class TraceContext:
    trace_id: str
    span_id: str | None = None
    run_id: str | None = None
    session_id: str | None = None
    node_id: str | None = None


_context: ContextVar[TraceContext | None] = ContextVar("deepsearch_trace_context", default=None)


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def current_context() -> TraceContext | None:
    return _context.get()


def set_context(context: TraceContext | None):
    return _context.set(context)


def reset_context(token) -> None:
    _context.reset(token)


@contextmanager
def bind_context(
    *, run_id: str | None = None, session_id: str | None = None, node_id: str | None = None
):
    """在当前异步上下文绑定业务关联字段，不改变 trace/span 身份。"""
    current = current_context()
    if current is None:
        context = TraceContext(
            "trace-unbound", run_id=run_id, session_id=session_id, node_id=node_id
        )
    else:
        context = TraceContext(
            trace_id=current.trace_id,
            span_id=current.span_id,
            run_id=run_id if run_id is not None else current.run_id,
            session_id=session_id if session_id is not None else current.session_id,
            node_id=node_id if node_id is not None else current.node_id,
        )
    token = set_context(context)
    try:
        yield context
    finally:
        reset_context(token)
