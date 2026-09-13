from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from uuid import uuid4


@dataclass(frozen=True)
class TraceContext:
    """绑定在 contextvars 上的"当前活跃上下文"，trace_id 必填。"""

    trace_id: str
    span_id: str | None = None
    run_id: str | None = None
    session_id: str | None = None
    node_id: str | None = None


@dataclass(frozen=True)
class SpanContext:
    """从活跃上下文冻结出的可携带关联身份（对齐 OTel ``SpanContext`` 语义）。

    用途：在 span 内取出、跨 ``with`` 边界（或跨协程交给 helper）写事件时，
    调用方传一个 ``link=`` 参数即可，不再裸传 trace/span 字符串。
    ``link=None`` 时事件工厂自动读取当前上下文，两条路径共享同一语义。

    ``trace_id``/``span_id`` 是追踪关联；``run_id``/``session_id``/``node_id``
    是业务关联标签（OTel 语义下近似 baggage），与追踪字段同置于一个只读快照。
    """

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
    """把当前上下文冻结成可带出的 SpanContext；无上下文时返回空快照。"""
    context = _context.get()
    if context is None:
        return SpanContext()
    return SpanContext(
        trace_id=context.trace_id,
        span_id=context.span_id,
        run_id=context.run_id,
        session_id=context.session_id,
        node_id=context.node_id,
    )


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
