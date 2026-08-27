import asyncio
from contextlib import contextmanager
from time import perf_counter
from typing import Iterator

from deepsearch_agent.observability.events.sink import JsonlSink
from deepsearch_agent.observability.tracing.context import (
    TraceContext,
    current_context,
    new_id,
    reset_context,
    set_context,
)


class TraceRecorder:
    """记录一棵 Trace 树；Span 通过 ContextVar 自动关联父子关系。"""

    def __init__(self, sink: JsonlSink):
        self.sink = sink

    @contextmanager
    def trace(
        self, name: str = "research", *, run_id: str | None = None, session_id: str | None = None
    ) -> Iterator[str]:
        trace_id = new_id("trace")
        started = perf_counter()
        parent = current_context()
        effective_run_id = run_id or (parent.run_id if parent else None)
        effective_session_id = session_id or (parent.session_id if parent else None)
        token = set_context(
            TraceContext(
                trace_id=trace_id,
                run_id=effective_run_id,
                session_id=effective_session_id,
                node_id=parent.node_id if parent else None,
            )
        )
        self.sink.write(
            {
                "record_type": "trace",
                "event_type": "trace_started",
                "trace_id": trace_id,
                "run_id": effective_run_id,
                "session_id": effective_session_id,
                "node_id": parent.node_id if parent else None,
                "name": name,
            }
        )
        error = None
        try:
            yield trace_id
            status = "completed"
        except BaseException as exc:
            status = (
                "cancelled"
                if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt))
                else "failed"
            )
            error = str(exc)[:500]
            raise
        finally:
            record = {
                "record_type": "trace",
                "event_type": f"trace_{status}",
                "trace_id": trace_id,
                "run_id": effective_run_id,
                "session_id": effective_session_id,
                "node_id": parent.node_id if parent else None,
                "duration_ms": round((perf_counter() - started) * 1000, 2),
            }
            if error:
                record["error"] = error
            self.sink.write(record)
            reset_context(token)

    @contextmanager
    def span(self, name: str, kind: str = "node"):
        parent = current_context()
        if parent is None:
            with self.trace() as trace_id:
                with self._span(trace_id, None, name, kind) as span_id:
                    yield span_id
            return
        with self._span(parent.trace_id, parent.span_id, name, kind) as span_id:
            yield span_id

    @contextmanager
    def _span(self, trace_id: str, parent_span_id: str | None, name: str, kind: str):
        span_id = new_id("span")
        started = perf_counter()
        parent = current_context()
        token = set_context(
            TraceContext(
                trace_id,
                span_id,
                run_id=parent.run_id if parent else None,
                session_id=parent.session_id if parent else None,
                node_id=parent.node_id if parent else None,
            )
        )
        self.sink.write(
            {
                "record_type": "span",
                "event_type": "span_started",
                "trace_id": trace_id,
                "span_id": span_id,
                "parent_span_id": parent_span_id,
                "run_id": parent.run_id if parent else None,
                "session_id": parent.session_id if parent else None,
                "node_id": parent.node_id if parent else None,
                "name": name,
                "kind": kind,
            }
        )
        error = None
        try:
            yield span_id
            status = "completed"
        except BaseException as exc:
            status = (
                "cancelled"
                if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt))
                else "failed"
            )
            error = str(exc)[:500]
            raise
        finally:
            record = {
                "record_type": "span",
                "event_type": f"span_{status}",
                "trace_id": trace_id,
                "span_id": span_id,
                "run_id": parent.run_id if parent else None,
                "session_id": parent.session_id if parent else None,
                "node_id": parent.node_id if parent else None,
                "duration_ms": round((perf_counter() - started) * 1000, 2),
            }
            if error:
                record["error"] = error
            self.sink.write(record)
            reset_context(token)
