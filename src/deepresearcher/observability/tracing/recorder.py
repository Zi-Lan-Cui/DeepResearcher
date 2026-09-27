"""TraceRecorder:span 父子嵌套靠 ContextVar 传递;span() 可独立成 trace,trace() 是显式包装。

每次 trace()/span() 同时经 telemetry_bridge 开一个真实 OTel span 并置为当前上下文:
langfuse 等观测后端的子 observation 因此挂在自研节点结构之下。记录内的自研 id
保持不变,OTel 双 id(otel_trace_id/otel_span_id)仅在桥产出有效 id 时增补。
"""

import asyncio
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from time import perf_counter
from typing import Any, Protocol

from deepresearcher.observability.tracing import telemetry_bridge
from deepresearcher.observability.tracing.context import (
    TraceContext,
    current_context,
    new_id,
    reset_context,
    set_context,
)


class TraceSink(Protocol):
    def write(self, record: Mapping[str, Any] | Any) -> None: ...


def _bridge_attributes(
    *,
    trace_id: str,
    span_id: str | None = None,
    parent_span_id: str | None = None,
    run_id: str | None,
    session_id: str | None,
    node_id: str | None,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """OTel 侧反查属性:面板行凭 deepresearcher.* 回到 run_events 账本。"""
    attributes: dict[str, object] = {
        "deepresearcher.trace_id": trace_id,
        "deepresearcher.span_id": span_id,
        "deepresearcher.parent_span_id": parent_span_id,
        "deepresearcher.run_id": run_id,
        "deepresearcher.session_id": session_id,
        "deepresearcher.node_id": node_id,
    }
    for key, value in (metadata or {}).items():
        attributes[f"langfuse.observation.metadata.{key}"] = value
    return attributes


def _attach_otel_ids(record: dict[str, Any], handle: telemetry_bridge.BridgeSpan) -> None:
    if handle.otel_trace_id:
        record["otel_trace_id"] = handle.otel_trace_id
    if handle.otel_span_id:
        record["otel_span_id"] = handle.otel_span_id


class TraceRecorder:
    """记录一棵 Trace 树；Span 通过 ContextVar 自动关联父子关系。"""

    def __init__(self, sink: TraceSink):
        self.sink = sink

    @contextmanager
    def trace(
        self,
        name: str = "research",
        *,
        run_id: str | None = None,
        session_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> Iterator[str]:
        trace_id = new_id("trace")
        started = perf_counter()
        parent = current_context()
        effective_run_id = run_id or (parent.run_id if parent else None)
        effective_session_id = session_id or (parent.session_id if parent else None)
        node_id = parent.node_id if parent else None
        token = set_context(
            TraceContext(
                trace_id=trace_id,
                run_id=effective_run_id,
                session_id=effective_session_id,
                node_id=node_id,
            )
        )
        # recorder 的上下文管理器必须在同一 task 内进出;LangGraph 在 task 创建时
        # 复制 contextvars,并行分支各自成枝,与自研 _context 同语义。
        try:
            with telemetry_bridge.begin_span(
                name,
                kind="trace",
                attributes=_bridge_attributes(
                    trace_id=trace_id,
                    run_id=effective_run_id,
                    session_id=effective_session_id,
                    node_id=node_id,
                    metadata=metadata,
                ),
            ) as bridge_span:
                started_record: dict[str, Any] = {
                    "record_type": "trace",
                    "event_type": "trace_started",
                    "trace_id": trace_id,
                    "run_id": effective_run_id,
                    "session_id": effective_session_id,
                    "node_id": node_id,
                    "name": name,
                    "metadata": dict(metadata or {}),
                }
                _attach_otel_ids(started_record, bridge_span)
                self.sink.write(started_record)
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
                    error = (str(exc).strip() or type(exc).__name__)[:500]
                    raise
                finally:
                    bridge_span.finish(status, error)
                    record = {
                        "record_type": "trace",
                        "event_type": f"trace_{status}",
                        "trace_id": trace_id,
                        "run_id": effective_run_id,
                        "session_id": effective_session_id,
                        "node_id": node_id,
                        "duration_ms": round((perf_counter() - started) * 1000, 2),
                        "metadata": dict(metadata or {}),
                    }
                    if error:
                        record["error"] = error
                    _attach_otel_ids(record, bridge_span)
                    self.sink.write(record)
        finally:
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
        run_id = parent.run_id if parent else None
        session_id = parent.session_id if parent else None
        node_id = parent.node_id if parent else None
        token = set_context(
            TraceContext(
                trace_id,
                span_id,
                run_id=run_id,
                session_id=session_id,
                node_id=node_id,
            )
        )
        try:
            with telemetry_bridge.begin_span(
                name,
                kind=kind,
                attributes=_bridge_attributes(
                    trace_id=trace_id,
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    run_id=run_id,
                    session_id=session_id,
                    node_id=node_id,
                ),
            ) as bridge_span:
                started_record: dict[str, Any] = {
                    "record_type": "span",
                    "event_type": "span_started",
                    "trace_id": trace_id,
                    "span_id": span_id,
                    "parent_span_id": parent_span_id,
                    "run_id": run_id,
                    "session_id": session_id,
                    "node_id": node_id,
                    "name": name,
                    "kind": kind,
                }
                _attach_otel_ids(started_record, bridge_span)
                self.sink.write(started_record)
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
                    error = (str(exc).strip() or type(exc).__name__)[:500]
                    raise
                finally:
                    bridge_span.finish(status, error)
                    record = {
                        "record_type": "span",
                        "event_type": f"span_{status}",
                        "trace_id": trace_id,
                        "span_id": span_id,
                        "parent_span_id": parent_span_id,
                        "run_id": run_id,
                        "session_id": session_id,
                        "node_id": node_id,
                        "name": name,
                        "kind": kind,
                        "duration_ms": round((perf_counter() - started) * 1000, 2),
                    }
                    if error:
                        record["error"] = error
                    _attach_otel_ids(record, bridge_span)
                    self.sink.write(record)
        finally:
            reset_context(token)
