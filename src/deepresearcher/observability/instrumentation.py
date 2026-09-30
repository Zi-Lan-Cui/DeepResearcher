"""节点级观测:instrument_node 记录生命周期(started/completed/failed/cancelled)并原样重抛。

本模块只记录、不裁决——失败转 RunStatus 的收口归 node_runner.execute_node;
两处失败事件的 error/code/retryable 口径共用 schemas 的 failure_event_fields。
也不解读业务字段:节点产出的字段摘要由装配层以 summarize 注入,
缺省只登记 updated_fields。
"""

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.errors import GraphBubbleUp

from deepresearcher.observability.events.models import make_node_event
from deepresearcher.observability.events.sink import JsonlSink
from deepresearcher.observability.logging_config import get_logger
from deepresearcher.observability.tracing.context import (
    SpanContext,
    bind_context,
    current_span_context,
    new_id,
)
from deepresearcher.observability.tracing.spans import span
from deepresearcher.schemas import failure_event_fields


def instrument_node(
    name: str,
    node: Callable[..., dict[str, Any] | Awaitable[dict[str, Any]]],
    *,
    logger=None,
    event_sink: JsonlSink | None = None,
    max_text_chars: int,
    summarize: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> Callable[..., Awaitable[dict[str, Any]]]:
    """统一记录节点 Log、Event 和 Span。业务字段摘要经 summarize 注入,缺省登记 updated_fields。"""
    summary_of = summarize if summarize is not None else _base_summary
    log = logger or get_logger("deepresearcher.observability.instrumentation")

    async def wrapped(state: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        state.setdefault("run_id", new_id("run"))
        with bind_context(
            run_id=state.get("run_id"), session_id=state.get("session_id"), node_id=name
        ):
            log.info("node_started", extra={"node": name})
            if event_sink is not None:
                event_sink.write(
                    make_node_event(
                        name,
                        "started",
                        node_id=name,
                        payload={"query_chars": len(str(state.get("query", "")))},
                    )
                )
            try:
                span_link: SpanContext | None = None
                with span(name):
                    # 节点生命周期事件统一归属节点 span;span 进入即冻结身份。
                    span_link = current_span_context()
                    value = node(state)
                    if inspect.isawaitable(value):
                        value = await value
                    result = dict(value)
                duration_ms = (time.perf_counter() - started) * 1000
                output = summary_of(result)
                log.info(
                    "node_completed duration_ms=%.2f updated_fields=%s summary=%s",
                    duration_ms,
                    output["updated_fields"],
                    output,
                    extra={"node": name},
                )
                event = make_node_event(
                    name,
                    "completed",
                    link=span_link,
                    node_id=name,
                    duration_ms=duration_ms,
                    payload=output,
                )
                if event_sink is not None:
                    event_sink.write(event)
                return result
            except GraphBubbleUp:
                # interrupt() 是子图控制流，由根图持久化，不是节点失败。
                raise
            except (asyncio.CancelledError, KeyboardInterrupt) as exc:
                duration_ms = (time.perf_counter() - started) * 1000
                log.info("node_cancelled duration_ms=%.2f", duration_ms, extra={"node": name})
                event = make_node_event(
                    name,
                    "cancelled",
                    link=span_link,
                    node_id=name,
                    duration_ms=duration_ms,
                    error=str(exc),
                    payload=summary_of(state),
                )
                if event_sink is not None:
                    event_sink.write(event)
                raise
            except Exception as exc:
                duration_ms = (time.perf_counter() - started) * 1000
                log.exception("node_failed duration_ms=%.2f", duration_ms, extra={"node": name})
                _, fields = failure_event_fields(name, exc)
                event = make_node_event(
                    name,
                    "failed",
                    link=span_link,
                    node_id=name,
                    duration_ms=duration_ms,
                    **fields,
                )
                if event_sink is not None:
                    event_sink.write(event)
                raise

    return wrapped


def _base_summary(value: dict[str, Any]) -> dict[str, Any]:
    """与业务无关的最小摘要:更新了哪些通道。"""
    return {"updated_fields": sorted(value.keys())}
