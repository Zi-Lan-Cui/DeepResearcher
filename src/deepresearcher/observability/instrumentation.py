"""节点级观测:instrument_node 记录生命周期(started/completed/failed/cancelled)并原样重抛。

本模块只记录、不裁决——失败转 RunStatus 的收口归 node_runner.execute_node;
两处失败事件的 error/code/retryable 口径共用 failure_event_fields。
"""

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.errors import GraphBubbleUp

from deepresearcher.observability.events.models import failure_event_fields, make_node_event
from deepresearcher.observability.events.sink import JsonlSink
from deepresearcher.observability.logging_config import get_logger
from deepresearcher.observability.tracing.context import (
    SpanContext,
    bind_context,
    current_span_context,
    new_id,
)
from deepresearcher.observability.tracing.recorder import TraceRecorder


def instrument_node(
    name: str,
    node: Callable[..., dict[str, Any] | Awaitable[dict[str, Any]]],
    *,
    logger=None,
    event_sink: JsonlSink | None = None,
    trace_recorder: TraceRecorder | None = None,
    max_text_chars: int,
) -> Callable[..., Awaitable[dict[str, Any]]]:
    """统一记录节点 Log、Event 和 Span。"""
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
                with trace_recorder.span(name) if trace_recorder is not None else _null_context():
                    # 节点生命周期事件统一归属节点 span;span 进入即冻结身份。
                    span_link = current_span_context()
                    value = node(state)
                    if inspect.isawaitable(value):
                        value = await value
                    result = dict(value)
                duration_ms = (time.perf_counter() - started) * 1000
                output = _node_result_summary(result, max_text_chars=max_text_chars)
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
                result["node_events"] = [*result.get("node_events", []), event]
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
                    payload=_node_result_summary(state, max_text_chars=max_text_chars),
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


def _node_result_summary(result: dict[str, Any], *, max_text_chars: int) -> dict[str, Any]:
    """记录运行元数据和有界预览；完整正文不进入事件流或普通日志。"""
    summary: dict[str, Any] = {"updated_fields": sorted(result.keys())}
    for key in (
        "route",
        "answer_mode",
        "evidence_count",
        "source_count",
    ):
        if key in result:
            summary[key] = result[key]
    if "route_reason" in result:
        summary["route_reason"] = str(result["route_reason"])[:240]
    if "clarified_query" in result:
        summary["clarified_query"] = str(result["clarified_query"])[:300]
    if "research_brief" in result:
        summary["research_brief"] = str(result["research_brief"])[:max_text_chars]
    run = result.get("run")
    if run is not None:
        summary["phase"] = getattr(run, "phase", None)
        summary["terminal_reason"] = getattr(run, "terminal_reason", "")
    for key in ("supervisor", "writer", "review"):
        section = result.get(key)
        if section is not None:
            summary[f"{key}_status"] = getattr(section, "status", None)
            if key == "supervisor":
                summary["current_round"] = getattr(section, "current_round", 0)
            if key in {"writer", "review"}:
                feedback = getattr(section, "feedback", "")
                if feedback:
                    summary[f"{key}_feedback"] = str(feedback)[:500]
    if "writer_draft" in result:
        text = str(result["writer_draft"])
        summary["writer_draft_chars"] = len(text)
        summary["writer_draft_preview"] = text[:max_text_chars]
    review = result.get("review")
    review_issues = getattr(review, "issues", None) if review is not None else None
    if review_issues:
        summary["review_issues"] = [
            {
                "severity": item.severity,
                "claim": item.claim[:200],
                "reason": item.reason[:300],
            }
            for item in review_issues
        ]
    if "report" in result:
        text = str(result["report"])
        summary["report_chars"] = len(text)
        summary["report_preview"] = text[:max_text_chars]
    directive_brief = getattr(result.get("writer_directive"), "report_brief", None)
    if directive_brief is not None:
        summary["report_brief"] = str(directive_brief)[:max_text_chars]
    research = result.get("supervisor")
    coverage_gaps = getattr(research, "coverage_gaps", None) if research is not None else None
    if coverage_gaps:
        summary["coverage_gaps"] = list(coverage_gaps)
    for key in (
        "citations",
        "citation_decisions",
        "task_results",
        "evidences",
        "paragraph_bindings",
    ):
        if key in result:
            summary[f"{key}_count"] = len(result[key])
    if "citation_decisions" in result:
        summary["citation_decisions"] = [
            {
                "id": str(item.get("id", "")),
                "supported": bool(item.get("supported", False)),
                "reason": str(item.get("reason", ""))[:240],
            }
            for item in result["citation_decisions"][:20]
        ]
    return summary


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False
