"""LedgerSpanProcessor：把本项目的 span 写成事件日志记录（run_events/JSONL）。

on_start 写 *_started，on_end 写 *_completed/failed/cancelled。只处理本
instrumentation scope 的 span，按 deepresearcher.run_id 路由到该 run 的 sink；
查无 sink 时丢弃并告警一次，写 sink 抛错不得波及 span 路径。
"""

import json
import logging

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.trace import StatusCode

from deepresearcher.observability.events.sink import TraceSink

logger = logging.getLogger("deepresearcher.observability.tracing.ledger")

_SPAN_SINKS: dict[str, TraceSink] = {}
_warned_runs: set[str] = set()


def attach_to_provider(provider) -> None:
    """给 provider 注册事件日志处理器，每 provider 只注册一次(装配根与测试共用)。"""
    if not getattr(provider, "_deepresearcher_ledger_attached", False):
        provider.add_span_processor(LedgerSpanProcessor())
        provider._deepresearcher_ledger_attached = True  # type: ignore[attr-defined]


def attach_run_sink(run_id: str, sink: TraceSink) -> None:
    """登记某 run 的事件日志 sink;RunExecutor 在 execute 入口调用。"""
    _SPAN_SINKS[run_id] = sink


def detach_run_sink(run_id: str) -> None:
    """注销 sink；调用方须在本 run 的根 span 结束后执行，否则终态记录会丢。"""
    _SPAN_SINKS.pop(run_id, None)
    _warned_runs.discard(run_id)


def _hex(value: int | None, width: int) -> str | None:
    return format(value, f"0{width}x") if value else None


def _labels(span: ReadableSpan) -> tuple:
    attributes = span.attributes or {}
    return (
        attributes.get("deepresearcher.run_id"),
        attributes.get("deepresearcher.session_id"),
        attributes.get("deepresearcher.node_id"),
    )


def _terminal_fields(span: ReadableSpan) -> tuple[str, str | None]:
    """(status, error):cancelled 与 GraphInterrupt 单独归类，failed 取 status.description。

    LangGraph 的澄清挂起以 Interrupt 异常文本进入 span status，它是设计内
    中断；记成 failed 会把正常挂起混进真错误统计。
    """
    attributes = span.attributes or {}
    if attributes.get("deepresearcher.status") == "cancelled":
        cancelled_error = attributes.get("deepresearcher.error")
        return "cancelled", str(cancelled_error) if cancelled_error else None
    if span.status.status_code == StatusCode.ERROR:
        description = span.status.description or "error"
        if description.startswith(("Interrupt(", "(Interrupt(")):
            return "interrupted", description
        return "failed", description
    return "completed", None


class LedgerSpanProcessor(SpanProcessor):
    """OTel 处理链上的事件日志写入端；与 langfuse/console 处理器并列、互不依赖。"""

    def on_start(self, span: ReadableSpan, parent_context=None) -> None:
        # started/终态的分流在 _record 内按 end_time 判定，两个钩子共用组装。
        self._write(span)

    def on_end(self, span: ReadableSpan) -> None:
        self._write(span)

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True

    def _write(self, span: ReadableSpan) -> None:
        scope = span.instrumentation_scope
        if scope is None or scope.name != "deepresearcher.tracing":
            return
        run_id, session_id, node_id = _labels(span)
        sink = _SPAN_SINKS.get(run_id) if run_id else None
        if sink is None:
            if run_id and run_id not in _warned_runs:
                _warned_runs.add(run_id)
                logger.warning("ledger_sink_missing run_id=%s span=%s", run_id, span.name)
            return
        kind = (span.attributes or {}).get("deepresearcher.kind", "node")
        record_type = "trace" if kind == "trace" else "span"
        record = self._record(
            span, record_type=record_type, run_id=run_id, session_id=session_id, node_id=node_id
        )
        if record is None:
            return
        try:
            sink.write(record)
        except Exception:
            logger.warning("ledger_write_failed span=%s", span.name, exc_info=True)

    def _record(self, span, *, record_type, run_id, session_id, node_id) -> dict | None:
        trace_id = _hex(span.context.trace_id, 32) if span.context else None
        if record_type == "trace":
            base = {
                "record_type": "trace",
                "trace_id": trace_id,
                "run_id": run_id,
                "session_id": session_id,
                "node_id": node_id,
                "name": span.name,
            }
            metadata = (span.attributes or {}).get("deepresearcher.metadata")
            base["metadata"] = json.loads(metadata) if metadata else {}
        else:
            base = {
                "record_type": "span",
                "trace_id": trace_id,
                "span_id": _hex(span.context.span_id, 16) if span.context else None,
                "parent_span_id": _hex(span.parent.span_id, 16) if span.parent else None,
                "run_id": run_id,
                "session_id": session_id,
                "node_id": node_id,
                "name": span.name,
                "kind": (span.attributes or {}).get("deepresearcher.kind", "node"),
            }
        if span.end_time is None:
            base["event_type"] = f"{record_type}_started"
            return base
        status, error = _terminal_fields(span)
        base["event_type"] = f"{record_type}_{status}"
        base["duration_ms"] = round((span.end_time - span.start_time) / 1_000_000, 2)
        if error:
            base["error"] = error
        return base
