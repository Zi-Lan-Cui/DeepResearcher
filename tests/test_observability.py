import asyncio
import json

import pytest

from deepresearcher.graph import summarize_node_result
from deepresearcher.observability.events import (
    JsonlSink,
    make_audit_event,
    make_node_event,
)
from deepresearcher.observability.tracing import ledger, spans
from deepresearcher.service.events.projector import project


def _ledger_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_trace_records_nested_spans(tmp_path):
    path = tmp_path / "traces.jsonl"
    ledger.attach_run_sink("run-nest", JsonlSink(path))
    try:
        with spans.trace(
            "test", run_id="run-nest", metadata={"attempt": 2, "resume": True}
        ) as trace_id:
            with spans.span("planner"):
                with spans.span("llm", kind="llm"):
                    pass
    finally:
        ledger.detach_run_sink("run-nest")

    records = _ledger_rows(path)
    assert records[0]["event_type"] == "trace_started"
    assert records[0]["metadata"] == {"attempt": 2, "resume": True}
    assert records[-1]["event_type"] == "trace_completed"
    assert records[-1]["metadata"] == {"attempt": 2, "resume": True}
    child_start = next(
        record
        for record in records
        if record.get("name") == "llm" and record["event_type"] == "span_started"
    )
    assert child_start["trace_id"] == trace_id
    planner_start = next(
        record
        for record in records
        if record.get("name") == "planner" and record["event_type"] == "span_started"
    )
    assert child_start["parent_span_id"] == planner_start["span_id"]


def test_trace_records_cancellation_without_error_status(tmp_path):
    path = tmp_path / "traces.jsonl"
    ledger.attach_run_sink("run-cancel", JsonlSink(path))
    try:
        with pytest.raises(asyncio.CancelledError):
            with spans.trace("cancelled", run_id="run-cancel"):
                raise asyncio.CancelledError()
    finally:
        ledger.detach_run_sink("run-cancel")

    records = _ledger_rows(path)
    assert records[-1]["event_type"] == "trace_cancelled"
    assert records[-1]["error"] == "CancelledError"


def test_failed_span_terminal_record_is_self_describing(tmp_path):
    path = tmp_path / "traces.jsonl"
    ledger.attach_run_sink("run-fail", JsonlSink(path))
    try:
        with pytest.raises(RuntimeError):
            with spans.trace("test", run_id="run-fail"):
                with spans.span("source_fetch", kind="tool"):
                    raise RuntimeError()
    finally:
        ledger.detach_run_sink("run-fail")

    record = next(row for row in _ledger_rows(path) if row["event_type"] == "span_failed")
    assert record["name"] == "source_fetch"
    assert record["kind"] == "tool"
    assert "parent_span_id" in record
    assert record["error"] == "RuntimeError"


def test_audit_events_are_identifiable_and_keep_domain_payload():
    event = make_audit_event(
        "research_task_completed",
        trace_id="trace-1",
        payload={"task_id": "r1-1", "evidence_count": 2},
    )

    assert event["record_type"] == "event"
    assert event["event_id"].startswith("evt-")
    assert event["event_type"] == "research_task_completed"
    assert event["payload"]["evidence_count"] == 2


def test_node_event_accepts_summary_payload():
    event = make_node_event("render_final_report", "completed", payload={"citation_count": 3})

    assert event.node == "render_final_report"
    assert event.payload == {"citation_count": 3}


def test_clarifier_summary_survives_instrumentation_and_projection():
    payload = summarize_node_result(
        {
            "clarified_query": "Redis 有什么作用？",
            "research_brief": "比较 Redis 在后端与 Agent 系统中的职责和知识要求",
        },
        max_text_chars=1_000,
    )
    frame = project(
        {
            "event_type": "node_completed",
            "node": "clarify",
            "seq": 1,
            "payload": payload,
        }
    )

    assert payload["research_brief"] == "比较 Redis 在后端与 Agent 系统中的职责和知识要求"
    assert frame is not None
    assert frame.data["text"] == payload["research_brief"]


def test_node_event_carries_correlation_fields():
    event = make_node_event(
        "writer",
        "completed",
        run_id="run-1",
        session_id="session-1",
        node_id="writer",
        payload={"report_chars": 5000, "report_preview": "..."},
    )

    assert event.run_id == "run-1"
    assert event.session_id == "session-1"
    assert event.node_id == "writer"
    assert event.payload["report_chars"] == 5000
