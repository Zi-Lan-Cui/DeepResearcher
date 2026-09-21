import asyncio
import json

import pytest

from deepresearcher.observability.events import (
    JsonlSink,
    make_artifact_event,
    make_audit_event,
    make_node_event,
)
from deepresearcher.observability.instrumentation import _node_result_summary
from deepresearcher.observability.tracing import TraceRecorder
from deepresearcher.service.events.projector import project


def test_trace_records_nested_spans(tmp_path):
    path = tmp_path / "traces.jsonl"
    recorder = TraceRecorder(JsonlSink(path))
    with recorder.trace("test", metadata={"attempt": 2, "resume": True}) as trace_id:
        with recorder.span("planner") as parent_span_id:
            with recorder.span("llm", kind="llm"):
                pass

    records = [json.loads(line) for line in path.read_text().splitlines()]
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
    assert child_start["parent_span_id"] == parent_span_id


def test_trace_records_cancellation_without_unbound_status(tmp_path):
    path = tmp_path / "traces.jsonl"
    recorder = TraceRecorder(JsonlSink(path))

    with pytest.raises(asyncio.CancelledError):
        with recorder.trace("cancelled"):
            raise asyncio.CancelledError()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[-1]["event_type"] == "trace_cancelled"
    assert records[-1]["error"] == "CancelledError"


def test_failed_span_terminal_record_is_self_describing(tmp_path):
    path = tmp_path / "traces.jsonl"
    recorder = TraceRecorder(JsonlSink(path))

    with pytest.raises(RuntimeError):
        with recorder.trace("test"):
            with recorder.span("source_fetch", kind="tool"):
                raise RuntimeError()

    record = next(
        row
        for row in (json.loads(line) for line in path.read_text().splitlines())
        if row["event_type"] == "span_failed"
    )
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
    payload = _node_result_summary(
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


def test_events_and_artifacts_have_separate_records_and_correlation():
    event = make_node_event(
        "writer",
        "completed",
        run_id="run-1",
        session_id="session-1",
        node_id="writer",
        payload={"report_chars": 5000, "report_preview": "..."},
    )
    artifact = make_artifact_event(
        "output",
        "完整报告",
        name="writer_report",
        run_id="run-1",
        session_id="session-1",
        node_id="writer",
    )

    assert event.run_id == "run-1"
    assert event.session_id == "session-1"
    assert event.node_id == "writer"
    assert event.payload["report_chars"] == 5000
    assert artifact["record_type"] == "artifact"
    assert artifact["content"] == "完整报告"
    assert artifact["run_id"] == "run-1"
