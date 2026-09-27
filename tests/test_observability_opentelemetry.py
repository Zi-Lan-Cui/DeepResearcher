"""TraceRecorder 与 OpenTelemetry 的集成:统一 span 树、账本双 id 关联、降级形。

本模块在全进程第一次设置 SDK TracerProvider(一次性 API);更早运行的模块
(test_observability.py)保持无 provider 形态,由 pytest 的文件序自然保证。
"""

import asyncio
import re
from collections.abc import Iterator

import pytest
from opentelemetry import trace as opentelemetry_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from deepresearcher.observability.tracing import telemetry_bridge
from deepresearcher.observability.tracing.recorder import TraceRecorder

HEX_TRACE_ID = re.compile(r"^[0-9a-f]{32}$")
HEX_SPAN_ID = re.compile(r"^[0-9a-f]{16}$")


class _ListSink:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def write(self, record) -> None:
        self.records.append(record)


@pytest.fixture(scope="module")
def otel_exporter() -> Iterator[InMemorySpanExporter]:
    provider = opentelemetry_trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        opentelemetry_trace.set_tracer_provider(provider)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield exporter
    # 本模块之后其他测试文件仍会向全局 provider 产 span,清空避免跨模块累积。
    exporter.clear()


async def _run_tree(recorder: TraceRecorder) -> str:
    """构一棵 根→{supervisor→search 工具, writer(失败), reviewer(取消)} 的树。"""

    async def tool_leg() -> None:
        with recorder.span("search", kind="tool"):
            await asyncio.sleep(0)

    async def failing_leg() -> None:
        with recorder.span("writer"):
            raise ValueError("boom")

    async def cancelled_leg() -> None:
        with recorder.span("reviewer"):
            raise asyncio.CancelledError

    with recorder.trace(
        "research_run", run_id="run-tree", session_id="run-tree", metadata={"attempt": 2}
    ) as trace_id:
        with recorder.span("supervisor"):
            await tool_leg()
        # return_exceptions=True:两个分支的异常只用于给 span 定终态,不得逸出 trace 体。
        await asyncio.gather(failing_leg(), cancelled_leg(), return_exceptions=True)
    return trace_id


@pytest.mark.asyncio
async def test_recorder_spans_form_single_otel_tree(otel_exporter):
    recorder = TraceRecorder(_ListSink())
    before = len(otel_exporter.get_finished_spans())
    await _run_tree(recorder)
    spans = [span for span in otel_exporter.get_finished_spans()[before:]]
    by_name = {span.name: span for span in spans}

    root = by_name["research_run"]
    assert root.parent is None
    assert by_name["supervisor"].parent.span_id == root.context.span_id
    assert by_name["search"].parent.span_id == by_name["supervisor"].context.span_id
    # gather 并行枝各自认根,不互相污染——OTel 上下文随 task 创建复制。
    assert by_name["writer"].parent.span_id == root.context.span_id
    assert by_name["reviewer"].parent.span_id == root.context.span_id


@pytest.mark.asyncio
async def test_otel_spans_carry_langfuse_and_correlation_attributes(otel_exporter):
    recorder = TraceRecorder(_ListSink())
    before = len(otel_exporter.get_finished_spans())
    trace_id = await _run_tree(recorder)
    by_name = _spans_by_name_in_window(otel_exporter, before)

    root = by_name["research_run"]
    assert root.attributes["langfuse.observation.type"] == "span"
    assert root.attributes["langfuse.observation.metadata.attempt"] == 2
    assert root.attributes["deepresearcher.trace_id"] == trace_id
    assert root.attributes["deepresearcher.run_id"] == "run-tree"
    assert by_name["search"].attributes["langfuse.observation.type"] == "tool"
    assert by_name["search"].kind == opentelemetry_trace.SpanKind.CLIENT
    assert by_name["supervisor"].kind == opentelemetry_trace.SpanKind.INTERNAL


def _spans_by_name_in_window(exporter: InMemorySpanExporter, start: int) -> dict:
    return {span.name: span for span in exporter.get_finished_spans()[start:]}


@pytest.mark.asyncio
async def test_failed_and_cancelled_spans_status(otel_exporter):
    recorder = TraceRecorder(_ListSink())
    before = len(otel_exporter.get_finished_spans())
    await _run_tree(recorder)
    by_name = _spans_by_name_in_window(otel_exporter, before)

    failed = by_name["writer"]
    assert failed.status.status_code == opentelemetry_trace.StatusCode.ERROR
    assert failed.attributes["langfuse.observation.level"] == "ERROR"
    assert failed.attributes["langfuse.observation.status_message"] == "boom"
    cancelled = by_name["reviewer"]
    assert cancelled.status.status_code != opentelemetry_trace.StatusCode.ERROR
    assert cancelled.attributes["deepresearcher.status"] == "cancelled"


@pytest.mark.asyncio
async def test_ledger_records_link_to_otel_spans_by_hex_ids(otel_exporter):
    sink = _ListSink()
    recorder = TraceRecorder(sink)
    before = len(otel_exporter.get_finished_spans())
    await _run_tree(recorder)
    by_name = _spans_by_name_in_window(otel_exporter, before)

    completed = [record for record in sink.records if record["event_type"] == "span_completed"]
    assert completed
    for record in completed:
        assert HEX_TRACE_ID.match(record["otel_trace_id"])
        assert HEX_SPAN_ID.match(record["otel_span_id"])
        otel_span = by_name[record["name"]]
        assert record["otel_span_id"] == format(otel_span.context.span_id, "016x")
        assert record["otel_trace_id"] == format(otel_span.context.trace_id, "032x")


@pytest.mark.asyncio
async def test_bridge_disabled_records_match_pre_otel_shape():
    sink = _ListSink()
    recorder = TraceRecorder(sink)
    with telemetry_bridge.override_enabled(False):
        with recorder.trace("research_run", run_id="run-off") as trace_id:
            with recorder.span("supervisor"):
                pass
    started = sink.records[0]
    assert started["event_type"] == "trace_started"
    assert set(started) == {
        "record_type",
        "event_type",
        "trace_id",
        "run_id",
        "session_id",
        "node_id",
        "name",
        "metadata",
    }
    assert started["trace_id"] == trace_id
    span_record = sink.records[1]
    assert "otel_trace_id" not in span_record and "otel_span_id" not in span_record
    assert span_record["event_type"] == "span_started"
    assert sink.records[2]["trace_id"] == trace_id  # span 仍认自研父
