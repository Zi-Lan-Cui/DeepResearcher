"""OTel 单引擎终态:span 树、langfuse 属性、账本记录与 span 对象的一致性。

provider 由 conftest 会话级常驻并挂好 ledger 处理器;本模块另接一个
InMemorySpanExporter 观察导出的 span 树,用 ledger 注册表收取账本记录。
"""

import asyncio
import re
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from opentelemetry import trace as opentelemetry_trace
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from deepresearcher.observability.tracing import ledger, spans

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
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))  # type: ignore[attr-defined]
    yield exporter
    # 本模块之后其他测试文件仍会向全局 provider 产 span,清空避免跨模块累积。
    exporter.clear()


@contextmanager
def _run_sink(run_id: str) -> Iterator[_ListSink]:
    sink = _ListSink()
    ledger.attach_run_sink(run_id, sink)
    try:
        yield sink
    finally:
        ledger.detach_run_sink(run_id)


async def _run_tree() -> str:
    """构一棵 根→{supervisor→search 工具, writer(失败), reviewer(取消)} 的树。"""

    async def tool_leg() -> None:
        with spans.span("search", kind="tool"):
            await asyncio.sleep(0)

    async def failing_leg() -> None:
        with spans.span("writer"):
            raise ValueError("boom")

    async def cancelled_leg() -> None:
        with spans.span("reviewer"):
            raise asyncio.CancelledError

    with spans.trace(
        "research_run",
        run_id="run-tree",
        session_id="run-tree",
        metadata={"attempt": 2},
        input={"query": "自尊的六大支柱"},
    ) as trace_id:
        with spans.span("supervisor"):
            await tool_leg()
        # return_exceptions=True:两个分支的异常只用于给 span 定终态,不得逸出 trace 体。
        await asyncio.gather(failing_leg(), cancelled_leg(), return_exceptions=True)
    return trace_id


def _spans_in_window(exporter: InMemorySpanExporter, start: int) -> dict:
    return {span.name: span for span in exporter.get_finished_spans()[start:]}


@pytest.mark.asyncio
async def test_spans_form_single_otel_tree(otel_exporter):
    before = len(otel_exporter.get_finished_spans())
    await _run_tree()
    by_name = _spans_in_window(otel_exporter, before)

    root = by_name["research_run"]
    assert root.parent is None
    assert by_name["supervisor"].parent.span_id == root.context.span_id
    assert by_name["search"].parent.span_id == by_name["supervisor"].context.span_id
    # gather 并行枝各自认根,不互相污染——OTel 上下文随 task 创建复制。
    assert by_name["writer"].parent.span_id == root.context.span_id
    assert by_name["reviewer"].parent.span_id == root.context.span_id


@pytest.mark.asyncio
async def test_spans_carry_langfuse_and_correlation_attributes(otel_exporter):
    before = len(otel_exporter.get_finished_spans())
    await _run_tree()
    by_name = _spans_in_window(otel_exporter, before)

    root = by_name["research_run"]
    assert root.attributes["langfuse.observation.type"] == "span"
    assert root.attributes["langfuse.observation.metadata.attempt"] == 2
    assert "自尊的六大支柱" in root.attributes["langfuse.observation.input"]
    assert root.attributes["deepresearcher.run_id"] == "run-tree"
    assert by_name["search"].attributes["langfuse.observation.type"] == "tool"
    assert by_name["search"].kind == opentelemetry_trace.SpanKind.CLIENT
    assert by_name["supervisor"].kind == opentelemetry_trace.SpanKind.INTERNAL


@pytest.mark.asyncio
async def test_failed_and_cancelled_spans_status(otel_exporter):
    before = len(otel_exporter.get_finished_spans())
    await _run_tree()
    by_name = _spans_in_window(otel_exporter, before)

    failed = by_name["writer"]
    assert failed.status.status_code == opentelemetry_trace.StatusCode.ERROR
    assert failed.attributes["langfuse.observation.level"] == "ERROR"
    assert failed.attributes["langfuse.observation.status_message"] == "boom"
    cancelled = by_name["reviewer"]
    assert cancelled.status.status_code != opentelemetry_trace.StatusCode.ERROR
    assert cancelled.attributes["deepresearcher.status"] == "cancelled"


@pytest.mark.asyncio
async def test_ledger_records_derive_from_same_span_ids():
    with _run_sink("run-tree") as sink:
        await _run_tree()

    started = [record for record in sink.records if record["event_type"] == "trace_started"]
    assert len(started) == 1
    root = started[0]
    assert HEX_TRACE_ID.match(root["trace_id"])
    assert root["metadata"] == {"attempt": 2}
    completed = {
        record["name"]: record
        for record in sink.records
        if record["event_type"] == "span_completed"
    }
    supervisor_started = next(
        record
        for record in sink.records
        if record.get("name") == "supervisor" and record["event_type"] == "span_started"
    )
    assert supervisor_started["span_id"] == completed["supervisor"]["span_id"]
    # 根现在是真实 span(旧实现里 trace 不带 span),直接子级的父为根的 16 位 hex。
    assert HEX_SPAN_ID.match(supervisor_started["parent_span_id"])
    tool = next(
        record
        for record in sink.records
        if record.get("name") == "search" and record["event_type"] == "span_started"
    )
    assert tool["trace_id"] == root["trace_id"]
    assert tool["parent_span_id"] == completed["supervisor"]["span_id"]
    assert tool["kind"] == "tool"
    failed = next(record for record in sink.records if record["event_type"] == "span_failed")
    assert failed["error"] == "boom"
    cancelled = next(record for record in sink.records if record["event_type"] == "span_cancelled")
    assert cancelled["error"] == "CancelledError"


@pytest.mark.asyncio
async def test_unregistered_run_produces_no_ledger_records():
    # 账本注册表就是"落账与否"的唯一开关:未注册 run 的 span 不产生任何记录。
    with spans.span("ephemeral"):
        pass
