import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage

from deepresearcher.agents.middleware import AgentEventsMiddleware
from deepresearcher.agents.researcher.state import (
    ResearcherContext,
    ResearcherDeps,
    ResearcherTaskState,
)
from deepresearcher.config import AgentConfig
from deepresearcher.observability.scope import AgentScope


def _real_researcher_context() -> ResearcherContext:
    """生产形状的真实 ResearcherContext,中间件契约字段 scope 必须在场。

    SimpleNamespace 假件测中间件如何对待有 scope 的对象,本函数测真 context
    是否提供中间件所需字段;两个方向都设守卫,删字段后测试会失败,
    而不是事件静默失去归因。
    """
    task = {
        "id": "task-0001",
        "run_id": "run-1",
        "question": "Redis 恢复",
    }
    return ResearcherContext(
        deps=ResearcherDeps(
            config=AgentConfig(),
            search_tool=object(),  # 中间件不触碰业务零件
            reader_tool=object(),
            material_store=None,
            emit=lambda *_: None,
        ),
        scope=AgentScope.from_task(task, agent_name="Researcher"),
        task=task,  # type: ignore[typeddict-item]
        task_state=ResearcherTaskState(),
    )


@pytest.mark.asyncio
async def test_real_researcher_context_carries_middleware_attribution_fields():
    events: list[tuple[str, dict[str, object]]] = []
    middleware = AgentEventsMiddleware(
        "Researcher",
        event_slug="researcher",
        run_limit=5,
        emit=lambda event_type, payload: events.append((event_type, payload)),
    )
    request = SimpleNamespace(
        tool_call={"name": "ReadSources", "id": "call-9", "args": {"candidate_ids": ["c-1"]}},
        state={"run_model_call_count": 1},
        runtime=SimpleNamespace(context=_real_researcher_context()),
    )

    async def handler(_request):
        return ToolMessage(content="ok", tool_call_id="call-9")

    await middleware.awrap_tool_call(request, handler)

    started = events[0][1]
    assert started["run_id"] == "run-1"
    assert started["task_id"] == "task-0001"
    assert started["operation_id"] == "task-0001"


def _request() -> SimpleNamespace:
    return SimpleNamespace(
        tool_call={
            "name": "SearchSources",
            "id": "call-1",
            "args": {"query": "Redis 恢复", "limit": 5},
        },
        state={"run_model_call_count": 2},
        runtime=SimpleNamespace(
            context=SimpleNamespace(
                scope=AgentScope(
                    run_id="run-1",
                    agent_name="Researcher",
                    task_id="task-1",
                )
            )
        ),
    )


@pytest.mark.asyncio
async def test_tool_lifecycle_records_scope_and_preserves_result():
    events: list[tuple[str, dict[str, object]]] = []
    middleware = AgentEventsMiddleware(
        "Researcher",
        event_slug="researcher",
        run_limit=5,
        emit=lambda event_type, payload: events.append((event_type, payload)),
    )
    expected = ToolMessage(content="ok", tool_call_id="call-1")

    async def handler(request):
        assert request.tool_call["id"] == "call-1"
        return expected

    result = await middleware.awrap_tool_call(_request(), handler)

    assert result is expected
    assert [event_type for event_type, _ in events] == [
        "researcher_tool_started",
        "researcher_tool_completed",
    ]
    started = events[0][1]
    assert started["run_id"] == "run-1"
    assert started["task_id"] == "task-1"
    assert started["tool_name"] == "SearchSources"
    assert started["tool_call_id"] == "call-1"
    assert started["turn"] == 2
    assert started["argument_keys"] == ["limit", "query"]
    assert '"query": "Redis 恢复"' in started["arguments_preview"]
    completed = events[1][1]
    assert completed["result_type"] == "ToolMessage"
    assert completed["content_chars"] == 2
    assert completed["content_preview"] == '"ok"'
    assert isinstance(completed["duration_ms"], int)


@pytest.mark.asyncio
async def test_result_metrics_hoist_scalars_beyond_truncated_preview():
    """grep 回执里排在 matches 之后的 total_matches/has_more/next_offset 常被预览截断;
    必须提取到 result_metrics 顶层才可审计。"""
    events: list[tuple[str, dict[str, object]]] = []
    middleware = AgentEventsMiddleware(
        "Researcher",
        event_slug="researcher",
        run_limit=9,
        emit=lambda event_type, payload: events.append((event_type, payload)),
    )
    big = {
        "status": "completed",
        "document_id": "doc-1",
        "pattern_totals": {"CUDA": 42},
        "segments": [
            {
                "offset": 0,
                "requested": 19,
                "returned": 19,
                "matches": [
                    {
                        "hit_patterns": ["CUDA"],
                        "start_line": i,
                        "end_line": i + 4,
                        "content": "x" * 200,
                    }
                    for i in range(1, 20)
                ],
            }
        ],
        "total_windows": 42,
        "has_more": True,
    }
    content = "【系统工具执行结果；不是用户补充】\n" + json.dumps(big, ensure_ascii=False)

    async def handler(request):
        return ToolMessage(content=content, tool_call_id="call-1")

    await middleware.awrap_tool_call(_request(), handler)
    completed = events[-1][1]
    assert len(completed["content_preview"]) == 2000
    assert '"total_windows"' not in completed["content_preview"]
    metrics = completed["result_metrics"]
    assert metrics["total_windows"] == 42
    assert metrics["has_more"] is True
    assert metrics["status"] == "completed"


def test_result_metrics_numeric_only_drops_strings():
    """读取类工具正文被 redact 时仍提取计数,但只放行标量数字、不漏字符串。"""
    payload = json.dumps(
        {
            "status": "completed",
            "query": "CUDA",
            "total_matches": 42,
            "has_more": True,
            "next_offset": 19,
            "matches": [{"content": "secret原文" * 30}],
            "note": "z" * 60,
        },
        ensure_ascii=False,
    )
    numeric = AgentEventsMiddleware._result_metrics(payload, numeric_only=True)
    assert numeric == {"total_matches": 42, "has_more": True, "next_offset": 19}
    full = AgentEventsMiddleware._result_metrics(payload, numeric_only=False)
    assert full["status"] == "completed" and full["query"] == "CUDA"
    assert "matches" not in full and "note" not in full


@pytest.mark.asyncio
async def test_tool_failure_is_recorded_and_reraised():
    events: list[tuple[str, dict[str, object]]] = []
    middleware = AgentEventsMiddleware(
        "Researcher",
        event_slug="researcher",
        run_limit=5,
        emit=lambda event_type, payload: events.append((event_type, payload)),
    )

    async def handler(request):
        del request
        raise ValueError("bad input")

    with pytest.raises(ValueError, match="bad input"):
        await middleware.awrap_tool_call(_request(), handler)

    assert [event_type for event_type, _ in events] == [
        "researcher_tool_started",
        "researcher_tool_failed",
    ]
    failure = events[1][1]
    assert failure["error_type"] == "ValueError"
    assert failure["error_preview"] == "bad input"
    assert failure["cancelled"] is False


@pytest.mark.asyncio
async def test_document_tool_content_is_redacted_from_lifecycle_events():
    events: list[tuple[str, dict[str, object]]] = []
    middleware = AgentEventsMiddleware(
        "Researcher",
        event_slug="researcher",
        run_limit=5,
        emit=lambda event_type, payload: events.append((event_type, payload)),
    )
    request = _request()
    request.tool_call = {
        "name": "AddEvidence",
        "id": "call-sensitive",
        "args": {"evidences": [{"quote_start": "不应进入事件的原文"}]},
    }

    async def handler(_request):
        return ToolMessage(content="包含原文的工具回执", tool_call_id="call-sensitive")

    await middleware.awrap_tool_call(request, handler)

    started = events[0][1]
    completed = events[1][1]
    assert started["arguments_redacted"] is True
    assert started["arguments_preview"] == ""
    assert completed["content_redacted"] is True
    assert "content_preview" not in completed


@pytest.mark.asyncio
async def test_model_turn_event_carries_wall_duration():
    """回合耗时在 awrap_model_call 探表、aafter_model 合入;未经探表的回合不硬塞键。"""
    from langchain_core.messages import AIMessage

    events: list[tuple[str, dict[str, object]]] = []
    middleware = AgentEventsMiddleware(
        "Researcher",
        event_slug="researcher",
        run_limit=5,
        emit=lambda event_type, payload: events.append((event_type, payload)),
    )
    reply = AIMessage(content="思考文本", id="msg-1")
    request = SimpleNamespace(
        state={"run_model_call_count": 1},
        runtime=SimpleNamespace(
            context=SimpleNamespace(
                scope=AgentScope(run_id="run-1", agent_name="Researcher", task_id="task-1")
            )
        ),
    )

    async def handler(_request):
        return reply

    assert await middleware.awrap_model_call(request, handler) is reply
    await middleware.aafter_model({"messages": [reply]}, request.runtime)
    event_type, payload = events[0]
    assert event_type == "researcher_model_turn"
    assert isinstance(payload["duration_ms"], int)

    await middleware.aafter_model(
        {"messages": [AIMessage(content="另一回合", id="msg-2")]}, request.runtime
    )
    assert "duration_ms" not in events[1][1]


@pytest.mark.asyncio
async def test_sink_failure_does_not_block_host():
    """旁路保护在 emit_agent_event 一层:sink 落盘异常被吞掉并转入错误日志。"""
    import logging

    from deepresearcher.observability.events.emit import emit_agent_event

    class ExplodingSink:
        def write(self, _record):
            raise OSError("sink down")

    emit_agent_event(
        ExplodingSink(), logging.getLogger("test"), "evt", {"a": 1}, component="c"
    )  # 调用不抛异常即说明保护生效在本层


def test_context_token_count_includes_tool_call_arguments():
    """工具参数常带草稿与逐字引用,是最重的载荷;只数 content 会让压缩条件无法触发。"""
    from langchain_core.messages import AIMessage

    from deepresearcher.agents.middleware.factory import count_message_tokens

    plain = AIMessage(content="短文本")
    with_draft = AIMessage(
        content="短文本",
        tool_calls=[
            {"name": "ReportComplete", "args": {"markdown": "长文内容" * 1000}, "id": "call-1"}
        ],
    )
    assert count_message_tokens([with_draft]) > count_message_tokens([plain]) * 10


def test_limit_marker_and_receipt_prefix_are_library_string_contracts():
    """跨层字符串契约校验:上游改文案时本用例必须失败,不允许静默放过。"""
    from langchain.agents.middleware.model_call_limit import _build_limit_exceeded_message

    from deepresearcher.agents.middleware.agent_events import LIMIT_MESSAGE_MARKER
    from deepresearcher.schemas import format_tool_receipt

    message = _build_limit_exceeded_message(1, 9, None, 8)
    assert LIMIT_MESSAGE_MARKER in message
    # observability 以 startswith 判定回执归属，再剥前缀提取指标
    receipt = format_tool_receipt({"evidence_count": 2})
    assert receipt.startswith("【系统工具执行结果")
    assert receipt.split("\n", 1)[-1] == '{"evidence_count": 2}'


@pytest.mark.asyncio
async def test_observable_summarization_emits_compaction_metrics():
    """压缩事件在整表重建时携带前后 token 数,供 clear_tool_inputs 决策积累数据。"""
    from langchain_core.messages import HumanMessage, RemoveMessage
    from langgraph.graph.message import REMOVE_ALL_MESSAGES

    from deepresearcher.agents.middleware import factory as factory_module
    from deepresearcher.agents.middleware.factory import ObservableSummarizationMiddleware

    events: list[tuple[str, dict[str, object]]] = []
    middleware = ObservableSummarizationMiddleware.__new__(ObservableSummarizationMiddleware)
    middleware._agent_name = "Writer"  # noqa: SLF001
    middleware._emit = lambda event_type, payload: events.append((event_type, payload))  # noqa: SLF001
    middleware._trigger_tokens = 4000  # noqa: SLF001

    original = HumanMessage(content="原始对话" * 20)
    summary = HumanMessage(content="摘要")

    async def fake_super(_self, _state, _runtime):
        # 替换真实压缩，只观察本类自身逻辑；真实触发路径由集成测试覆盖。
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), summary]}

    real_super = factory_module.SummarizationMiddleware.abefore_model
    factory_module.SummarizationMiddleware.abefore_model = fake_super  # type: ignore[method-assign]
    try:
        update = await middleware.abefore_model({"messages": [original]}, None)
    finally:
        factory_module.SummarizationMiddleware.abefore_model = real_super  # type: ignore[method-assign]

    assert len(update["messages"]) == 2  # 观测层不改写压缩产物
    assert events, "压缩发生时必须发射 context_compacted"
    event_type, payload = events[0]
    assert event_type == "context_compacted"
    assert payload["agent"] == "Writer"
    assert payload["before_tokens"] > payload["after_tokens"] > 0
    assert payload["trigger_tokens"] == 4000
