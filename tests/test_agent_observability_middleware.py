import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage

from deepresearcher.agents.middleware import AgentObservabilityMiddleware
from deepresearcher.agents.researcher.state import (
    ResearcherDeps,
    ResearcherLoopContext,
    ResearcherLoopState,
)
from deepresearcher.config import AgentConfig
from deepresearcher.observability.execution import AgentExecutionScope


def _real_researcher_context() -> ResearcherLoopContext:
    """生产形状的真实 ResearcherLoopContext:中间件契约字段(scope/tool_gate)必须在场。

    上方 SimpleNamespace 假件测的是中间件"如何对待有 scope 的对象";本函数测
    "真 context 是否提供中间件所需字段"——两个方向都有守卫,才不会再出现
    删字段后 CI 全绿、事件静默失去归因的漂移。
    """
    task = {
        "id": "task-0001",
        "run_id": "run-1",
        "question": "Redis 恢复",
        "type": "search",
        "status": "pending",
        "assigned_agent": "research_agent",
    }
    return ResearcherLoopContext(
        deps=ResearcherDeps(
            config=AgentConfig(),
            search_tool=object(),  # 中间件不触碰业务零件
            reader_tool=object(),
            material_store=None,
            emit=lambda *_: None,
        ),
        scope=AgentExecutionScope.from_task(task, agent_name="ResearchAgent"),
        task=task,  # type: ignore[typeddict-item]
        loop_state=ResearcherLoopState(),
    )


@pytest.mark.asyncio
async def test_real_researcher_context_carries_middleware_attribution_fields():
    events: list[tuple[str, dict[str, object]]] = []
    middleware = AgentObservabilityMiddleware(
        "ResearchAgent",
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
    # serial_tools 的栅栏读者同样在场(缺席则该工具失去与 Complete 的互斥)。
    assert getattr(request.runtime.context, "tool_gate", None) is not None


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
                scope=AgentExecutionScope(
                    run_id="run-1",
                    agent_name="ResearchAgent",
                    task_id="task-1",
                )
            )
        ),
    )


@pytest.mark.asyncio
async def test_tool_lifecycle_records_scope_and_preserves_result():
    events: list[tuple[str, dict[str, object]]] = []
    middleware = AgentObservabilityMiddleware(
        "ResearchAgent",
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
        "researchagent_tool_started",
        "researchagent_tool_completed",
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
    必须 hoist 到 result_metrics 顶层才可审计。"""
    events: list[tuple[str, dict[str, object]]] = []
    middleware = AgentObservabilityMiddleware(
        "ResearchAgent",
        run_limit=9,
        emit=lambda event_type, payload: events.append((event_type, payload)),
    )
    big = {
        "status": "completed",
        "document_id": "doc-1",
        "query": "CUDA",
        "matches": [
            {"query": "CUDA", "start_line": i, "end_line": i + 4, "content": "x" * 200}
            for i in range(1, 20)
        ],
        "match_count": 19,
        "total_matches": 42,
        "has_more": True,
        "next_offset": 19,
    }
    content = "【系统工具执行结果；不是用户补充】\n" + json.dumps(big, ensure_ascii=False)

    async def handler(request):
        return ToolMessage(content=content, tool_call_id="call-1")

    await middleware.awrap_tool_call(_request(), handler)
    completed = events[-1][1]
    assert len(completed["content_preview"]) == 2000  # 预览被截断(工具结果预览上限)
    assert '"total_matches"' not in completed["content_preview"]  # 预览里根本看不到
    metrics = completed["result_metrics"]
    assert metrics["total_matches"] == 42
    assert metrics["has_more"] is True
    assert metrics["next_offset"] == 19
    assert metrics["match_count"] == 19
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
    numeric = AgentObservabilityMiddleware._result_metrics(payload, numeric_only=True)
    assert numeric == {"total_matches": 42, "has_more": True, "next_offset": 19}
    full = AgentObservabilityMiddleware._result_metrics(payload, numeric_only=False)
    assert full["status"] == "completed" and full["query"] == "CUDA"
    assert "matches" not in full and "note" not in full  # 大数组/长串不搬


@pytest.mark.asyncio
async def test_tool_failure_is_recorded_and_reraised():
    events: list[tuple[str, dict[str, object]]] = []
    middleware = AgentObservabilityMiddleware(
        "ResearchAgent",
        run_limit=5,
        emit=lambda event_type, payload: events.append((event_type, payload)),
    )

    async def handler(request):
        del request
        raise ValueError("bad input")

    with pytest.raises(ValueError, match="bad input"):
        await middleware.awrap_tool_call(_request(), handler)

    assert [event_type for event_type, _ in events] == [
        "researchagent_tool_started",
        "researchagent_tool_failed",
    ]
    failure = events[1][1]
    assert failure["error_type"] == "ValueError"
    assert failure["error_preview"] == "bad input"
    assert failure["cancelled"] is False


@pytest.mark.asyncio
async def test_document_tool_content_is_redacted_from_lifecycle_events():
    events: list[tuple[str, dict[str, object]]] = []
    middleware = AgentObservabilityMiddleware(
        "ResearchAgent",
        run_limit=5,
        emit=lambda event_type, payload: events.append((event_type, payload)),
    )
    request = _request()
    request.tool_call = {
        "name": "AddEvidence",
        "id": "call-sensitive",
        "args": {"evidences": [{"quote": "不应进入事件的原文"}]},
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
async def test_observability_sink_failure_does_not_block_tool():
    middleware = AgentObservabilityMiddleware(
        "ResearchAgent",
        run_limit=5,
        emit=lambda _event_type, _payload: (_ for _ in ()).throw(OSError("sink down")),
    )
    expected = ToolMessage(content="ok", tool_call_id="call-1")

    async def handler(request):
        del request
        return expected

    assert await middleware.awrap_tool_call(_request(), handler) is expected


def test_context_token_count_includes_tool_call_arguments():
    """工具参数里住着最重的载荷(草稿/逐字引用);只数 content 会让压缩闸失明。"""
    from langchain_core.messages import AIMessage

    from deepresearcher.agents.middleware.factory import count_message_tokens

    plain = AIMessage(content="短文本")
    with_draft = AIMessage(
        content="短文本",
        tool_calls=[
            {"name": "CompleteReport", "args": {"markdown": "长文内容" * 1000}, "id": "call-1"}
        ],
    )
    assert count_message_tokens([with_draft]) > count_message_tokens([plain]) * 10
