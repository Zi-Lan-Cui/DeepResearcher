from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage

from deepresearcher.agents.middleware import AgentObservabilityMiddleware
from deepresearcher.context.execution import AgentExecutionScope


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
