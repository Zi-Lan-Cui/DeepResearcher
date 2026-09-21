from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from deepresearcher.agents.middleware import ToolLoopGuardMiddleware


@pytest.mark.asyncio
async def test_submission_guard_injects_dynamic_final_turn_reminder():
    events: list[tuple[str, dict[str, object]]] = []
    middleware = ToolLoopGuardMiddleware(
        agent_name="ResearchAgent",
        event_slug="researcher",
        nudge_message="请提交",
        submitted_probe=lambda _context: False,
        run_limit=6,
        reminder_turns=2,
        reminder_message="仅剩 {remaining_turns} 回合，请提交 Evidence。",
        emit=lambda event_type, payload: events.append((event_type, payload)),
    )
    runtime = SimpleNamespace(context=SimpleNamespace())

    assert await middleware.abefore_model({"run_model_call_count": 3}, runtime) is None
    update = await middleware.abefore_model({"run_model_call_count": 4}, runtime)

    assert update is not None
    assert update["messages"] == [HumanMessage(content="仅剩 2 回合，请提交 Evidence。")]
    assert events == [
        (
            "researcher_finalization_reminded",
            {"agent": "ResearchAgent", "remaining_turns": 2, "run_limit": 6},
        )
    ]


@pytest.mark.asyncio
async def test_submission_guard_does_not_repeat_same_dynamic_reminder():
    middleware = ToolLoopGuardMiddleware(
        agent_name="ResearchAgent",
        event_slug="researcher",
        nudge_message="请提交",
        submitted_probe=lambda _context: False,
        run_limit=5,
        reminder_turns=2,
        reminder_message="仅剩 {remaining_turns} 回合。",
    )
    reminder = HumanMessage(content="仅剩 2 回合。")

    update = await middleware.abefore_model(
        {"run_model_call_count": 3, "messages": [reminder]},
        SimpleNamespace(context=SimpleNamespace()),
    )

    assert update is None


@pytest.mark.asyncio
async def test_nudge_budget_counts_from_private_state_channel_not_message_history():
    """压缩会整表重建历史;历史匹配的计数会被静默清零、突破 max_nudges。"""
    from langchain_core.messages import AIMessage

    middleware = ToolLoopGuardMiddleware(
        agent_name="Writer",
        nudge_message="请提交",
        submitted_probe=lambda _context: False,
        max_nudges=2,
    )
    runtime = SimpleNamespace(context=SimpleNamespace())

    # 通道说已踢满:哪怕历史里一条 nudge 都没有,也绝不续命。
    state = {"messages": [AIMessage(content="草稿直接写正文")], "nudge_count": 2}
    assert await middleware.aafter_model(state, runtime) is None

    update = await middleware.aafter_model({"messages": [AIMessage(content="x")]}, runtime)
    assert update is not None
    assert update["nudge_count"] == 1
    assert update["jump_to"] == "model"


@pytest.mark.asyncio
async def test_softened_model_failure_text_is_not_nudged_as_protocol_violation():
    from langchain_core.messages import AIMessage

    from deepresearcher.agents.middleware.retry import MODEL_FAILURE_MARKER

    middleware = ToolLoopGuardMiddleware(
        agent_name="Writer",
        nudge_message="请提交",
        submitted_probe=lambda _context: False,
        max_nudges=2,
    )
    softened = AIMessage(
        content=(
            f"Writer {MODEL_FAILURE_MARKER}：APIConnectionError。"
            "请基于当前上下文调整下一步行动；不要重复提交相同的无效调用。"
        )
    )
    assert (
        await middleware.aafter_model(
            {"messages": [softened]}, SimpleNamespace(context=SimpleNamespace())
        )
        is None
    )
