from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from deepresearcher.agents.middleware import ToolLoopGuardMiddleware


@pytest.mark.asyncio
async def test_submission_guard_injects_dynamic_final_turn_reminder():
    events: list[tuple[str, dict[str, object]]] = []
    middleware = ToolLoopGuardMiddleware(
        agent_name="ResearchAgent",
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
            "researchagent_finalization_reminded",
            {"agent": "ResearchAgent", "remaining_turns": 2, "run_limit": 6},
        )
    ]


@pytest.mark.asyncio
async def test_submission_guard_does_not_repeat_same_dynamic_reminder():
    middleware = ToolLoopGuardMiddleware(
        agent_name="ResearchAgent",
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
