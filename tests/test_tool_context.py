from deepsearch_agent.tools.context import ToolExecutionContext


def test_tool_execution_context_preserves_stable_identity_fields():
    context = ToolExecutionContext.from_task(
        {
            "id": "task-0007",
            "run_id": "run-abc",
            "question": "测试",
            "type": "search",
            "status": "pending",
            "assigned_agent": "research_agent",
            "operation_id": "search-0003",
            "parent_task_id": "task-0001",
        }
    )

    assert context.run_id == "run-abc"
    assert context.task_id == "task-0007"
    assert context.operation_id == "search-0003"
    assert context.event_fields() == {
        "run_id": "run-abc",
        "task_id": "task-0007",
        "operation_id": "search-0003",
        "parent_task_id": "task-0001",
    }


def test_tool_execution_context_keeps_legacy_task_fallback():
    context = ToolExecutionContext.from_task(
        {
            "id": "task-0001",
            "question": "测试",
            "type": "search",
            "status": "pending",
            "assigned_agent": "research_agent",
        }
    )

    assert context.run_id == ""
    assert context.operation_id is None
    assert context.event_fields()["operation_id"] == "task-0001"
