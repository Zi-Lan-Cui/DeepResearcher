from deepsearch_agent.agents.runtime import AgentExecutionScope


def test_execution_scope_builds_stable_task_identity():
    scope = AgentExecutionScope.from_task(
        {
            "id": "task-0001",
            "run_id": "run-0001",
            "parent_task_id": "parent-0001",
            "operation_id": "operation-0001",
        },
        agent_name="ResearchAgent",
    )

    assert scope.event_fields() == {
        "run_id": "run-0001",
        "agent": "ResearchAgent",
        "task_id": "task-0001",
        "operation_id": "operation-0001",
        "parent_task_id": "parent-0001",
    }


def test_execution_scope_falls_back_to_task_id_for_operation():
    scope = AgentExecutionScope.from_task(
        {"id": "task-0001", "run_id": "run-0001"},
        agent_name="ResearchAgent",
    )

    assert scope.event_fields()["operation_id"] == "task-0001"
