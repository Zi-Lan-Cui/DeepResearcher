from datetime import datetime, timezone

from deepsearch_agent.context.runtime import get_runtime_environment


def test_runtime_environment_exposes_dynamic_facts_without_message_policy():
    environment = get_runtime_environment(
        now=datetime(2026, 9, 10, 8, 30, tzinfo=timezone.utc),
    )

    assert environment.payload() == {
        "current_date": "2026-09-10",
        "timezone": "UTC",
    }
