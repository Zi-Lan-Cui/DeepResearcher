from deepsearch_agent.config import get_settings


def test_settings_are_cached_and_grouped():
    first = get_settings()
    second = get_settings()
    assert first is second
    assert first.agent.max_research_rounds >= 1
    assert isinstance(first.llm.model, str)
