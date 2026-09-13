from deepresearcher.config import get_settings


def test_settings_are_cached_and_grouped():
    first = get_settings()
    second = get_settings()
    assert first is second
    assert first.agent.max_research_rounds >= 1
    assert isinstance(first.llm.model, str)


def test_output_language_default_and_directive():
    from deepresearcher.config import AgentConfig, language_directive

    assert AgentConfig().output_language == "中文"
    directive = language_directive("English")
    assert "English" in directive and "quote" in directive
