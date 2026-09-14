import pytest

from deepresearcher.config import SearchConfig, _search_config, get_settings


def test_settings_are_cached_and_grouped():
    first = get_settings()
    second = get_settings()
    assert first is second
    assert first.agent.max_research_rounds >= 1
    assert isinstance(first.llm.model, str)
    assert first.agent.evidence_full_context_max_tokens >= 1_000
    assert first.agent.evidence_bm25_top_k >= 1
    assert first.agent.evidence_bm25_window >= 0
    assert first.agent.evidence_retriever_backend == "bm25"
    assert first.tool_cache.extractor_prompt_version == "evidence-prompt-v2"
    assert first.tool_cache.chunking_version == "chunks-v2"


def test_output_language_default_and_directive():
    from deepresearcher.config import AgentConfig, language_directive

    assert AgentConfig().output_language == "中文"
    directive = language_directive("English")
    assert "English" in directive and "quote" in directive


def test_fetch_provider_order_preserves_configured_order(monkeypatch):
    monkeypatch.setenv("FETCH_PROVIDER_ORDER", "aliyun, direct")

    assert _search_config().fetch_provider_order == ("aliyun", "direct")


@pytest.mark.parametrize("value", ["aliyun,unknown", "", "direct,"])
def test_fetch_provider_order_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("FETCH_PROVIDER_ORDER", value)

    with pytest.raises(ValueError, match="FETCH_PROVIDER_ORDER"):
        _search_config()


def test_explicit_search_provider_requires_matching_key():
    assert not SearchConfig(provider="tavily", baidu_api_key="wrong-provider-key").configured
    assert SearchConfig(provider="tavily", tavily_api_key="configured").configured
    assert SearchConfig(provider="aliyun").configured
