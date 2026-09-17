import pytest

from deepresearcher.config import AgentConfig, SearchConfig, _search_config, get_settings
from deepresearcher.schemas.tool_args import ReleaseEvidence, RestoreEvidence


def test_settings_are_cached_and_grouped():
    first = get_settings()
    second = get_settings()
    assert first is second
    assert first.agent.max_research_rounds >= 1
    assert isinstance(first.llm.model, str)


def test_output_language_default_and_directive():
    from deepresearcher.config import language_directive

    assert AgentConfig().output_language == "中文"
    directive = language_directive("English")
    assert "English" in directive and "quote" in directive


def test_agent_config_rejects_active_evidence_limit_above_archive_limit():
    with pytest.raises(ValueError, match="MAX_EVIDENCES_PER_DIRECTION"):
        AgentConfig(
            research_agent_max_evidences_per_direction=13,
            research_agent_max_evidence_candidates_per_direction=12,
        )


def test_agent_config_rejects_per_source_limit_above_archive_limit():
    with pytest.raises(ValueError, match="EVIDENCE_MAX_PER_SOURCE"):
        AgentConfig(
            evidence_max_per_source=13,
            research_agent_max_evidence_candidates_per_direction=12,
        )


def test_agent_config_rejects_report_caveats_above_schema_limit():
    with pytest.raises(ValueError, match="AGENT_REPORT_MAX_CAVEATS"):
        AgentConfig(report_max_caveats=51)


def test_agent_config_allows_business_caveat_limit_above_six():
    assert AgentConfig(report_max_caveats=10).report_max_caveats == 10


def test_release_and_restore_accept_model_selected_batch_size():
    evidence_ids = [f"e-{index}" for index in range(32)]

    assert (
        ReleaseEvidence(evidence_ids=evidence_ids, reason="批量释放").evidence_ids == evidence_ids
    )
    assert (
        RestoreEvidence(evidence_ids=evidence_ids, reason="批量恢复").evidence_ids == evidence_ids
    )


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
