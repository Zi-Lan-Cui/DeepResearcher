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
    from deepresearcher.prompts import language_directive

    assert AgentConfig().output_language == "中文"
    directive = language_directive("English")
    assert "English" in directive and "quote" in directive


def test_agent_config_rejects_active_evidence_limit_above_archive_limit():
    with pytest.raises(ValueError, match="MAX_EVIDENCES_PER_TASK"):
        AgentConfig(
            research_agent_max_evidences_per_task=13,
            research_agent_max_evidence_candidates_per_task=12,
        )


def test_agent_config_rejects_per_source_limit_above_archive_limit():
    with pytest.raises(ValueError, match="EVIDENCE_MAX_PER_SOURCE"):
        AgentConfig(
            evidence_max_per_source=13,
            research_agent_max_evidence_candidates_per_task=12,
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


def _clear_search_credentials(monkeypatch):
    # 优先级顺序会改变 auto 的选择结果，本机 .env 里写过它，必须一并清掉。
    for name in (
        "ALIBABA_CLOUD_ACCESS_KEY_ID",
        "ALIBABA_CLOUD_ACCESS_KEY_SECRET",
        "ALIBABA_CLOUD_CREDENTIALS_URI",
        "BAIDU_API_KEY",
        "SERPAPI_API_KEY",
        "DEEPSEEK_API_KEY",
        "SEARCH_PROVIDER",
        "SEARCH_PROVIDER_ORDER",
    ):
        monkeypatch.delenv(name, raising=False)


def test_auto_selects_aliyun_when_credentials_present(monkeypatch):
    _clear_search_credentials(monkeypatch)
    monkeypatch.setenv("ALIBABA_CLOUD_ACCESS_KEY_ID", "ak")
    monkeypatch.setenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET", "sk")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly")

    config = _search_config()

    assert config.aliyun_configured is True
    assert config.search_provider == "aliyun"
    assert config.aliyun_in_use is True


def test_auto_skips_aliyun_without_credentials(monkeypatch):
    _clear_search_credentials(monkeypatch)
    monkeypatch.setenv("TAVILY_API_KEY", "tvly")

    config = _search_config()

    assert config.aliyun_configured is False
    assert config.search_provider == "tavily"


def test_auto_requires_any_provider_credential(monkeypatch):
    _clear_search_credentials(monkeypatch)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    config = _search_config()

    assert config.search_provider == "unconfigured"
    assert config.configured is False


def test_fetch_only_aliyun_still_requires_aliyun_client():
    # 搜索不走阿里云、只把抓取交给它时，装配期照样要建客户端。
    assert SearchConfig(fetch_provider_order=("aliyun",)).aliyun_in_use is True
    assert SearchConfig(tavily_api_key="k").aliyun_in_use is False


def test_fetch_proxy_defaults_to_empty_and_reads_configured_value(monkeypatch):
    monkeypatch.delenv("FETCH_PROXY", raising=False)
    assert _search_config().fetch_proxy == ""

    monkeypatch.setenv("FETCH_PROXY", "http://127.0.0.1:7890")
    assert _search_config().fetch_proxy == "http://127.0.0.1:7890"
