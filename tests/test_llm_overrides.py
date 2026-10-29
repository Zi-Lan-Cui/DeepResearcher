"""按 agent 覆盖模型请求参数的入口。"""

import pytest
from langchain_openai import ChatOpenAI

from deepresearcher.config import (
    AgentConfig,
    AppConfig,
    LLMConfig,
    ObservabilityConfig,
    SearchConfig,
    Settings,
    _request_overrides_env,
    get_settings,
)
from deepresearcher.graph import build_graph
from deepresearcher.llm import LLMConfigurationError, build_llm, derive_llm


def _llm(**kwargs) -> ChatOpenAI:
    return ChatOpenAI(model="m", api_key="k", base_url="http://127.0.0.1:1", **kwargs)


def test_derive_llm_returns_the_baseline_without_overrides():
    llm = _llm()
    assert derive_llm(llm, None) is llm
    assert derive_llm(llm, {}) is llm


def test_derive_llm_puts_extra_body_into_the_request_payload():
    llm = _llm()
    derived = derive_llm(llm, {"extra_body": {"thinking": {"type": "disabled"}}})
    # 带上 tools 形参走一遍：create_agent 会在 bind_tools 之后才发请求，
    # 覆盖项在那条链路上仍要在 payload 里。
    payload = derived._get_request_payload("hi", tools=[])
    assert payload["extra_body"] == {"thinking": {"type": "disabled"}}
    assert llm._get_request_payload("hi").get("extra_body") is None


def test_derive_llm_merges_extra_body_from_the_constructor():
    llm = _llm(extra_body={"enable_thinking": False})
    derived = derive_llm(llm, {"extra_body": {"thinking": {"type": "disabled"}}})
    assert derived._get_request_payload("hi")["extra_body"] == {
        "enable_thinking": False,
        "thinking": {"type": "disabled"},
    }


def test_derive_llm_rejects_unknown_request_field():
    # 拼错的键必须当场报错：静默忽略会让人以为开关生效了。
    with pytest.raises(LLMConfigurationError, match="enable_think"):
        derive_llm(_llm(), {"enable_think": False})


def test_request_overrides_env_reads_per_agent_json(monkeypatch):
    monkeypatch.setenv(
        "LLM_REQUEST_OVERRIDES",
        '{"researcher": {"extra_body": {"thinking": {"type": "disabled"}}},'
        ' "reviewer": {"temperature": 0.2}}',
    )
    assert _request_overrides_env() == {
        "researcher": {"extra_body": {"thinking": {"type": "disabled"}}},
        "reviewer": {"temperature": 0.2},
    }


def test_request_overrides_env_is_empty_when_unset(monkeypatch):
    monkeypatch.delenv("LLM_REQUEST_OVERRIDES", raising=False)
    assert _request_overrides_env() == {}


@pytest.mark.parametrize("raw", ["not json", '["researcher"]', '{"researcher": 1}'])
def test_request_overrides_env_rejects_bad_shape(monkeypatch, raw):
    monkeypatch.setenv("LLM_REQUEST_OVERRIDES", raw)
    with pytest.raises(ValueError, match="LLM_REQUEST_OVERRIDES"):
        _request_overrides_env()


def test_build_llm_applies_request_overrides(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("LLM_MODEL_ID", "m")
    get_settings.cache_clear()
    try:
        llm = build_llm(request_overrides={"extra_body": {"thinking": {"type": "disabled"}}})
        assert llm._get_request_payload("hi")["extra_body"] == {"thinking": {"type": "disabled"}}
    finally:
        get_settings.cache_clear()


def test_build_graph_rejects_unknown_override_actor(tmp_path):
    # 角色名写错不该静默落空：跑完一轮评测才发现开关没生效，代价比报错大。
    settings = Settings(
        llm=LLMConfig(request_overrides={"research": {"extra_body": {"thinking": {}}}}),
        agent=AgentConfig(),
        search=SearchConfig(),
        app=AppConfig(),
        observability=ObservabilityConfig(log_dir=tmp_path),
    )
    with pytest.raises(ValueError, match="未知角色：research"):
        build_graph(settings, llm=_llm())
