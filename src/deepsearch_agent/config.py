"""统一配置入口。

只有本模块负责读取环境变量和 ``env/.env``。业务模块通过
``get_settings()`` 获取只读配置，或在测试/依赖注入时显式传入 ``Settings``。
"""

from dataclasses import dataclass
from functools import lru_cache
from os import getenv
from pathlib import Path
from typing import Literal, cast

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
WriterSupportLevel = Literal["insufficient", "partial", "direct"]


def _env(name: str, default: str = "") -> str:
    return getenv(name, default).strip()


def _int_env(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _choice_env(name: str, default: str, choices: set[str]) -> str:
    value = _env(name, default).lower()
    return value if value in choices else default


def _bool_env(name: str, default: bool) -> bool:
    raw = _env(name, "true" if default else "false").lower()
    if raw in {"1", "true", "yes"}:
        return True
    if raw in {"0", "false", "no"}:
        return False
    return default


@dataclass(frozen=True)
class LLMRetryConfig:
    """LLM 调用策略；传输重试和结构化修复是两种不同的预算。"""

    transport_attempts: int = 3
    initial_seconds: float = 1.0
    max_seconds: float = 20.0
    exp_base: float = 2.0
    jitter: float = 1.0
    structured_repair_attempts: int = 1


@dataclass(frozen=True)
class LLMConfig:
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    temperature: float = 0.0
    timeout: float = 60.0
    context_window_tokens: int = 32_768
    retry: LLMRetryConfig = LLMRetryConfig()

    @property
    def configured(self) -> bool:
        return all((self.api_key, self.base_url, self.model))


@dataclass(frozen=True)
class AgentConfig:
    # Supervisor 的研究轮次上限；一轮包含一次评估、派发和汇总。
    max_research_rounds: int = 3
    # Reflection 判定 fatal 后，Supervisor 允许组织的修复循环次数。
    max_post_review_recovery_cycles: int = 1
    # Writer 的 Agent turn 上限；每个 turn 是一次模型决策及其工具执行。
    writer_max_turns: int = 8
    writer_feedback_chars: int = 800
    writer_minimum_support: WriterSupportLevel = "direct"
    writer_max_selected_evidence: int = 24
    writer_max_markdown_chars: int = 24_000
    # 研究未完全覆盖时，达到该最低材料门槛仍允许 Writer 产出部分报告。
    partial_report_min_evidences: int = 1
    partial_report_min_sources: int = 1
    # 单次 ReadEvidence 请求的最大返回数量；不限制 Writer 在整个运行中的读取总量。
    writer_read_batch_size: int = 15
    max_subtasks_per_round: int = 12
    max_parallel_workers: int = 3
    supervisor_preview_chars: int = 300
    # Supervisor 交给 Writer 的报告任务书输出限制。
    report_max_topics: int = 6
    report_max_caveats: int = 6
    # 单一网页可贡献的 Evidence 上限，避免某篇来源主导整个研究方向。
    evidence_max_per_source: int = 2
    evidence_input_budget_tokens: int = 24_000
    evidence_output_budget_tokens: int = 4_000
    evidence_safety_margin_tokens: int = 2_000
    evidence_chunk_concurrency: int = 2
    # ResearchAgent 的 Agent turn 上限；每个 turn 是一次模型决策及其工具执行。
    research_agent_max_turns: int = 3
    research_agent_max_queries: int = 6
    # 一个方向级 ResearchAgent 最多带回的 Evidence 数；限制上下文与成本，
    # 但不等同于单网页抽取上限。
    research_agent_max_evidences_per_direction: int = 6
    research_agent_read_concurrency: int = 3
    # 来源处理的分层 deadline；总时限必须大于各阶段的正常预算。
    source_fetch_timeout: float = 30.0
    source_parse_timeout: float = 20.0
    evidence_extract_timeout: float = 120.0
    source_total_timeout: float = 180.0
    research_query_chars: int = 180
    research_observation_quote_chars: int = 500
    research_failure_history_limit: int = 5


@dataclass(frozen=True)
class SearchConfig:
    baidu_api_key: str = ""
    tavily_api_key: str = ""
    serpapi_api_key: str = ""
    provider: Literal["auto", "baidu", "tavily", "serpapi"] = "auto"
    timeout: float = 20.0
    max_results: int = 5
    retry_attempts: int = 3
    retry_initial_seconds: float = 1.0
    retry_max_seconds: float = 10.0
    retry_after_max_seconds: float = 60.0
    # 同一供应商同时允许的搜索请求数（跨 worker 全局）；限流窗口下降低并发比快速重试更有效。
    max_concurrent_requests: int = 2
    tavily_include_raw_content: bool = True

    @property
    def configured(self) -> bool:
        return bool(self.baidu_api_key or self.tavily_api_key or self.serpapi_api_key)


@dataclass(frozen=True)
class AppConfig:
    environment: str = "development"
    log_level: str = "INFO"


@dataclass(frozen=True)
class ObservabilityConfig:
    log_dir: Path
    log_file: str = "agent.log"
    event_file: str = "events.jsonl"
    trace_file: str = "traces.jsonl"
    max_text_chars: int = 1_000


@dataclass(frozen=True)
class Settings:
    llm: LLMConfig
    agent: AgentConfig
    search: SearchConfig
    app: AppConfig
    observability: ObservabilityConfig


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """加载并缓存全局配置；进程内只读取一次。"""
    env_file = Path(_env("DEEPSEARCH_ENV_FILE", str(_PROJECT_ROOT / "env" / ".env")))
    load_dotenv(env_file, override=False)
    return Settings(
        llm=_llm_config(),
        agent=_agent_config(),
        search=_search_config(),
        app=_app_config(),
        observability=_observability_config(),
    )


def _llm_config() -> LLMConfig:
    return LLMConfig(
        api_key=_env("LLM_API_KEY"),
        base_url=_env("LLM_BASE_URL"),
        model=_env("LLM_MODEL_ID"),
        temperature=_float_env("LLM_TEMPERATURE", 0.0),
        timeout=_float_env("LLM_TIMEOUT", 60.0),
        context_window_tokens=max(4_096, _int_env("LLM_CONTEXT_WINDOW_TOKENS", 32_768)),
        retry=LLMRetryConfig(
            transport_attempts=max(1, _int_env("LLM_RETRY_ATTEMPTS", 3)),
            initial_seconds=max(0.0, _float_env("LLM_RETRY_INITIAL_SECONDS", 1.0)),
            max_seconds=max(0.0, _float_env("LLM_RETRY_MAX_SECONDS", 20.0)),
            exp_base=max(1.0, _float_env("LLM_RETRY_EXP_BASE", 2.0)),
            jitter=max(0.0, _float_env("LLM_RETRY_JITTER", 1.0)),
            structured_repair_attempts=max(0, _int_env("LLM_STRUCTURED_REPAIR_ATTEMPTS", 1)),
        ),
    )


def _agent_config() -> AgentConfig:
    return AgentConfig(
        max_research_rounds=max(1, _int_env("AGENT_MAX_RESEARCH_ROUNDS", 3)),
        max_post_review_recovery_cycles=max(
            0, _int_env("AGENT_MAX_POST_REVIEW_RECOVERY_CYCLES", 1)
        ),
        writer_max_turns=max(1, _int_env("AGENT_WRITER_MAX_TURNS", 8)),
        writer_feedback_chars=max(100, _int_env("AGENT_WRITER_FEEDBACK_CHARS", 800)),
        writer_minimum_support=cast(
            WriterSupportLevel,
            _choice_env(
                "AGENT_WRITER_MINIMUM_SUPPORT",
                "direct",
                {"insufficient", "partial", "direct"},
            ),
        ),
        writer_max_selected_evidence=max(
            1, _int_env("AGENT_WRITER_MAX_SELECTED_EVIDENCE", 24)
        ),
        writer_max_markdown_chars=max(
            1_000, _int_env("AGENT_WRITER_MAX_MARKDOWN_CHARS", 24_000)
        ),
        partial_report_min_evidences=max(
            1, _int_env("AGENT_PARTIAL_REPORT_MIN_EVIDENCES", 1)
        ),
        partial_report_min_sources=max(1, _int_env("AGENT_PARTIAL_REPORT_MIN_SOURCES", 1)),
        writer_read_batch_size=max(1, _int_env("AGENT_WRITER_READ_BATCH_SIZE", 15)),
        max_subtasks_per_round=max(1, _int_env("AGENT_MAX_SUBTASKS_PER_ROUND", 12)),
        max_parallel_workers=max(1, _int_env("AGENT_MAX_PARALLEL_WORKERS", 3)),
        supervisor_preview_chars=max(100, _int_env("AGENT_SUPERVISOR_PREVIEW_CHARS", 300)),
        report_max_topics=max(1, _int_env("AGENT_REPORT_MAX_TOPICS", 6)),
        report_max_caveats=max(1, _int_env("AGENT_REPORT_MAX_CAVEATS", 6)),
        evidence_max_per_source=max(1, _int_env("AGENT_EVIDENCE_MAX_PER_SOURCE", 2)),
        evidence_input_budget_tokens=max(
            1_000, _int_env("AGENT_EVIDENCE_INPUT_BUDGET_TOKENS", 24_000)
        ),
        evidence_output_budget_tokens=max(
            500, _int_env("AGENT_EVIDENCE_OUTPUT_BUDGET_TOKENS", 4_000)
        ),
        evidence_safety_margin_tokens=max(
            0, _int_env("AGENT_EVIDENCE_SAFETY_MARGIN_TOKENS", 2_000)
        ),
        evidence_chunk_concurrency=max(1, _int_env("AGENT_EVIDENCE_CHUNK_CONCURRENCY", 2)),
        research_agent_max_turns=max(1, _int_env("AGENT_RESEARCH_MAX_TURNS", 3)),
        research_agent_max_queries=max(1, _int_env("AGENT_RESEARCH_MAX_QUERIES", 6)),
        research_agent_max_evidences_per_direction=max(
            1, _int_env("AGENT_RESEARCH_MAX_EVIDENCES_PER_DIRECTION", 6)
        ),
        research_agent_read_concurrency=max(1, _int_env("AGENT_RESEARCH_READ_CONCURRENCY", 3)),
        source_fetch_timeout=max(1.0, _float_env("AGENT_SOURCE_FETCH_TIMEOUT", 30.0)),
        source_parse_timeout=max(1.0, _float_env("AGENT_SOURCE_PARSE_TIMEOUT", 20.0)),
        evidence_extract_timeout=max(1.0, _float_env("AGENT_EVIDENCE_EXTRACT_TIMEOUT", 120.0)),
        source_total_timeout=max(1.0, _float_env("AGENT_SOURCE_TOTAL_TIMEOUT", 180.0)),
        research_query_chars=max(40, _int_env("AGENT_RESEARCH_QUERY_CHARS", 180)),
        research_observation_quote_chars=max(
            100, _int_env("AGENT_RESEARCH_OBSERVATION_QUOTE_CHARS", 500)
        ),
        research_failure_history_limit=max(1, _int_env("AGENT_RESEARCH_FAILURE_HISTORY_LIMIT", 5)),
    )


def _search_config() -> SearchConfig:
    return SearchConfig(
        provider=cast(
            Literal["auto", "baidu", "tavily", "serpapi"],
            _choice_env("SEARCH_PROVIDER", "auto", {"auto", "baidu", "tavily", "serpapi"}),
        ),
        baidu_api_key=_env("BAIDU_API_KEY"),
        tavily_api_key=_env("TAVILY_API_KEY"),
        serpapi_api_key=_env("SERPAPI_API_KEY"),
        timeout=_float_env("SEARCH_TIMEOUT", 20.0),
        max_results=max(1, _int_env("SEARCH_MAX_RESULTS", 5)),
        retry_attempts=max(1, _int_env("SEARCH_RETRY_ATTEMPTS", 3)),
        retry_initial_seconds=max(0.0, _float_env("SEARCH_RETRY_INITIAL_SECONDS", 1.0)),
        retry_max_seconds=max(0.0, _float_env("SEARCH_RETRY_MAX_SECONDS", 10.0)),
        retry_after_max_seconds=max(0.0, _float_env("SEARCH_RETRY_AFTER_MAX_SECONDS", 60.0)),
        max_concurrent_requests=max(1, _int_env("SEARCH_MAX_CONCURRENT_REQUESTS", 2)),
        tavily_include_raw_content=_bool_env("TAVILY_INCLUDE_RAW_CONTENT", True),
    )


def _app_config() -> AppConfig:
    return AppConfig(
        environment=_env("APP_ENV", "development"),
        log_level=_env("LOG_LEVEL", "INFO").upper(),
    )


def _observability_config() -> ObservabilityConfig:
    return ObservabilityConfig(
        log_dir=Path(_env("OBSERVABILITY_LOG_DIR", str(_PROJECT_ROOT / "var" / "logs"))),
        log_file=_env("OBSERVABILITY_LOG_FILE", "agent.log"),
        event_file=_env("OBSERVABILITY_EVENT_FILE", "events.jsonl"),
        trace_file=_env("OBSERVABILITY_TRACE_FILE", "traces.jsonl"),
        max_text_chars=max(100, _int_env("OBSERVABILITY_MAX_TEXT_CHARS", 1_000)),
    )


def clear_settings_cache() -> None:
    """仅供测试或显式热加载配置使用。"""
    get_settings.cache_clear()
