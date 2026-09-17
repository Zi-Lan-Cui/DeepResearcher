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

from deepresearcher.schemas.limits import (
    EVIDENCE_REFERENCES_HARD_LIMIT,
    REPORT_CAVEATS_HARD_LIMIT,
    REPORT_MARKDOWN_HARD_LIMIT_CHARS,
)

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


def _ordered_choices_env(name: str, default: str, choices: set[str]) -> tuple[str, ...]:
    """读取有序、逗号分隔的选项列表；显式拒绝未知值。"""
    values = tuple(item.strip().lower() for item in _env(name, default).split(","))
    invalid = [item or "<empty>" for item in values if item not in choices]
    if invalid:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{name} 包含无效值: {', '.join(invalid)}；可选值: {allowed}")
    return values


def _bool_env(name: str, default: bool) -> bool:
    raw = _env(name, "true" if default else "false").lower()
    if raw in {"1", "true", "yes"}:
        return True
    if raw in {"0", "false", "no"}:
        return False
    return default


def _project_path_env(name: str, default: Path) -> Path:
    """读取路径配置；相对路径统一以项目根目录为基准。"""
    path = Path(_env(name, str(default))).expanduser()
    return path if path.is_absolute() else _PROJECT_ROOT / path


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
    max_concurrent_requests: int = 6
    provider_requests_per_minute: int = 0
    provider_tokens_per_minute: int = 0
    provider_output_token_reserve: int = 1_024
    max_tokens_per_run: int = 0
    max_cost_usd_per_run: float = 0.0
    max_cost_usd_per_user_daily: float = 0.0
    max_cost_usd_platform_hourly: float = 0.0
    max_cost_usd_platform_daily: float = 0.0
    input_usd_per_million: float = 0.0
    output_usd_per_million: float = 0.0
    cached_input_usd_per_million: float = 0.0
    price_version: str = "unpriced"
    retry: LLMRetryConfig = LLMRetryConfig()

    @property
    def configured(self) -> bool:
        return all((self.api_key, self.base_url, self.model))


@dataclass(frozen=True)
class AgentConfig:
    # 结构化收尾未提交时允许的额外纠正次数；Researcher/Writer 共用。
    finalization_attempts: int = 2
    # Supervisor 的研究轮次上限；一轮包含一次评估、派发和汇总。
    max_research_rounds: int = 3
    # Reflection 判定 fatal 后，Supervisor 允许组织的修复循环次数。
    max_post_review_recovery_cycles: int = 1
    # Writer 的 Agent turn 上限；每个 turn 是一次模型决策及其工具执行。
    writer_max_turns: int = 8
    writer_feedback_chars: int = 800
    writer_minimum_support: WriterSupportLevel = "direct"
    writer_max_markdown_chars: int = 24_000
    # 单次 ReadEvidence 每轮交付量（可一次请求至多 50 条，差额 not_read_ids 排队）。
    # 引用总条数不设上限——writer_max_selected_evidence 已移除：它制造过两次
    # 提交死循环，而聚焦度实际由"只能引用已读"+审阅把关，与条数无关。
    writer_read_batch_size: int = 30
    # reflection 对内容审查抖动/坏 JSON 的额外重试次数（传输重试另有其层）。
    reflection_retry_attempts: int = 1
    reflection_retry_initial_seconds: float = 2.0
    max_subtasks_per_round: int = 12
    max_parallel_workers: int = 3
    # Supervisor 当前可激活并交给研究综合稿选择的 Evidence 上限。
    supervisor_max_active_evidences: int = 30
    supervisor_preview_chars: int = 300
    # Supervisor 交给 Writer 的报告任务书输出限制。
    report_max_caveats: int = 6
    # 单一网页可贡献的 Evidence 上限，避免某篇来源主导整个研究方向。
    evidence_max_per_source: int = 2
    # 抓取正文统一写入 ResearchMaterialStore；短文内联，长文按行读取。
    document_inline_max_tokens: int = 6_000
    document_inline_total_max_tokens: int = 12_000
    document_read_max_ranges: int = 6
    document_read_max_lines: int = 240
    document_read_max_chars: int = 24_000
    document_grep_context_lines: int = 2
    document_grep_max_matches: int = 12
    document_grep_max_chars: int = 12_000
    evidence_add_batch_size: int = 8
    # ResearchAgent 的 Agent turn 上限；每个 turn 是一次模型决策及其工具执行。
    research_agent_max_turns: int = 12
    research_agent_max_queries: int = 6
    # 一个方向级 ResearchAgent 最多带回的 Evidence 数；限制上下文与成本，
    # 但不等同于单网页抽取上限。
    research_agent_max_evidences_per_direction: int = 6
    # 单方向发现过的完整候选档案上限；活跃工作集仍由上一项限制。
    research_agent_max_evidence_candidates_per_direction: int = 12
    research_agent_read_concurrency: int = 3
    # 来源抓取与解析的分层 deadline。
    source_fetch_timeout: float = 30.0
    source_parse_timeout: float = 20.0
    source_total_timeout: float = 180.0
    research_query_chars: int = 180
    research_observation_quote_chars: int = 500
    research_failure_history_limit: int = 5
    # 所有面向用户的自然语言输出（理由/旁白/任务描述/正文）的统一语言。
    # 注入各 system prompt 的 language_directive() 使用；quote 永远保持原文。
    output_language: str = "中文"

    def __post_init__(self) -> None:
        """校验跨层容量契约，避免配置突破下游 schema 边界。"""
        if (
            self.research_agent_max_evidences_per_direction
            > self.research_agent_max_evidence_candidates_per_direction
        ):
            raise ValueError(
                "AGENT_RESEARCH_MAX_EVIDENCES_PER_DIRECTION 不能大于 "
                "AGENT_RESEARCH_MAX_EVIDENCE_CANDIDATES_PER_DIRECTION"
            )
        if self.evidence_max_per_source > self.research_agent_max_evidence_candidates_per_direction:
            raise ValueError(
                "AGENT_EVIDENCE_MAX_PER_SOURCE 不能大于 "
                "AGENT_RESEARCH_MAX_EVIDENCE_CANDIDATES_PER_DIRECTION"
            )
        if self.report_max_caveats > REPORT_CAVEATS_HARD_LIMIT:
            raise ValueError(
                "AGENT_REPORT_MAX_CAVEATS 不能大于 ReportBrief 的安全上限 "
                f"{REPORT_CAVEATS_HARD_LIMIT}"
            )
        if self.supervisor_max_active_evidences > EVIDENCE_REFERENCES_HARD_LIMIT:
            raise ValueError(
                "AGENT_SUPERVISOR_MAX_ACTIVE_EVIDENCES 不能大于 Evidence 引用的"
                f"安全上限 {EVIDENCE_REFERENCES_HARD_LIMIT}"
            )
        if self.research_agent_max_evidence_candidates_per_direction > (
            EVIDENCE_REFERENCES_HARD_LIMIT
        ):
            raise ValueError(
                "AGENT_RESEARCH_MAX_EVIDENCE_CANDIDATES_PER_DIRECTION 不能大于 "
                f"Evidence 引用的安全上限 {EVIDENCE_REFERENCES_HARD_LIMIT}"
            )
        if self.writer_max_markdown_chars > REPORT_MARKDOWN_HARD_LIMIT_CHARS:
            raise ValueError(
                "AGENT_WRITER_MAX_MARKDOWN_CHARS 不能大于报告 Schema 的安全上限 "
                f"{REPORT_MARKDOWN_HARD_LIMIT_CHARS}"
            )


def language_directive(language: str) -> str:
    """生成注入各 agent/node system prompt 的语言纪律行（模板见 prompts/language.md）。"""
    from deepresearcher.prompts import (
        load_prompt,  # 延迟导入，避免 config 被 prompts 反向依赖时成环
    )

    return load_prompt("language").format(language=language)


@dataclass(frozen=True)
class SearchConfig:
    baidu_api_key: str = ""
    tavily_api_key: str = ""
    serpapi_api_key: str = ""
    provider: Literal["auto", "aliyun", "baidu", "tavily", "serpapi"] = "auto"
    fetch_provider_order: tuple[Literal["aliyun", "direct"], ...] = ("direct",)
    aliyun_region_id: str = "cn-beijing"
    aliyun_endpoint: str = "dtsai.cn-beijing.aliyuncs.com"
    aliyun_agent_name: str = "deepresearcher"
    aliyun_fetch_output_format: Literal["markdown", "html", "text"] = "markdown"
    timeout: float = 20.0
    max_results: int = 5
    retry_attempts: int = 3
    retry_initial_seconds: float = 1.0
    retry_max_seconds: float = 10.0
    retry_after_max_seconds: float = 60.0
    # 同一供应商同时允许的搜索请求数（跨 worker 全局）；限流窗口下降低并发比快速重试更有效。
    max_concurrent_requests: int = 2
    max_concurrent_fetches: int = 6
    tavily_include_raw_content: bool = True

    @property
    def configured(self) -> bool:
        if self.provider == "aliyun":
            # 阿里云 SDK 通过默认凭据链读取身份，不强制使用固定环境变量。
            return True
        if self.provider == "baidu":
            return bool(self.baidu_api_key)
        if self.provider == "tavily":
            return bool(self.tavily_api_key)
        if self.provider == "serpapi":
            return bool(self.serpapi_api_key)
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
    env_file = Path(_env("DEEPRESEARCHER_ENV_FILE", str(_PROJECT_ROOT / "env" / ".env")))
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
        max_concurrent_requests=max(1, _int_env("LLM_MAX_CONCURRENT_REQUESTS", 6)),
        provider_requests_per_minute=max(0, _int_env("LLM_PROVIDER_RPM", 0)),
        provider_tokens_per_minute=max(0, _int_env("LLM_PROVIDER_TPM", 0)),
        provider_output_token_reserve=max(0, _int_env("LLM_PROVIDER_OUTPUT_TOKEN_RESERVE", 1_024)),
        max_tokens_per_run=max(0, _int_env("LLM_MAX_TOKENS_PER_RUN", 0)),
        max_cost_usd_per_run=max(0.0, _float_env("LLM_MAX_COST_USD_PER_RUN", 0.0)),
        max_cost_usd_per_user_daily=max(0.0, _float_env("LLM_MAX_COST_USD_PER_USER_DAILY", 0.0)),
        max_cost_usd_platform_hourly=max(0.0, _float_env("LLM_MAX_COST_USD_PLATFORM_HOURLY", 0.0)),
        max_cost_usd_platform_daily=max(0.0, _float_env("LLM_MAX_COST_USD_PLATFORM_DAILY", 0.0)),
        input_usd_per_million=max(0.0, _float_env("LLM_INPUT_USD_PER_MILLION", 0.0)),
        output_usd_per_million=max(0.0, _float_env("LLM_OUTPUT_USD_PER_MILLION", 0.0)),
        cached_input_usd_per_million=max(0.0, _float_env("LLM_CACHED_INPUT_USD_PER_MILLION", 0.0)),
        price_version=_env("LLM_PRICE_VERSION", "unpriced") or "unpriced",
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
        finalization_attempts=max(0, _int_env("AGENT_FINALIZATION_ATTEMPTS", 2)),
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
        writer_max_markdown_chars=max(1_000, _int_env("AGENT_WRITER_MAX_MARKDOWN_CHARS", 24_000)),
        writer_read_batch_size=max(1, _int_env("AGENT_WRITER_READ_BATCH_SIZE", 30)),
        reflection_retry_attempts=max(0, _int_env("AGENT_REFLECTION_RETRY_ATTEMPTS", 1)),
        reflection_retry_initial_seconds=max(
            0.0, _float_env("AGENT_REFLECTION_RETRY_INITIAL_SECONDS", 2.0)
        ),
        max_subtasks_per_round=max(1, _int_env("AGENT_MAX_SUBTASKS_PER_ROUND", 12)),
        max_parallel_workers=max(1, _int_env("AGENT_MAX_PARALLEL_WORKERS", 3)),
        supervisor_max_active_evidences=max(
            1, _int_env("AGENT_SUPERVISOR_MAX_ACTIVE_EVIDENCES", 30)
        ),
        supervisor_preview_chars=max(100, _int_env("AGENT_SUPERVISOR_PREVIEW_CHARS", 300)),
        report_max_caveats=max(1, _int_env("AGENT_REPORT_MAX_CAVEATS", 6)),
        evidence_max_per_source=max(1, _int_env("AGENT_EVIDENCE_MAX_PER_SOURCE", 2)),
        document_inline_max_tokens=max(500, _int_env("AGENT_DOCUMENT_INLINE_MAX_TOKENS", 6_000)),
        document_inline_total_max_tokens=max(
            500, _int_env("AGENT_DOCUMENT_INLINE_TOTAL_MAX_TOKENS", 12_000)
        ),
        document_read_max_ranges=max(1, _int_env("AGENT_DOCUMENT_READ_MAX_RANGES", 6)),
        document_read_max_lines=max(1, _int_env("AGENT_DOCUMENT_READ_MAX_LINES", 240)),
        document_read_max_chars=max(1_000, _int_env("AGENT_DOCUMENT_READ_MAX_CHARS", 24_000)),
        document_grep_context_lines=max(0, _int_env("AGENT_DOCUMENT_GREP_CONTEXT_LINES", 2)),
        document_grep_max_matches=max(1, _int_env("AGENT_DOCUMENT_GREP_MAX_MATCHES", 12)),
        document_grep_max_chars=max(1_000, _int_env("AGENT_DOCUMENT_GREP_MAX_CHARS", 12_000)),
        evidence_add_batch_size=max(1, _int_env("AGENT_EVIDENCE_ADD_BATCH_SIZE", 8)),
        research_agent_max_turns=max(1, _int_env("AGENT_RESEARCH_MAX_TURNS", 12)),
        research_agent_max_queries=max(1, _int_env("AGENT_RESEARCH_MAX_QUERIES", 6)),
        research_agent_max_evidences_per_direction=max(
            1, _int_env("AGENT_RESEARCH_MAX_EVIDENCES_PER_DIRECTION", 6)
        ),
        research_agent_max_evidence_candidates_per_direction=max(
            1, _int_env("AGENT_RESEARCH_MAX_EVIDENCE_CANDIDATES_PER_DIRECTION", 12)
        ),
        research_agent_read_concurrency=max(1, _int_env("AGENT_RESEARCH_READ_CONCURRENCY", 3)),
        source_fetch_timeout=max(1.0, _float_env("AGENT_SOURCE_FETCH_TIMEOUT", 30.0)),
        source_parse_timeout=max(1.0, _float_env("AGENT_SOURCE_PARSE_TIMEOUT", 20.0)),
        source_total_timeout=max(1.0, _float_env("AGENT_SOURCE_TOTAL_TIMEOUT", 180.0)),
        research_query_chars=max(40, _int_env("AGENT_RESEARCH_QUERY_CHARS", 180)),
        research_observation_quote_chars=max(
            100, _int_env("AGENT_RESEARCH_OBSERVATION_QUOTE_CHARS", 500)
        ),
        research_failure_history_limit=max(1, _int_env("AGENT_RESEARCH_FAILURE_HISTORY_LIMIT", 5)),
        output_language=_env("AGENT_OUTPUT_LANGUAGE", "中文") or "中文",
    )


def _search_config() -> SearchConfig:
    fetch_provider_order = _ordered_choices_env(
        "FETCH_PROVIDER_ORDER", "direct", {"aliyun", "direct"}
    )
    return SearchConfig(
        provider=cast(
            Literal["auto", "aliyun", "baidu", "tavily", "serpapi"],
            _choice_env(
                "SEARCH_PROVIDER", "auto", {"auto", "aliyun", "baidu", "tavily", "serpapi"}
            ),
        ),
        fetch_provider_order=cast(tuple[Literal["aliyun", "direct"], ...], fetch_provider_order),
        aliyun_region_id=_env("ALIYUN_DTS_REGION_ID", "cn-beijing"),
        aliyun_endpoint=_env("ALIYUN_DTS_ENDPOINT", "dtsai.cn-beijing.aliyuncs.com"),
        aliyun_agent_name=_env("ALIYUN_DTS_AGENT_NAME", "deepresearcher"),
        aliyun_fetch_output_format=cast(
            Literal["markdown", "html", "text"],
            _choice_env("ALIYUN_DTS_FETCH_OUTPUT_FORMAT", "markdown", {"markdown", "html", "text"}),
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
        max_concurrent_fetches=max(1, _int_env("FETCH_MAX_CONCURRENT_REQUESTS", 6)),
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
