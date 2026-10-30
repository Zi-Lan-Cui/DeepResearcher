"""统一配置入口"""

import json
import re
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, cast

from dotenv import load_dotenv

from deepresearcher.env import env_bool as _bool_env
from deepresearcher.env import env_float as _float_env
from deepresearcher.env import env_int as _int_env
from deepresearcher.env import env_str as _env
from deepresearcher.health import Cooldown
from deepresearcher.pricing import PRICE_TABLE_VERSION
from deepresearcher.schemas.limits import (
    EVIDENCE_REFERENCES_HARD_LIMIT,
    REPORT_CAVEATS_HARD_LIMIT,
    REPORT_MARKDOWN_HARD_LIMIT_CHARS,
)
from deepresearcher.vocab import SUPPORT_ORDER, Support

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 已实现的搜索供应商，也是 SEARCH_PROVIDER_ORDER 的合法取值。
SEARCH_PROVIDERS = frozenset({"aliyun", "baidu", "tavily", "serpapi", "deepseek"})
# DeepSeek 排在最前：它自带正文与权威性评分，凭据未配置时自动跳过。
SEARCH_PROVIDER_ORDER = ("deepseek", "aliyun", "baidu", "tavily", "serpapi")
FETCH_PROVIDERS = frozenset({"aliyun", "direct", "deepseek"})
# Tavily 的延迟与相关度档位。ultra-fast 每个 URL 只回一条摘要，
# 其它三档回 chunks_per_source 段可拼接片段。
TAVILY_SEARCH_DEPTHS = frozenset({"basic", "advanced", "fast", "ultra-fast"})


def _choice_env(name: str, default: str, choices: set[str]) -> str:
    value = _env(name, default).lower()
    return value if value in choices else default


def _cidr_list_env(name: str) -> tuple[str, ...]:
    """读取 CIDR 清单：dotenv 多行引号值或逗号分隔均可；未配置返回空。"""
    return tuple(item.strip() for item in re.split(r"[,\n]", _env(name)) if item.strip())


def _ordered_choices_env(name: str, default: str, choices: AbstractSet[str]) -> tuple[str, ...]:
    """读取有序、逗号分隔的选项列表；显式拒绝未知值。"""
    values = tuple(item.strip().lower() for item in _env(name, default).split(","))
    invalid = [item or "<empty>" for item in values if item not in choices]
    if invalid:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{name} 包含无效值：{', '.join(invalid)}；可选值：{allowed}")
    return values


def project_path_env(name: str, default: Path) -> Path:
    """读取路径配置；相对路径统一以项目根目录为基准。"""
    path = Path(_env(name, str(default))).expanduser()
    return path if path.is_absolute() else _PROJECT_ROOT / path


# 模型上下文窗口默认值；生产由 graph 从 settings 显式传，构造默认只服务测试。
DEFAULT_CONTEXT_WINDOW_TOKENS = 32_768


@dataclass(frozen=True)
class LLMConfig:
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    temperature: float = 0.0
    timeout: float = 60.0
    context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS
    max_concurrent_requests: int = 6
    provider_requests_per_minute: int = 0
    provider_tokens_per_minute: int = 0
    provider_output_token_reserve: int = 1_024
    # 单次请求输出上限
    max_completion_tokens: int = 32_768
    max_tokens_per_run: int = 0
    max_cost_cny_per_run: float = 0.0
    max_cost_cny_per_user_daily: float = 0.0
    max_cost_cny_platform_hourly: float = 0.0
    max_cost_cny_platform_daily: float = 0.0
    price_version: str = PRICE_TABLE_VERSION
    # 结构化输出不合规时最多请求几次（含首次）；用尽后把最后一次异常抛给调用方。
    structured_max_attempts: int = 2
    # 传递自定义参数给模型。
    request_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def configured(self) -> bool:
        return all((self.api_key, self.base_url, self.model))


@dataclass(frozen=True)
class AgentConfig:
    # 模型只输出纯文本、未调提交工具时最多驳回几回合；用尽后放行，交业务层作回退。Researcher/Writer 共用。
    submission_max_rejections: int = 2
    # Supervisor 的研究轮次上限；一轮包含一次评估、派发和汇总。
    max_research_rounds: int = 3
    # 审阅轮次上限（含首次审阅）；用尽后不再返工，按最后一版草稿交付。
    max_review_attempts: int = 2
    # Writer 的 Agent turn 上限；每个 turn 是一次模型决策及其工具执行。
    writer_max_turns: int = 8
    writer_feedback_chars: int = 800
    # 低于该等级的 Evidence 不进写作材料
    writer_minimum_support: Support = "partial"
    writer_max_markdown_chars: int = 24_000
    # 单次 ReadEvidence 每轮交付量；
    writer_read_batch_size: int = 30
    max_tasks_per_round: int = 12
    # Supervisor 的 Agent turn 上限；每个 turn 是一次模型决策及其工具执行。
    supervisor_max_turns: int = 34
    supervisor_max_concurrent_tasks: int = 3
    # Supervisor 当前可激活并交给研究综合稿选择的 Evidence 上限。
    supervisor_max_active_evidences: int = 30
    supervisor_preview_chars: int = 300
    # Supervisor 交给 Writer 的报告任务书输出限制。
    report_max_caveats: int = 6
    # 单一网页可贡献的 Evidence 上限，避免某篇来源主导整个研究任务。
    evidence_max_per_source: int = 4
    # 抓取正文统一写入 ResearchMaterialStore；短文内联，长文按行读取。
    document_inline_max_tokens: int = 6_000
    document_inline_total_max_tokens: int = 12_000
    document_read_max_ranges: int = 6
    document_read_max_lines: int = 240
    document_read_max_chars: int = 24_000
    document_grep_context_lines: int = 2
    document_grep_max_matches: int = (
        12  # 单个查询词一次调用的返回窗口上限；多词靠并行多次调用，不是跨词总数
    )
    document_grep_max_chars: int = 12_000
    document_grep_max_line_chars: int = 2_000
    evidence_add_batch_size: int = 8
    # Researcher 的 Agent turn 上限；每个 turn 是一次模型决策及其工具执行。
    research_agent_max_turns: int = 12
    research_agent_max_queries: int = 6
    # 一个任务级 Researcher 最多带回的 Evidence 数
    research_agent_max_evidences_per_task: int = 6
    # 单任务发现过的完整候选档案上限；活跃工作集仍由上一项限制。
    research_agent_max_evidence_candidates_per_task: int = 12
    research_agent_max_concurrent_source_reads: int = 3
    # 来源抓取与解析的分层 deadline。
    source_fetch_timeout: float = 30.0
    source_parse_timeout: float = 20.0
    source_total_timeout: float = 180.0
    research_query_chars: int = 180
    research_observation_quote_chars: int = 500
    research_failure_history_limit: int = 5
    # 面向用户的自然语言输出的统一语言
    output_language: str = "中文"

    def __post_init__(self) -> None:
        """校验跨层容量契约，避免配置突破下游 schema 边界。"""
        if (
            self.research_agent_max_evidences_per_task
            > self.research_agent_max_evidence_candidates_per_task
        ):
            raise ValueError(
                "AGENT_RESEARCH_MAX_EVIDENCES_PER_TASK 不能大于 "
                "AGENT_RESEARCH_MAX_EVIDENCE_CANDIDATES_PER_TASK"
            )
        if self.evidence_max_per_source > self.research_agent_max_evidence_candidates_per_task:
            raise ValueError(
                "AGENT_EVIDENCE_MAX_PER_SOURCE 不能大于 "
                "AGENT_RESEARCH_MAX_EVIDENCE_CANDIDATES_PER_TASK"
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
        if self.research_agent_max_evidence_candidates_per_task > (EVIDENCE_REFERENCES_HARD_LIMIT):
            raise ValueError(
                "AGENT_RESEARCH_MAX_EVIDENCE_CANDIDATES_PER_TASK 不能大于 "
                f"Evidence 引用的安全上限 {EVIDENCE_REFERENCES_HARD_LIMIT}"
            )
        if self.writer_max_markdown_chars > REPORT_MARKDOWN_HARD_LIMIT_CHARS:
            raise ValueError(
                "AGENT_WRITER_MAX_MARKDOWN_CHARS 不能大于报告 Schema 的安全上限 "
                f"{REPORT_MARKDOWN_HARD_LIMIT_CHARS}"
            )


@dataclass(frozen=True)
class SearchConfig:
    baidu_api_key: str = ""
    tavily_api_key: str = ""
    serpapi_api_key: str = ""
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    # 检索时一并取回正文：命中量不变、只多花响应体积，换取抓取失败时能就地降级。
    deepseek_search_content: bool = True
    # 阿里云 SDK 无 API Key 可读，只能按环境变量判断凭据来源是否已给出。
    aliyun_configured: bool = False
    provider: Literal["auto", "aliyun", "baidu", "tavily", "serpapi", "deepseek"] = "auto"
    # provider=auto 时的尝试顺序；显式写某一家则只用它，不降级。
    search_provider_order: tuple[str, ...] = SEARCH_PROVIDER_ORDER
    # 连续失败几次把该供应商移出候选池，以及首次冷却时长（之后按被踢次数递增）。
    fail_threshold: int = 3
    cooldown_seconds: float = 60.0
    fetch_provider_order: tuple[Literal["aliyun", "direct", "deepseek"], ...] = ("direct",)
    aliyun_region_id: str = "cn-beijing"
    aliyun_endpoint: str = "dtsai.cn-beijing.aliyuncs.com"
    aliyun_agent_name: str = "deepresearcher"
    aliyun_fetch_output_format: Literal["markdown", "html", "text"] = "markdown"
    # 阿里云 SDK 的出口代理。留空即直连，并显式写 no_proxy=* —— SDK 会自己读
    # HTTP(S)_PROXY，进程被外部注入代理时，请求会被转到一个未必连得上阿里云的代理。
    aliyun_proxy: str = ""
    # 搜索与抓取共用同一个 HTTP 客户端，这里是它的单请求超时。
    http_timeout: float = 20.0
    max_results: int = 5
    # 单次请求最多发几次（含首次）；用尽后抛带上一次原因的 ToolRequestError。
    max_attempts: int = 3
    retry_initial_seconds: float = 1.0
    retry_max_seconds: float = 10.0
    retry_after_max_seconds: float = 60.0
    # 同一供应商同时允许的搜索请求数，为单个 worker 限制
    search_max_concurrent_requests: int = 2
    # 抓取侧的同时在线请求数，与搜索分开计数
    fetch_max_concurrent_requests: int = 6
    # 单份来源正文的字节上限：超限判内容不可用，不进解析，也不让 BeautifulSoup 建 DOM。
    max_fetch_bytes: int = 24 * 1024 * 1024
    # 抓取出口代理。留空即直连，并把代理显式置空 —— libcurl 在 CURLOPT_PROXY
    # 未设置时会自己读 HTTPS_PROXY，进程被外部注入代理时 primary_ip 变成代理
    # 地址，逐跳校验必然误判成内网。填了地址才走代理，此时不再校验连接对端。
    fetch_proxy: str = ""
    tavily_include_raw_content: bool = True
    # 档位只影响 content 的相关度与切分粒度，不改变「是否返回完整正文」。
    tavily_search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] = "basic"
    # 追加拦截清单：私网与保留段由 PublicUrlGuard 的可达性基线直接拦下，这里只封堵
    # 需要额外拒绝的公网段。支持逐行或逗号分隔两种写法，网段合法性在构造时校验。
    ssrf_block_cidrs: tuple[str, ...] = ()
    # fake-IP 代理会把公网域名解析到 198.18.0.0/15 之类的私网段，这类环境需整体
    # 关掉基线判据，只留上面那份清单。生产环境不要打开。
    ssrf_allow_private_addresses: bool = False

    @property
    def candidates(self) -> tuple[str, ...]:
        """本次配置允许尝试的供应商序列；显式指定时只有一家。"""
        return self.search_provider_order if self.provider == "auto" else (self.provider,)

    @property
    def search_provider(self) -> str:
        """按 candidates 取第一个凭据就绪的供应商；全不可用返回 unconfigured。"""
        for name in self.candidates:
            if self.ready(name):
                return name
        return "unconfigured"

    @property
    def cooldown(self) -> Cooldown:
        """连续失败后的冷却策略，时长按被踢次数递增，五倍封顶。"""
        return Cooldown(
            threshold=self.fail_threshold,
            base_seconds=self.cooldown_seconds,
            max_seconds=self.cooldown_seconds * 5,
        )

    def ready(self, name: str) -> bool:
        """该供应商的凭据是否已给出。"""
        if name == "aliyun":
            # 显式选择时按可用处理：身份由 SDK 默认凭据链决定，构造期验证不了。
            return self.provider == "aliyun" or self.aliyun_configured
        return bool(self._api_key(name))

    @property
    def configured(self) -> bool:
        """是否有至少一个凭据就绪的供应商。"""
        return self.search_provider != "unconfigured"

    def _api_key(self, provider: str) -> str:
        return {
            "baidu": self.baidu_api_key,
            "tavily": self.tavily_api_key,
            "serpapi": self.serpapi_api_key,
            "deepseek": self.deepseek_api_key,
        }.get(provider, "")

    @property
    def aliyun_in_use(self) -> bool:
        """装配期是否需要创建阿里云客户端。"""
        search_wants = self.aliyun_configured and "aliyun" in self.candidates
        return search_wants or "aliyun" in self.fetch_provider_order


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


def _request_overrides_env() -> dict[str, dict[str, Any]]:
    """读取按 agent 的请求参数覆盖表；未配置返回空表。

    这一项无法用声明式字段表达，模型提供商的思考开关键名不同。
    """
    raw = _env("LLM_REQUEST_OVERRIDES")
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM_REQUEST_OVERRIDES 不是合法 JSON：{exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("LLM_REQUEST_OVERRIDES 必须是 {agent 名: 请求参数对象}")
    overrides: dict[str, dict[str, Any]] = {}
    for actor, value in parsed.items():
        if not isinstance(value, dict):
            raise ValueError(f"LLM_REQUEST_OVERRIDES 的 {actor} 必须是请求参数对象")
        overrides[str(actor)] = dict(value)
    return overrides


def _llm_config() -> LLMConfig:
    return LLMConfig(
        api_key=_env("LLM_API_KEY"),
        base_url=_env("LLM_BASE_URL"),
        model=_env("LLM_MODEL_ID"),
        temperature=_float_env("LLM_TEMPERATURE", 0.0),
        timeout=_float_env("LLM_TIMEOUT", 60.0),
        context_window_tokens=max(
            4_096, _int_env("LLM_CONTEXT_WINDOW_TOKENS", DEFAULT_CONTEXT_WINDOW_TOKENS)
        ),
        max_concurrent_requests=max(1, _int_env("LLM_MAX_CONCURRENT_REQUESTS", 6)),
        provider_requests_per_minute=max(0, _int_env("LLM_PROVIDER_RPM", 0)),
        provider_tokens_per_minute=max(0, _int_env("LLM_PROVIDER_TPM", 0)),
        provider_output_token_reserve=max(0, _int_env("LLM_PROVIDER_OUTPUT_TOKEN_RESERVE", 1_024)),
        max_completion_tokens=max(1024, _int_env("LLM_MAX_COMPLETION_TOKENS", 32_768)),
        max_tokens_per_run=max(0, _int_env("LLM_MAX_TOKENS_PER_RUN", 0)),
        max_cost_cny_per_run=max(0.0, _float_env("LLM_MAX_COST_CNY_PER_RUN", 0.0)),
        max_cost_cny_per_user_daily=max(0.0, _float_env("LLM_MAX_COST_CNY_PER_USER_DAILY", 0.0)),
        max_cost_cny_platform_hourly=max(0.0, _float_env("LLM_MAX_COST_CNY_PLATFORM_HOURLY", 0.0)),
        max_cost_cny_platform_daily=max(0.0, _float_env("LLM_MAX_COST_CNY_PLATFORM_DAILY", 0.0)),
        price_version=_env("LLM_PRICE_VERSION", PRICE_TABLE_VERSION) or PRICE_TABLE_VERSION,
        structured_max_attempts=max(1, _int_env("LLM_STRUCTURED_MAX_ATTEMPTS", 2)),
        request_overrides=_request_overrides_env(),
    )


def _agent_config() -> AgentConfig:
    return AgentConfig(
        submission_max_rejections=max(0, _int_env("AGENT_SUBMISSION_MAX_REJECTIONS", 2)),
        max_research_rounds=max(1, _int_env("AGENT_MAX_RESEARCH_ROUNDS", 3)),
        max_review_attempts=max(1, _int_env("AGENT_MAX_REVIEW_ATTEMPTS", 2)),
        writer_max_turns=max(1, _int_env("AGENT_WRITER_MAX_TURNS", 8)),
        writer_feedback_chars=max(100, _int_env("AGENT_WRITER_FEEDBACK_CHARS", 800)),
        writer_minimum_support=cast(
            Support,
            _choice_env(
                "AGENT_WRITER_MINIMUM_SUPPORT",
                "partial",
                set(SUPPORT_ORDER),
            ),
        ),
        writer_max_markdown_chars=max(1_000, _int_env("AGENT_WRITER_MAX_MARKDOWN_CHARS", 24_000)),
        writer_read_batch_size=max(1, _int_env("AGENT_WRITER_READ_BATCH_SIZE", 30)),
        max_tasks_per_round=max(1, _int_env("AGENT_MAX_TASKS_PER_ROUND", 12)),
        supervisor_max_turns=max(1, _int_env("AGENT_SUPERVISOR_MAX_TURNS", 34)),
        supervisor_max_concurrent_tasks=max(
            1, _int_env("AGENT_SUPERVISOR_MAX_CONCURRENT_TASKS", 3)
        ),
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
        document_grep_max_line_chars=max(
            100, _int_env("AGENT_DOCUMENT_GREP_MAX_LINE_CHARS", 2_000)
        ),
        evidence_add_batch_size=max(1, _int_env("AGENT_EVIDENCE_ADD_BATCH_SIZE", 8)),
        research_agent_max_turns=max(1, _int_env("AGENT_RESEARCH_MAX_TURNS", 12)),
        research_agent_max_queries=max(1, _int_env("AGENT_RESEARCH_MAX_QUERIES", 6)),
        research_agent_max_evidences_per_task=max(
            1, _int_env("AGENT_RESEARCH_MAX_EVIDENCES_PER_TASK", 6)
        ),
        research_agent_max_evidence_candidates_per_task=max(
            1, _int_env("AGENT_RESEARCH_MAX_EVIDENCE_CANDIDATES_PER_TASK", 12)
        ),
        research_agent_max_concurrent_source_reads=max(
            1, _int_env("AGENT_RESEARCH_MAX_CONCURRENT_SOURCE_READS", 3)
        ),
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


def _aliyun_credentials_present() -> bool:
    """环境变量里是否存在可用的阿里云凭据来源。"""
    if _env("ALIBABA_CLOUD_ACCESS_KEY_ID") and _env("ALIBABA_CLOUD_ACCESS_KEY_SECRET"):
        return True
    return bool(_env("ALIBABA_CLOUD_CREDENTIALS_URI"))


def _search_config() -> SearchConfig:
    fetch_provider_order = _ordered_choices_env("FETCH_PROVIDER_ORDER", "direct", FETCH_PROVIDERS)
    search_provider_order = _ordered_choices_env(
        "SEARCH_PROVIDER_ORDER", ",".join(SEARCH_PROVIDER_ORDER), SEARCH_PROVIDERS
    )
    return SearchConfig(
        provider=cast(
            Literal["auto", "aliyun", "baidu", "tavily", "serpapi", "deepseek"],
            _choice_env("SEARCH_PROVIDER", "auto", {"auto"} | SEARCH_PROVIDERS),
        ),
        search_provider_order=search_provider_order,
        fail_threshold=max(1, _int_env("SEARCH_PROVIDER_FAIL_THRESHOLD", 3)),
        cooldown_seconds=max(1.0, _float_env("SEARCH_PROVIDER_COOLDOWN_SECONDS", 60.0)),
        aliyun_configured=_aliyun_credentials_present(),
        fetch_provider_order=cast(
            tuple[Literal["aliyun", "direct", "deepseek"], ...], fetch_provider_order
        ),
        aliyun_region_id=_env("ALIYUN_WEB_REGION_ID", "cn-beijing"),
        aliyun_endpoint=_env("ALIYUN_WEB_ENDPOINT", "dtsai.cn-beijing.aliyuncs.com"),
        aliyun_agent_name=_env("ALIYUN_WEB_AGENT_NAME", "deepresearcher"),
        aliyun_proxy=_env("ALIYUN_WEB_PROXY"),
        aliyun_fetch_output_format=cast(
            Literal["markdown", "html", "text"],
            _choice_env("ALIYUN_WEB_FETCH_OUTPUT_FORMAT", "markdown", {"markdown", "html", "text"}),
        ),
        baidu_api_key=_env("BAIDU_API_KEY"),
        tavily_api_key=_env("TAVILY_API_KEY"),
        serpapi_api_key=_env("SERPAPI_API_KEY"),
        deepseek_api_key=_env("DEEPSEEK_API_KEY"),
        deepseek_base_url=_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/"),
        deepseek_search_content=_bool_env("DEEPSEEK_SEARCH_CONTENT", True),
        http_timeout=_float_env("HTTP_TIMEOUT", 20.0),
        max_results=max(1, _int_env("SEARCH_MAX_RESULTS", 5)),
        max_attempts=max(1, _int_env("HTTP_MAX_ATTEMPTS", 3)),
        retry_initial_seconds=max(0.0, _float_env("HTTP_RETRY_INITIAL_SECONDS", 1.0)),
        retry_max_seconds=max(0.0, _float_env("HTTP_RETRY_MAX_SECONDS", 10.0)),
        retry_after_max_seconds=max(0.0, _float_env("HTTP_RETRY_AFTER_MAX_SECONDS", 60.0)),
        search_max_concurrent_requests=max(1, _int_env("SEARCH_MAX_CONCURRENT_REQUESTS", 2)),
        fetch_max_concurrent_requests=max(1, _int_env("FETCH_MAX_CONCURRENT_REQUESTS", 6)),
        max_fetch_bytes=max(1, _int_env("FETCH_MAX_BYTES", 24 * 1024 * 1024)),
        fetch_proxy=_env("FETCH_PROXY"),
        tavily_include_raw_content=_bool_env("TAVILY_INCLUDE_RAW_CONTENT", True),
        tavily_search_depth=cast(
            Literal["basic", "advanced", "fast", "ultra-fast"],
            _choice_env("TAVILY_SEARCH_DEPTH", "basic", set(TAVILY_SEARCH_DEPTHS)),
        ),
        ssrf_block_cidrs=_cidr_list_env("SSRF_BLOCK_CIDRS"),
        ssrf_allow_private_addresses=_bool_env("SSRF_ALLOW_PRIVATE_ADDRESSES", False),
    )


def _app_config() -> AppConfig:
    return AppConfig(
        environment=_env("APP_ENV", "development"),
        log_level=_env("LOG_LEVEL", "INFO").upper(),
    )


def _observability_config() -> ObservabilityConfig:
    return ObservabilityConfig(
        log_dir=project_path_env("OBSERVABILITY_LOG_DIR", _PROJECT_ROOT / "var" / "logs"),
        log_file=_env("OBSERVABILITY_LOG_FILE", "agent.log"),
        event_file=_env("OBSERVABILITY_EVENT_FILE", "events.jsonl"),
        trace_file=_env("OBSERVABILITY_TRACE_FILE", "traces.jsonl"),
        max_text_chars=max(100, _int_env("OBSERVABILITY_MAX_TEXT_CHARS", 1_000)),
    )


def clear_settings_cache() -> None:
    """仅供测试或显式热加载配置使用。"""
    get_settings.cache_clear()
