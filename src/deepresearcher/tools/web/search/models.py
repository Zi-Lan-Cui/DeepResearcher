"""搜索层的输入输出契约。"""

import re
from typing import Literal, NotRequired, TypedDict

from pydantic import BaseModel, Field

from deepresearcher.schemas.sources import SourceProfile, source_domain
from deepresearcher.state import SubTask


class SearchResult(TypedDict, total=False):
    """供应商无关的候选结果结构。"""

    title: str
    url: str
    snippet: NotRequired[str]
    raw_content: NotRequired[str]
    content_provider: NotRequired[str]
    score: NotRequired[float]
    published_at: NotRequired[str]
    source_profile: NotRequired[dict[str, str | bool]]


class SearchCandidate(BaseModel):
    """ResearchAgent 可选择读取的稳定候选来源。"""

    candidate_id: str
    title: str = ""
    url: str
    snippet: str = ""
    # 仅供 Reader 在原站抓取失败时降级；Agent 候选卡不渲染此字段。
    raw_content: str = ""
    score: float = 0.0
    content_provider: str = ""
    published_at: str = ""  # 搜索引擎给出的发布时间（时效性判断用，非正文事实）
    # 域名启发式的一手/权威分级，暴露给模型以引导"优先一手来源"的选择（见 classify_source）。
    source_tier: str = ""
    source_profile: SourceProfile = Field(default_factory=SourceProfile)


# 规则只是读取前的候选先验；必须匹配完整 hostname，避免
# ``fake-gov.cn`` / ``bilibili.com.evil.test`` 等后缀欺骗。``(?:^|\.)`` 自动覆盖子域。
def _domain(*domains: str) -> re.Pattern[str]:
    alternatives = "|".join(re.escape(domain) for domain in domains)
    return re.compile(rf"(?:^|\.)(?:{alternatives})$")


_SOURCE_RULES: tuple[tuple[re.Pattern[str], SourceProfile], ...] = (
    (
        _domain("arxiv.org"),
        SourceProfile(
            source_type="academic",
            authority_tier="primary",
            publication_status="preprint",
            primary_source=True,
        ),
    ),
    (
        re.compile(r"(?:^|\.)(?:gov(?:\.cn)?|go\.jp)$"),
        SourceProfile(
            source_type="government",
            authority_tier="primary",
            publication_status="official",
            primary_source=True,
        ),
    ),
    (
        re.compile(r"(?:^|\.)(?:edu|edu\.cn|ac\.uk|ac\.cn)$"),
        SourceProfile(source_type="academic", authority_tier="primary", primary_source=True),
    ),
    (
        _domain(
            "openreview.net",
            "aclanthology.org",
            "acm.org",
            "ieee.org",
            "nature.com",
            "science.org",
            "sciencedirect.com",
            "springer.com",
            "wiley.com",
            "pubmed.ncbi.nlm.nih.gov",
        ),
        SourceProfile(source_type="academic", authority_tier="primary", primary_source=True),
    ),
    (
        _domain("ietf.org", "w3.org", "iso.org", "itu.int"),
        SourceProfile(
            source_type="standard",
            authority_tier="primary",
            publication_status="official",
            primary_source=True,
        ),
    ),
    (
        _domain("who.int", "oecd.org", "imf.org", "worldbank.org", "un.org"),
        SourceProfile(
            source_type="organization",
            authority_tier="primary",
            publication_status="official",
            primary_source=True,
        ),
    ),
    (
        re.compile(
            r"(?:^|\.)(?:docs|developer|developers|platform)\."
            r"(?:microsoft\.com|apple\.com|google\.com|openai\.com|anthropic\.com|nvidia\.com)$"
        ),
        SourceProfile(
            source_type="company",
            authority_tier="primary",
            publication_status="official",
            primary_source=True,
        ),
    ),
    (
        _domain(
            "zhihu.com",
            "csdn.net",
            "cnblogs.com",
            "juejin.cn",
            "jianshu.com",
            "medium.com",
            "substack.com",
            "wordpress.com",
            "blogspot.com",
        ),
        SourceProfile(source_type="blog", authority_tier="secondary"),
    ),
    (
        _domain(
            "bilibili.com",
            "youtube.com",
            "youtu.be",
            "reddit.com",
            "weibo.com",
            "twitter.com",
            "x.com",
        ),
        SourceProfile(source_type="media", authority_tier="secondary"),
    ),
    (
        _domain(
            "reuters.com",
            "bbc.com",
            "bbc.co.uk",
            "apnews.com",
            "xinhuanet.com",
            "people.com.cn",
            "thepaper.cn",
            "36kr.com",
        ),
        SourceProfile(source_type="media", authority_tier="secondary"),
    ),
)


def _hostname(url: str) -> str:
    return source_domain(url)


def describe_source(url: str) -> SourceProfile:
    """从 URL 保守推导来源画像；无法确认的属性必须保留 unknown。"""
    host = _hostname(url)
    for pattern, profile in _SOURCE_RULES:
        if pattern.search(host):
            return profile.model_copy(deep=True)
    return SourceProfile(authority_tier="secondary")


def classify_source(url: str) -> str:
    """primary=论文/官方/学术一手；general=其余（含商业媒体/聚合，默认按二手对待）。"""
    return "primary" if describe_source(url).authority_tier == "primary" else "general"


class SearchFailure(BaseModel):
    """单条搜索查询的失败信息；部分成功时也必须保留。"""

    query: str
    error: str


class SearchToolResult(BaseModel):
    """搜索工具的稳定返回契约；results 保留供应商原始候选字段。"""

    task_id: str
    status: Literal["completed", "failed"]
    results: list[SearchResult] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    failures: list[SearchFailure] = Field(default_factory=list)
    error: str = ""
    # 账户级不可用（额度/鉴权）：与普通"这次没搜到"区分，上层据此停止重试并快速收尾。
    provider_exhausted: bool = False
    provider_user_code: str = ""


def failed_search(
    task: SubTask,
    error: Exception,
    *,
    queries: list[str] | None = None,
    failures: list[SearchFailure] | None = None,
) -> SearchToolResult:
    """构造失败结果，同时保留已知的逐查询错误。"""
    search_queries = queries or [task["question"]]
    details = failures or [
        SearchFailure(query=query, error=str(error)[:500]) for query in search_queries
    ]
    from deepresearcher.tools.errors import ProviderExhaustedError

    exhausted = isinstance(error, ProviderExhaustedError)
    return SearchToolResult(
        task_id=task["id"],
        status="failed",
        queries=search_queries,
        failures=details,
        error=str(error)[:500],
        provider_exhausted=exhausted,
        provider_user_code=getattr(error, "user_code", "") if exhausted else "",
    )
