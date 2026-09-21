"""工具调用入参与工具结果契约。"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from deepresearcher.schemas.limits import (
    EVIDENCE_REFERENCES_HARD_LIMIT,
    SEARCH_QUERIES_PER_CALL,
    SEARCH_RESULTS_PAGE_HARD_LIMIT,
    SEARCH_RESULTS_PREVIEW_COUNT,
    SOURCE_CANDIDATES_PER_READ,
    STRUCTURED_COLLECTION_HARD_LIMIT,
    STRUCTURED_SUMMARY_HARD_LIMIT_CHARS,
    STRUCTURED_TEXT_HARD_LIMIT_CHARS,
)
from deepresearcher.schemas.reporting import ResearchAspect
from deepresearcher.vocab import Support


class SearchSources(BaseModel):
    """ResearchAgent 请求发现当前方向的候选来源。"""

    reason: str = Field(description="为什么这些检索式能补足当前方向的证据。")
    queries: list[str] = Field(
        min_length=1,
        max_length=SEARCH_QUERIES_PER_CALL,
        description=(
            "一到两条针对当前方向缺口的短检索式；写成自然关键词串，"
            "不要加英文引号包裹短语，也不要 AND/OR/NOT、+/-、括号等布尔语法。"
        ),
    )


class ReadSources(BaseModel):
    """ResearchAgent 从候选目录中选择实际读取的来源。"""

    candidate_ids: list[str] = Field(
        min_length=1,
        max_length=SOURCE_CANDIDATES_PER_READ,
        description="要读取的候选来源 ID；只能使用 SearchSources 返回的 ID。",
    )
    reason: str = Field(description="说明这些来源与当前方向缺口的关系。")


class ListSearchResults(BaseModel):
    """分页查看已落盘的搜索结果。"""

    search_id: str = Field(min_length=1, description="SearchSources 返回的搜索结果句柄。")
    offset: int = Field(default=0, ge=0, description="从第几条结果开始，从 0 计数。")
    limit: int = Field(
        default=SEARCH_RESULTS_PREVIEW_COUNT,
        ge=1,
        le=SEARCH_RESULTS_PAGE_HARD_LIMIT,
        description="本次返回的结果数。",
    )
    reason: str = Field(description="为什么需要继续查看该批搜索结果。")


class GrepDocument(BaseModel):
    """在已读取长文中按**单个**普通文本查询定位相关行。

    需要检索多个关键词时,在同一回合**并行发起多个 GrepDocument 调用**(该工具可
    并行)。不提供 queries 批量接口——单查询让每个词各自分页、互不抢占,也无需
    跨词配额簿记。返回带 total_matches/has_more/next_offset,用 offset 翻页取后续。
    """

    document_id: str = Field(min_length=1)
    query: str = Field(min_length=1, max_length=200)
    context_lines: int = Field(default=2, ge=0, le=10)
    offset: int = Field(default=0, ge=0, description="跳过该词前 N 个命中窗口以翻页。")
    reason: str = Field(description="该关键词与当前证据缺口的关系。")


class DocumentLineRange(BaseModel):
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)


class ReadDocument(BaseModel):
    """按行号批量读取已登记文档的有限窗口。"""

    document_id: str = Field(min_length=1)
    ranges: list[DocumentLineRange] = Field(min_length=1, max_length=32)
    reason: str = Field(description="为什么需要读取这些行段。")


class EvidenceSubmission(BaseModel):
    """Researcher 从已经看到的文档原文中提出一条原子 Evidence。"""

    document_id: str = Field(min_length=1)
    claim: str = Field(min_length=1, max_length=STRUCTURED_TEXT_HARD_LIMIT_CHARS)
    quote: str = Field(min_length=1, max_length=STRUCTURED_TEXT_HARD_LIMIT_CHARS)
    support: Support = "direct"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class AddEvidence(BaseModel):
    """批量提交 Evidence 候选；系统逐字验证后才会入池。"""

    evidences: list[EvidenceSubmission] = Field(min_length=1, max_length=32)
    reason: str = Field(description="这些 Evidence 如何回答当前方向。")


class ReadWorkingSet(BaseModel):
    """查看当前 Agent 工作集的轻量摘要。"""

    reason: str = Field(default="", description="说明需要重新检查工作集的原因。")


class ReleaseEvidence(BaseModel):
    """从当前 Agent 活跃工作集释放 Evidence；不删除 Evidence 档案。"""

    evidence_ids: list[str] = Field(min_length=1)
    reason: str = Field(description="说明这些 Evidence 为什么应从当前工作集中释放。")


class RestoreEvidence(BaseModel):
    """将 Evidence 档案中的候选重新放回当前活跃工作集。"""

    evidence_ids: list[str] = Field(min_length=1)
    reason: str = Field(description="说明为什么需要重新启用这些 Evidence。")


class ResearchDirectionComplete(BaseModel):
    """ResearchAgent 提交方向结果并结束当前工具循环。"""

    reason: str = Field(description="为什么当前方向可以停止探索。")
    selected_evidence_ids: list[str] = Field(
        default_factory=list,
        max_length=EVIDENCE_REFERENCES_HARD_LIMIT,
        description="最终推荐给 Supervisor 的当前活跃 Evidence ID。",
    )
    conclusion: str = Field(
        default="",
        description="基于最终选中 Evidence 的方向级简短综合；不得引入未取证事实。",
    )
    remaining_gaps: list[str] = Field(
        default_factory=list,
        max_length=STRUCTURED_COLLECTION_HARD_LIMIT,
        description="局部未解问题；不代表整项研究的全局缺口。",
    )


class ResearchDelegate(BaseModel):
    """派发方向级研究任务的工具调用 Schema；task_id 由本地程序分配，不信任模型。"""

    research_topic: str = Field(
        description=(
            "要研究的具体方向。必须包含研究对象、范围、待回答的局部问题、"
            "与已有方向的区别和完成标准；补缺时必须缩小到明确缺口，不能重述原问题。"
        )
    )


class ResearchComplete(BaseModel):
    """冻结最新研究综合稿并终止研究阶段。"""

    synthesis_revision: int = Field(ge=1)
    reason: str = Field(description="为什么该综合版本已经足以形成完整报告。")


class ReviseResearchSynthesis(BaseModel):
    """提交 Supervisor 对当前研究状态的下一版完整规范化综合稿。"""

    expected_revision: int = Field(
        ge=0,
        description="当前综合稿版本；尚未建立综合稿时传 0。",
    )
    expected_working_set_revision: int = Field(
        ge=0,
        description="当前工具观察到的 Evidence/任务工作集版本。",
    )
    answer_goal: str = Field(min_length=1, max_length=STRUCTURED_TEXT_HARD_LIMIT_CHARS)
    overall_summary: str = Field(min_length=1, max_length=STRUCTURED_SUMMARY_HARD_LIMIT_CHARS)
    aspects: list[ResearchAspect] = Field(
        min_length=1,
        max_length=STRUCTURED_COLLECTION_HARD_LIMIT,
        description=(
            "Supervisor 跨研究方向建立的完整证据支撑认知单元列表；"
            "不是 Researcher 方向或固定报告章节。covered 必须绑定 Evidence；"
            "partial、uncovered、conflicted 必须说明 remaining_gap。"
        ),
    )
    open_gaps: list[str] = Field(default_factory=list, max_length=STRUCTURED_COLLECTION_HARD_LIMIT)
    conflicts: list[str] = Field(default_factory=list, max_length=STRUCTURED_COLLECTION_HARD_LIMIT)
    next_actions: list[str] = Field(
        default_factory=list,
        max_length=STRUCTURED_COLLECTION_HARD_LIMIT,
        description="建议性后续研究动作；是否执行由后续的显式工具调用决定，系统不会自动执行。",
    )
    decision_rationale: str = Field(
        min_length=1,
        max_length=STRUCTURED_TEXT_HARD_LIMIT_CHARS,
        description="只解释当前证据选择、覆盖判断、缺口与冲突；不得引入未经 Evidence 支持的新事实。",
    )


# 工具回执前缀:模型靠它区分工具结果与用户补充;observability 剥同一前缀
# 提取计数指标。三处共用,单一来源在此。
TOOL_RECEIPT_PREFIX = "【系统工具执行结果"


def format_tool_receipt(payload: object) -> str:
    """统一工具回执格式:前缀行 + JSON 串(或原样字符串)。"""
    body = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return f"{TOOL_RECEIPT_PREFIX}；不是用户补充】\n{body}"
