"""研究子 Agent 的输入输出契约。"""

from typing import Literal

from pydantic import BaseModel, Field

from deepsearch_agent.evidence.models import Evidence
from deepsearch_agent.state import SubTask
from deepsearch_agent.tools.search import SearchResult


class SearchCandidate(BaseModel):
    """ResearchAgent 可选择读取的稳定候选来源。"""

    candidate_id: str
    title: str = ""
    url: str
    snippet: str = ""
    score: float = 0.0
    content_provider: str = ""


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


class SourceReaderToolResult(BaseModel):
    """单来源读取与 Evidence 抽取工具的稳定返回契约。"""

    task_id: str
    status: Literal["completed", "failed", "skipped"]
    evidences: list[Evidence] = Field(default_factory=list)
    source_url: str = ""
    error: str = ""
    reason_code: str = ""


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
    return SearchToolResult(
        task_id=task["id"],
        status="failed",
        queries=search_queries,
        failures=details,
        error=str(error)[:500],
    )


def failed_read(task: SubTask, error: Exception) -> SourceReaderToolResult:
    return SourceReaderToolResult(task_id=task["id"], status="failed", error=str(error)[:500])


def skipped_read(
    task: SubTask, *, source_url: str, reason_code: str, reason: str
) -> SourceReaderToolResult:
    return SourceReaderToolResult(
        task_id=task["id"],
        status="skipped",
        source_url=source_url,
        reason_code=reason_code,
        error=reason[:500],
    )
