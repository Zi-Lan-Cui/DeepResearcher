"""搜索层的输入输出契约。"""

from typing import Literal, NotRequired, TypedDict

from pydantic import BaseModel, Field

from deepsearch_agent.state import SubTask


class SearchResult(TypedDict, total=False):
    """供应商无关的候选结果结构。"""

    title: str
    url: str
    snippet: NotRequired[str]
    raw_content: NotRequired[str]
    content_provider: NotRequired[str]
    score: NotRequired[float]
    published_at: NotRequired[str]


class SearchCandidate(BaseModel):
    """ResearchAgent 可选择读取的稳定候选来源。"""

    candidate_id: str
    title: str = ""
    url: str
    snippet: str = ""
    score: float = 0.0
    content_provider: str = ""
    published_at: str = ""  # 搜索引擎给出的发布时间（时效性判断用，非正文事实）


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
