"""ResearchAgent 的方向级运行状态和工具执行上下文。"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypedDict

from langchain_core.messages import BaseMessage

from deepsearch_agent.evidence.models import Evidence
from deepsearch_agent.state import SubTask
from deepsearch_agent.tools.context import ToolExecutionContext
from deepsearch_agent.tools.search.models import SearchCandidate


class ResearchAgentState(TypedDict, total=False):
    """ResearchAgent 子图可保存、可恢复的业务状态。"""

    messages: list[BaseMessage]
    task: SubTask
    candidates: dict[str, SearchCandidate]
    evidences: list[Evidence]
    source_refs: list[str]
    queries: list[str]
    read_urls: list[str]
    skipped: list[str]
    failures: list[str]
    answered_points: list[str]
    remaining_gaps: list[str]
    conclusion: str
    status: str
    stop_reason: str


@dataclass
class ResearchRuntimeContext:
    """不进入 State 的 ResearchAgent 运行时依赖。"""

    task: SubTask
    execution: ToolExecutionContext
    run_state: "DirectionRunState"
    search_sources: Callable[[list[str], str], Awaitable[dict[str, object]]]
    read_sources: Callable[[list[str], str], Awaitable[dict[str, object]]]
    on_url_already_attempted: Callable[[str], None] | None = None
    event_context: dict[str, object] = field(default_factory=dict)


@dataclass
class DirectionRunState:
    evidences: list[Evidence] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    read_urls: list[str] = field(default_factory=list)
    candidates: dict[str, SearchCandidate] = field(default_factory=dict)
    selected_candidate_ids: set[str] = field(default_factory=set)
    skipped: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    answered_points: list[str] = field(default_factory=list)
    remaining_gaps: list[str] = field(default_factory=list)
    conclusion: str = ""
    stop_reason: str = "step_budget_exhausted"
    stop_detail: str = "方向级探索步数预算已耗尽。"
