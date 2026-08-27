"""ResearchAgent 的方向级运行状态和工具执行上下文。"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from langchain_core.messages import BaseMessage

from deepsearch_agent.evidence.models import Evidence
from deepsearch_agent.schemas import ResearchDirectionDecision
from deepsearch_agent.state import SubTask
from deepsearch_agent.tools.research_models import SearchCandidate


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


@dataclass(frozen=True)
class ToolExecutionContext:
    task: SubTask
    decision: ResearchDirectionDecision
    tool_call_id: str
    messages: list[BaseMessage]
    run_state: DirectionRunState
    event_context: dict[str, object]
    claim_url: Callable[[str], Awaitable[bool]]
    on_url_already_attempted: Callable[[str], None] | None


ToolExecutor = Callable[[ToolExecutionContext], Awaitable[bool]]
