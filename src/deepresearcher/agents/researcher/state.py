"""ResearchAgent 的方向级运行状态和工具执行上下文。"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypedDict

from langchain_core.messages import BaseMessage

from deepresearcher.context.execution import AgentExecutionScope
from deepresearcher.evidence.models import Evidence
from deepresearcher.schemas.sources import source_domain
from deepresearcher.state import SubTask
from deepresearcher.tools.web.search.models import SearchCandidate


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
    remaining_gaps: list[str]
    conclusion: str
    status: str
    stop_reason: str


def evidence_observation_card(evidence: Evidence, *, quote_chars: int) -> dict[str, object]:
    """Researcher 读源后的观察视图；保留短原文，不暴露定位与审计正文。"""
    return {
        "evidence_id": evidence.evidence_id,
        "claim": evidence.claim,
        "quote": evidence.quote[:quote_chars],
        "support": evidence.support,
        "confidence": evidence.confidence,
        "source_title": evidence.source_title,
        "source_domain": source_domain(evidence.source_url),
        "source_profile": evidence.source_profile.model_dump(mode="json"),
        **({"published_at": evidence.published_at} if evidence.published_at else {}),
    }


@dataclass
class ResearchRuntimeContext:
    """不进入 State 的 ResearchAgent 运行时依赖。"""

    task: SubTask
    scope: AgentExecutionScope
    run_state: "DirectionRunState"
    search_sources: Callable[[list[str], str], Awaitable[dict[str, object]]]
    read_sources: Callable[[list[str], str], Awaitable[dict[str, object]]]
    on_url_already_attempted: Callable[[str], None] | None = None
    event_context: dict[str, object] = field(default_factory=dict)
    tool_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass
class DirectionRunState:
    # evidences 是有界完整候选档案；active_evidence_ids 才是模型当前工作集。
    evidences: list[Evidence] = field(default_factory=list)
    active_evidence_ids: set[str] = field(default_factory=set)
    active_evidence_limit: int = 6
    evidence_archive_limit: int = 12
    source_refs: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    read_urls: list[str] = field(default_factory=list)
    candidates: dict[str, SearchCandidate] = field(default_factory=dict)
    selected_candidate_ids: set[str] = field(default_factory=set)
    skipped: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    remaining_gaps: list[str] = field(default_factory=list)
    conclusion: str = ""
    stop_reason: str = "step_budget_exhausted"
    stop_detail: str = "方向级探索步数预算已耗尽。"

    def active_evidences(self) -> list[Evidence]:
        return [item for item in self.evidences if item.evidence_id in self.active_evidence_ids]

    def add_evidences(self, evidences: list[Evidence]) -> list[Evidence]:
        """加入候选档案，并按剩余槽位自动激活新 Evidence。"""
        existing = {item.evidence_id for item in self.evidences}
        archive_slots = max(0, self.evidence_archive_limit - len(self.evidences))
        added = [item for item in evidences if item.evidence_id not in existing][:archive_slots]
        self.evidences.extend(added)
        slots = max(0, self.active_evidence_limit - len(self.active_evidence_ids))
        self.active_evidence_ids.update(item.evidence_id for item in added[:slots])
        return added

    def release_evidence(self, evidence_ids: list[str]) -> list[str]:
        released = self.active_evidence_ids.intersection(evidence_ids)
        self.active_evidence_ids.difference_update(released)
        return sorted(released)

    def restore_evidence(self, evidence_ids: list[str]) -> list[str]:
        archived = {item.evidence_id for item in self.evidences}
        candidates = [
            item
            for item in dict.fromkeys(evidence_ids)
            if item in archived and item not in self.active_evidence_ids
        ]
        slots = max(0, self.active_evidence_limit - len(self.active_evidence_ids))
        restored = candidates[:slots]
        self.active_evidence_ids.update(restored)
        return restored
