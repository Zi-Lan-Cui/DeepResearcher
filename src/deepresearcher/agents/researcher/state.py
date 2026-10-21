"""ResearchAgent 的方向级运行状态和工具执行上下文。"""

from dataclasses import dataclass, field
from typing import Annotated, Any

from langchain.agents import AgentState

from deepresearcher.agents.channels import (
    merge_active_set,
    merge_dict_by_key,
    merge_or,
    replace_channel,
    retain_if_absent,
)
from deepresearcher.config import AgentConfig
from deepresearcher.evidence.models import Evidence
from deepresearcher.observability.events import AgentEmit
from deepresearcher.observability.execution import AgentExecutionScope
from deepresearcher.schemas import DirectionStopReason
from deepresearcher.schemas.sources import source_domain
from deepresearcher.state import SubTask, merge_evidences, merge_unique
from deepresearcher.tools import SearchTool, SourceReaderTool
from deepresearcher.tools.web.documents import DocumentRef
from deepresearcher.tools.web.materials import ResearchMaterialStore
from deepresearcher.tools.web.search.models import SearchCandidate


def _candidate_channel_entry(candidate: SearchCandidate) -> SearchCandidate:
    """进通道的候选去掉 raw_content:正文留在材料缓存,检查点只存可重取的引用信息。"""
    return candidate.model_copy(update={"raw_content": ""})


def _document_channel_entry(view: DocumentRef) -> DocumentRef:
    """文档进通道降为 DocumentRef:去掉 content 与 inline,grep/read 时经 material_store 按需取回。"""
    return DocumentRef.model_validate(view.model_dump())


class ResearcherAgentState(AgentState):
    """ResearchAgent 内层图的 state_schema:方向业务状态的持久化载体。

    AddEvidence、ReadSources 与 SearchSources 并发执行,只写合并语义通道;Release、
    Restore、ResearchDirectionComplete 串行执行(单独一轮),conclusion、remaining_gaps、
    stop_detail 这几个 replace 通道只由它们写。active 工作集通道存集合本身,
    工具出口只交 {add,remove} 差量;候选与文档进通道时剥离正文,只留可重取的引用信息。
    """

    evidences: Annotated[list[Evidence], merge_evidences]
    active_evidence_ids: Annotated[list[str], merge_active_set]
    source_refs: Annotated[list[str], merge_unique]
    queries: Annotated[list[str], merge_unique]
    read_urls: Annotated[list[str], merge_unique]
    skipped: Annotated[list[str], merge_unique]
    failures: Annotated[list[str], merge_unique]
    candidates: Annotated[dict[str, SearchCandidate], merge_dict_by_key]
    search_batches: Annotated[dict[str, list[str]], merge_dict_by_key]
    documents: Annotated[dict[str, DocumentRef], merge_dict_by_key]
    selected_candidate_ids: Annotated[list[str], merge_unique]
    remaining_gaps: Annotated[list[str], replace_channel]
    conclusion: Annotated[str, replace_channel]
    stop_reason: Annotated[DirectionStopReason | None, retain_if_absent]
    stop_detail: Annotated[str, replace_channel]
    provider_exhausted: Annotated[bool, merge_or]


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


def working_set_snapshot(loop_state: "ResearcherLoopState") -> dict[str, object]:
    """方向工作集的轻量摘要唯一构造处;与 supervisor 的 working_set_snapshot 对称。"""
    evidences = loop_state.active_evidences()
    return {
        "active_evidence": [
            {
                "evidence_id": item.evidence_id,
                "subtask_id": item.subtask_id,
                "claim": item.claim,
                "support": item.support,
                "confidence": item.confidence,
                "source_title": item.source_title,
                "source_domain": source_domain(item.source_url),
            }
            for item in evidences
        ],
        "active_evidence_count": len(evidences),
        "active_evidence_limit": loop_state.active_evidence_limit,
        "reserve_evidence_count": len(loop_state.evidences) - len(evidences),
        "archive_evidence_count": len(loop_state.evidences),
        "archive_evidence_limit": loop_state.evidence_archive_limit,
    }


@dataclass(frozen=True)
class ResearcherDeps:
    """ResearchAgent 构造期的稳定零件;跨并发 run 共享、只读,frozen 数据类保证这一点。"""

    config: AgentConfig
    search_tool: SearchTool
    reader_tool: SourceReaderTool
    material_store: ResearchMaterialStore | None
    emit: AgentEmit


@dataclass
class ResearcherLoopContext:
    """ResearchAgent 一次 run 的注入载荷:字段全是数据,不含行为。

    deps 为共享零件;task、loop_state、event_context 属本轮 run。
    行为实现是 services.py 里的纯函数,tools.py 从本对象转交参数;本清单之外,
    工具不依赖 agent 内部任何其他对象。

    消费方除 tools.py 还有中间件:observability._event_context 用 getattr 读取 scope,
    模型回合与工具事件的 run_id 等归因依赖它;缺席不崩溃,但事件静默失去归属。
    """

    deps: ResearcherDeps
    scope: AgentExecutionScope
    task: SubTask
    loop_state: "ResearcherLoopState"
    event_context: dict[str, object] = field(default_factory=dict)

    def rebuild_loop_state(self, values: dict[str, Any]) -> None:
        """每次模型回合前从通道现值重建 loop_state;两个上限按本轮 config 现取,不进通道。"""
        config = self.deps.config
        self.loop_state = ResearcherLoopState.from_channel_values(
            values,
            active_evidence_limit=config.research_agent_max_evidences_per_direction,
            evidence_archive_limit=config.research_agent_max_evidence_candidates_per_direction,
        )


@dataclass
class ResearcherLoopState:
    # evidences 是有界完整候选档案；active_evidence_ids 才是模型当前工作集。
    evidences: list[Evidence] = field(default_factory=list)
    active_evidence_ids: set[str] = field(default_factory=set)
    active_evidence_limit: int = 6
    evidence_archive_limit: int = 12
    source_refs: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    read_urls: list[str] = field(default_factory=list)
    candidates: dict[str, SearchCandidate] = field(default_factory=dict)
    search_batches: dict[str, list[str]] = field(default_factory=dict)
    documents: dict[str, DocumentRef] = field(default_factory=dict)
    selected_candidate_ids: set[str] = field(default_factory=set)
    skipped: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    remaining_gaps: list[str] = field(default_factory=list)
    conclusion: str = ""
    stop_reason: DirectionStopReason = DirectionStopReason.STEP_BUDGET_EXHAUSTED
    stop_detail: str = "方向级探索步数预算已耗尽。"
    # 搜索提供方账户级不可用,即额度耗尽或鉴权失败。仅作数据:本方向与 Supervisor 的
    # 模型读到后自然收尾、停止派发,不引入控制流分支。
    provider_exhausted: bool = False

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

    def capture_active(self) -> set[str]:
        """并发工具执行前的活跃集快照,出口据此算 add/remove 差量。"""
        return set(self.active_evidence_ids)

    def active_diff_since(self, before: set[str]) -> dict[str, list[str]]:
        """本批活跃集变化折成 {add,remove} 差量,只经 active_evidence_ids 通道提交。"""
        return {
            "add": sorted(self.active_evidence_ids - before),
            "remove": sorted(before - self.active_evidence_ids),
        }

    def initial_channel_state(self) -> dict[str, Any]:
        """把当前 loop_state 一次性种进内层通道,作 ainvoke 入口 input。"""
        return {
            "evidences": list(self.evidences),
            "active_evidence_ids": {"add": sorted(self.active_evidence_ids), "remove": []},
            "source_refs": list(self.source_refs),
            "queries": list(self.queries),
            "read_urls": list(self.read_urls),
            "skipped": list(self.skipped),
            "failures": list(self.failures),
            "candidates": {
                key: _candidate_channel_entry(value) for key, value in self.candidates.items()
            },
            "search_batches": dict(self.search_batches),
            "documents": {
                key: _document_channel_entry(value) for key, value in self.documents.items()
            },
            "selected_candidate_ids": sorted(self.selected_candidate_ids),
            "remaining_gaps": list(self.remaining_gaps),
            "conclusion": self.conclusion,
            "stop_reason": self.stop_reason,
            "stop_detail": self.stop_detail,
            "provider_exhausted": self.provider_exhausted,
        }

    @classmethod
    def from_channel_values(
        cls,
        values: dict[str, Any],
        *,
        active_evidence_limit: int,
        evidence_archive_limit: int,
    ) -> "ResearcherLoopState":
        """从内层通道现值重建 loop_state;缺席键回落空态默认。"""
        view = cls(
            active_evidence_limit=active_evidence_limit,
            evidence_archive_limit=evidence_archive_limit,
        )
        if "evidences" in values:
            view.evidences = [
                item if isinstance(item, Evidence) else Evidence.model_validate(item)
                for item in values["evidences"]
            ]
        if "active_evidence_ids" in values:
            view.active_evidence_ids = set(values["active_evidence_ids"])
        for key in ("source_refs", "queries", "read_urls", "skipped", "failures", "remaining_gaps"):
            if key in values:
                setattr(view, key, list(values[key] or []))
        if "candidates" in values:
            view.candidates = {
                key: value
                if isinstance(value, SearchCandidate)
                else SearchCandidate.model_validate(value)
                for key, value in (values["candidates"] or {}).items()
            }
        if "search_batches" in values:
            view.search_batches = {
                key: list(value) for key, value in (values["search_batches"] or {}).items()
            }
        if "documents" in values:
            view.documents = {
                key: value if isinstance(value, DocumentRef) else DocumentRef.model_validate(value)
                for key, value in (values["documents"] or {}).items()
            }
        if "selected_candidate_ids" in values:
            view.selected_candidate_ids = set(values["selected_candidate_ids"] or [])
        if "conclusion" in values:
            view.conclusion = str(values["conclusion"] or "")
        if "stop_detail" in values:
            view.stop_detail = str(values["stop_detail"] or "")
        if values.get("stop_reason") is not None:
            view.stop_reason = DirectionStopReason(values["stop_reason"])
        if "provider_exhausted" in values:
            view.provider_exhausted = bool(values["provider_exhausted"])
        return view
