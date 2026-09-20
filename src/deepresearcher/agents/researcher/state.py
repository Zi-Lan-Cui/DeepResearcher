"""ResearchAgent 的方向级运行状态和工具执行上下文。"""

import asyncio
from dataclasses import dataclass, field

from deepresearcher.agents.middleware.concurrency import ToolExecutionGate
from deepresearcher.config import AgentConfig
from deepresearcher.evidence.models import Evidence
from deepresearcher.observability.events import AgentEmit
from deepresearcher.observability.execution import AgentExecutionScope
from deepresearcher.schemas.sources import source_domain
from deepresearcher.state import SubTask
from deepresearcher.tools import SearchTool, SourceReaderTool
from deepresearcher.tools.web.documents import DocumentRef
from deepresearcher.tools.web.materials import ResearchMaterialStore
from deepresearcher.tools.web.search.models import SearchCandidate


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


@dataclass(frozen=True)
class ResearcherDeps:
    """ResearchAgent 构造期的稳定零件；跨并发 run 共享、只读(frozen 是纪律载体)。"""

    config: AgentConfig
    search_tool: SearchTool
    reader_tool: SourceReaderTool
    material_store: ResearchMaterialStore | None
    emit: AgentEmit


@dataclass
class ResearcherLoopContext:
    """ResearchAgent 一次 run 的注入载荷:全部名词,没有动词。

    deps 为共享零件;task/loop_state/event_context/commit_lock 属本轮 run。
    实现住在 services.py 的纯函数里,tools.py 从本对象转交参数;本清单之外,
    工具对 agent 内部一无所知。

    读者不只是 tools.py——中间件消费以下字段,删改前必须先查 agents/middleware/
    (注意两种读法、缺席后果不同):
      scope     → observability._event_context 以 getattr 鸭子读:模型回合/工具事件
                  归因(run_id 等);缺席不崩,但事件静默失去归属。
      tool_gate → serial_tools 直接属性读:全库唯一的调度栅栏(serial 工具 exclusive,
                  其余 shared 可并发但被 pending exclusive 挡);缺席当场
                  AttributeError。业务锁另名另责:本 agent 的 commit_lock
                  只护证据入池,与调度无关。
    """

    deps: ResearcherDeps
    scope: AgentExecutionScope
    task: SubTask
    loop_state: "ResearcherLoopState"
    event_context: dict[str, object] = field(default_factory=dict)
    commit_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    tool_gate: ToolExecutionGate = field(default_factory=ToolExecutionGate)


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
    stop_reason: str = "step_budget_exhausted"
    stop_detail: str = "方向级探索步数预算已耗尽。"
    # 搜索提供方账户级不可用（额度/鉴权）。仅作数据：让本方向与 Supervisor 的模型读到后
    # 自然收尾/停止派发，不引入控制流分支。
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
