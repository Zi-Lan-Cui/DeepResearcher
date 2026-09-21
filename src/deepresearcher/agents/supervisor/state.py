"""Supervisor 的运行态辅助对象。"""

import asyncio
from dataclasses import dataclass, field

from deepresearcher.agents.middleware.concurrency import ToolExecutionGate
from deepresearcher.agents.researcher import ResearchAgent
from deepresearcher.config import AgentConfig
from deepresearcher.evidence.models import Evidence
from deepresearcher.observability.events import AgentEmit
from deepresearcher.observability.execution import AgentExecutionScope
from deepresearcher.schemas import (
    ResearchDirectionResult,
    ResearchSynthesis,
    StopReason,
    SupervisorProgress,
)
from deepresearcher.schemas.sources import source_domain
from deepresearcher.state import ResearchState, SubTask, section


def evidence_card(evidence: Evidence) -> dict[str, object]:
    """Supervisor 的唯一 Evidence 观察视图；不暴露 quote 与审计正文。"""
    return {
        "evidence_id": evidence.evidence_id,
        "claim": evidence.claim,
        "support": evidence.support,
        "confidence": evidence.confidence,
        "source_title": evidence.source_title,
        "source_domain": source_domain(evidence.source_url),
        "source_profile": evidence.source_profile.model_dump(mode="json"),
        "research_direction": evidence.research_direction,
        **({"published_at": evidence.published_at} if evidence.published_at else {}),
    }


def synthesis_card(synthesis: ResearchSynthesis | None) -> dict[str, object] | None:
    """综合稿的扁平内容卡;无稿为 None。"""
    if synthesis is None:
        return None
    return {
        "based_on_working_set_revision": synthesis.based_on_working_set_revision,
        "answer_goal": synthesis.answer_goal,
        "overall_summary": synthesis.overall_summary,
        "aspects": [aspect.model_dump(mode="json") for aspect in synthesis.aspects],
        "selected_evidence_ids": synthesis.selected_evidence_ids,
        "open_gaps": synthesis.open_gaps,
        "conflicts": synthesis.conflicts,
        "next_actions": synthesis.next_actions,
        "decision_rationale": synthesis.decision_rationale,
    }


def synthesis_snapshot(synthesis: ResearchSynthesis | None) -> dict[str, object]:
    """Supervisor 工具回执与轮次观察共用的视图:恒两键、形状不随有无综合稿漂移。"""
    return {
        "synthesis_revision": synthesis.revision if synthesis is not None else 0,
        "research_synthesis": synthesis_card(synthesis),
    }


@dataclass(frozen=True)
class SupervisorDeps:
    """ResearchSupervisor 构造期的稳定零件；跨并发 loop 共享、只读(frozen 是纪律载体)。"""

    config: AgentConfig
    research_agent: ResearchAgent
    worker_limit: asyncio.Semaphore
    emit: AgentEmit


@dataclass
class SupervisorLoopContext:
    """Supervisor 一次工具 loop 的注入载荷:全部名词,没有动词。

    deps 是共享零件;loop_state 是本轮业务工作副本;scope 供观测归属。
    仅活在一次 loop 内,不作为 LangGraph 顶层 State 持久化。当前轮次唯一
    来源是 ``loop_state.current_round``。实现住在 services.py,tools.py 从
    本对象转交参数;本清单之外,工具对 supervisor 内部一无所知。

    读者含中间件(删改字段前先 grep agents/middleware/):
      scope        → observability:事件归因
      tool_gate    → serial_tools:读写栅栏(Revise/Complete 等 exclusive,
                     与在飞 ResearchDelegate 互斥——消灭"完结后状态仍变"竞态)
      bookkeeping_lock → 业务锁:services.delegate_research 的编号/absorb 临界区
    """

    scope: AgentExecutionScope
    deps: SupervisorDeps
    loop_state: "SupervisorLoopState"
    bookkeeping_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    tool_gate: ToolExecutionGate = field(default_factory=ToolExecutionGate)


@dataclass
class TaskExecution:
    """一次方向研究执行的完整产物：结果模型与 absorb 所需的 Evidence/来源清单。

    模型看到的方向报告由 delegate_research 返回值经 tools.py 协议层打包,
    本对象只承载数据、不承载消息。
    """

    task_result: ResearchDirectionResult
    evidences: list[Evidence]
    selected_evidence_ids: list[str]
    source_refs: list[str]

    @classmethod
    def failed_for(cls, task: SubTask, round_no: int, error: str) -> "TaskExecution":
        """构造 worker 异常降级的 failed 结果；失败经方向报告让模型可见。"""
        task_result = ResearchDirectionResult(
            task_id=task["id"],
            round=round_no,
            task_index=int(task.get("sequence", 0)),
            question=task["question"],
            research_direction=task["question"],
            execution_status="failed",
            coverage_status="insufficient",
            evidence_count=0,
            source_count=0,
            failures=[error],
            stop_reason="worker_exception",
            stop_detail=error,
        )
        return cls(
            task_result=task_result,
            evidences=[],
            selected_evidence_ids=[],
            source_refs=[],
        )


class SupervisorLoopState:
    """Supervisor 单次工具循环内的可变业务状态（非全局 State、非某 ResearchAgent 状态）。

    从顶层 ResearchState 载入初始数据，仅在本次 loop 内修改，不直接进入 LangGraph
    State；loop 结束后由 SupervisorStateUpdate 转成顶层 State 增量。职责：Evidence
    聚合、active 工作集、task result、coverage gaps、failure details、
    research synthesis、working_set_revision、stop reason、当前轮次。

    节点结束时把整份副本交给幂等 reducer 合并（merge_evidences / merge_task_results /
    merge_unique），reducer 按 id 折回原样——不手搓增量切片,以免把不变式拆成
    多处平行簿记而静默丢字段。
    """

    def __init__(
        self,
        state: ResearchState,
        *,
        current_round: int,
        active_evidence_limit: int,
    ):
        self.evidences = list(state.get("evidences", []))
        active_ids = state.get("active_evidence_ids")
        self.active_evidence_ids = (
            set(active_ids)
            if active_ids is not None
            else {item.evidence_id for item in self.evidences}
        )
        self.released_evidence_ids: set[str] = set()
        self.active_evidence_limit = active_evidence_limit
        self.source_refs = list(state.get("source_refs", []))
        self.task_results = list(state.get("task_results", []))
        # 当前轮次唯一来源是构造参数（由 _run_agent_loop 从持久化的 SupervisorProgress 推一次）；
        # 这里仍读 SupervisorProgress 只为取持久化的 coverage_gaps，不决定轮次。
        persisted = section(state, "supervisor", SupervisorProgress)
        self.current_round: int = current_round
        self.coverage_gaps = list(persisted.coverage_gaps)
        # 执行失败详情（截断异常文本等）。coverage_gaps 只装研究内容缺口，会渲染进
        # 用户报告的"未闭合缺口"；两者可见面与恢复策略不同，不得混用一个字段。
        self.failure_details: list[str] = []
        self.research_query = str(state.get("clarified_query", state.get("query", "")))
        self.sufficient = False
        self.working_set_revision = int(state.get("working_set_revision", 0) or 0)
        self.research_synthesis = self._restore_synthesis(state.get("research_synthesis"))
        self.completed_synthesis: ResearchSynthesis | None = None
        self.stop_reason: StopReason | None = None

        self._task_counter: int = 0

    @staticmethod
    def _restore_synthesis(value: object) -> ResearchSynthesis | None:
        if value is None:
            return None
        if isinstance(value, ResearchSynthesis):
            return value
        return ResearchSynthesis.model_validate(value)

    def set_stop_reason(self, reason: StopReason) -> None:
        """按 `StopReason.rank` 采纳终止原因；低权威度不覆盖已采纳的高权威度。

        取代过去"直接赋值 + 零散 is None 守卫"的隐式优先级：谁都能写、顺序说了算。
        现在写点只需调用本方法并选对 reason，冲突由声明式 rank 裁决（见 StopReason.rank）。
        """
        current = self.stop_reason
        if current is None or reason.rank >= current.rank:
            self.stop_reason = reason

    def _bump_working_set(self) -> None:
        """工作集版本前进的唯一出口。

        不变式：任何改变 `active_evidence_ids` 成员的操作都必须经此推进 revision，
        否则 `synthesis_is_fresh` 会误判综合稿仍然新鲜、放行基于过期证据的 ResearchComplete。
        新增 mutation 方法时记得调用它（版本单调性无法由类型系统强制，只能约定）。
        """
        self.working_set_revision += 1

    def allocate_task_index(self) -> int:
        """分配独立的任务序号，不把研究轮次编码进任务 ID。

        序号必须在**分配时刻**（runtime.bookkeeping_lock 内）消费掉：从已完成结果反推的
        派生值会被并行 delegate 的双方读到同一个 max+1，撞出的 task_id 让
        merge_task_results 按 id 去重时静默吞掉一个方向的研究结果。
        max(计数器, 已完成最大值)+1 同时兼容恢复执行时从快照重建的场景。
        """
        completed_max = max((item.task_index for item in self.task_results), default=0)
        self._task_counter = max(self._task_counter, completed_max) + 1
        return self._task_counter

    def absorb(self, execution: TaskExecution) -> None:
        """把一次方向研究产物并入工作状态。

        方向 Agent 的 remaining_gaps 只是 Supervisor 的观察线索，不能单独决定全局充分性；
        但必须保留下来，避免最终状态丢失诊断信息。
        """
        self.evidences.extend(execution.evidences)
        available = {item.evidence_id for item in execution.evidences}
        selected = [item for item in execution.selected_evidence_ids if item in available]
        slots = max(0, self.active_evidence_limit - len(self.active_evidence_ids))
        self.active_evidence_ids.update(selected[:slots])
        self.source_refs.extend(execution.source_refs)
        self.task_results.append(execution.task_result)
        self.coverage_gaps.extend(execution.task_result.remaining_gaps)
        self.coverage_gaps = list(dict.fromkeys(gap for gap in self.coverage_gaps if gap.strip()))
        self._bump_working_set()

    def release_evidence(self, evidence_ids: list[str]) -> list[str]:
        """从 Supervisor 当前工作集释放 Evidence；全量档案仍保留。"""
        existing = self.active_evidence_ids.intersection(evidence_ids)
        self.active_evidence_ids.difference_update(existing)
        self.released_evidence_ids.update(existing)
        if existing:
            self._bump_working_set()
        return sorted(existing)

    def restore_evidence(self, evidence_ids: list[str]) -> list[str]:
        """从全局 Evidence 档案恢复候选，同时遵守活跃工作集上限。"""
        archived = {item.evidence_id for item in self.evidences}
        candidates = [
            item
            for item in dict.fromkeys(evidence_ids)
            if item in archived and item not in self.active_evidence_ids
        ]
        slots = max(0, self.active_evidence_limit - len(self.active_evidence_ids))
        restored = candidates[:slots]
        self.active_evidence_ids.update(restored)
        self.released_evidence_ids.difference_update(restored)
        if restored:
            self._bump_working_set()
        return restored

    def synthesis_is_fresh(self, synthesis: ResearchSynthesis | None) -> bool:
        return bool(
            synthesis is not None
            and synthesis.based_on_working_set_revision == self.working_set_revision
        )

    def active_evidences(self) -> list[Evidence]:
        """返回当前工作集中的 Evidence。"""
        return [item for item in self.evidences if item.evidence_id in self.active_evidence_ids]


def working_set_snapshot(
    loop_state: "SupervisorLoopState", *, include_reserve: bool = True
) -> dict[str, object]:
    """Supervisor 工作集目录的唯一构造处。

    tools.py 的 ReadWorkingSet/拒绝回执用全量(include_reserve=True);
    轮次观察只取 active(省 token)。两处共用同一卡片形状与 revision 字段,
    不会再各自演化出子集分叉。
    """
    active = loop_state.active_evidences()
    snapshot: dict[str, object] = {
        "working_set_revision": loop_state.working_set_revision,
        "active_evidence": [evidence_card(item) for item in active],
        "active_evidence_count": len(active),
        "active_evidence_limit": loop_state.active_evidence_limit,
    }
    if include_reserve:
        active_ids = {item.evidence_id for item in active}
        reserve = [item for item in loop_state.evidences if item.evidence_id not in active_ids]
        snapshot["reserve_evidence"] = [evidence_card(item) for item in reserve]
        snapshot["reserve_evidence_count"] = len(reserve)
    return snapshot
