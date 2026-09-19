"""Supervisor 的运行态辅助对象。"""

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from langchain_core.messages import ToolMessage

from deepresearcher.context.execution import AgentExecutionScope
from deepresearcher.evidence.models import Evidence
from deepresearcher.schemas import (
    ResearchDirectionResult,
    ResearchProgress,
    ResearchSynthesis,
    ResearchToolResult,
    StopReason,
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


def synthesis_snapshot(synthesis: ResearchSynthesis | None) -> dict[str, object]:
    """Supervisor 工具回执与轮次观察共用的完整综合稿视图。"""
    if synthesis is None:
        return {"synthesis_revision": 0, "research_synthesis": None}
    return {
        "synthesis_revision": synthesis.revision,
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


@dataclass
class SupervisorRuntimeContext:
    """本次 Supervisor Agent 运行的依赖和可变工作状态。"""

    scope: AgentExecutionScope
    working: "WorkingState"
    delegate_research: Callable[[str], Awaitable[dict[str, object]]]
    round_no: int = 0
    tool_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass
class TaskExecution:
    """一次方向研究执行的完整产物：结果模型、聚合载荷与注入历史的工具消息。"""

    task_result: ResearchDirectionResult
    evidences: list[Evidence]
    selected_evidence_ids: list[str]
    source_refs: list[str]
    message: ToolMessage

    @classmethod
    def failed_for(
        cls,
        task: SubTask,
        round_no: int,
        error: str,
        *,
        tool_call_id: str,
    ) -> "TaskExecution":
        """构造 worker 异常降级的 failed 结果；失败同样注入历史让模型可见。"""
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
            message=cls._result_message(task, task_result, [], tool_call_id=tool_call_id),
        )

    @staticmethod
    def _result_message(
        task: SubTask,
        task_result: ResearchDirectionResult,
        evidences: list[Evidence],
        *,
        tool_call_id: str,
    ) -> ToolMessage:
        """方向结果以 JSON 载荷注入 Supervisor 上下文。

        携带方向结论与带回的 Evidence claim，每条 claim 只在所属方向的
        工具结果中出现一次；quote 留给 Writer 与引用审计，不进上下文。
        """
        tool_result = ResearchToolResult.from_direction_result(task_result)
        payload = {
            "research_direction": tool_result.question,
            "execution_status": tool_result.execution_status,
            "coverage_status": tool_result.coverage_status,
            "stop_reason": tool_result.stop_reason,
            "conclusion": tool_result.conclusion,
            "remaining_gaps": tool_result.remaining_gaps,
            "failures": tool_result.failures,
            "evidence": [evidence_card(item) for item in evidences],
        }
        return ToolMessage(
            content=json.dumps(payload, ensure_ascii=False),
            name="ResearchDelegate",
            tool_call_id=tool_call_id,
            artifact=task_result,
        )


class WorkingState:
    """工具循环的工作状态：State 载入的可变副本 + 去重簿记。

    节点结束时把整份副本交给 LangGraph 的幂等 reducer 合并
    （`merge_evidences` / `merge_task_results` / `merge_unique`）——reducer 会把
    已存在项按 id 折回原样。曾经这里靠 `_snapshot` 三元组 + `deltas()` 手搓
    "入口长度快照 / 出口位置切片"来只发新增,但那把一个不变式拆成了三处平行簿记
    (定义快照、切片、`_final_update` 搬运),加字段忘同步就静默丢数据。改回全量
    交给幂等 reducer 后,数据形状跟着 channel 语义走,少一层脆弱机制。
    """

    def __init__(
        self,
        state: ResearchState,
        *,
        dedup_key: Callable[[str], str],
        active_evidence_limit: int,
    ):
        self._dedup_key = dedup_key
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
        research = section(state, "research", ResearchProgress)
        self.current_round: int = research.current_round
        self.coverage_gaps = list(research.coverage_gaps)
        self.research_query = str(state.get("clarified_query", state.get("query", "")))
        self.seen_questions = {
            dedup_key(item.question) for item in self.task_results if item.question
        }
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

        序号必须在**分配时刻**（runtime.tool_lock 内）消费掉：从已完成结果反推的
        派生值会被并行 delegate 的双方读到同一个 max+1，撞出的 task_id 让
        merge_task_results 按 id 去重时静默吞掉一个方向的研究结果。
        max(计数器, 已完成最大值)+1 同时兼容恢复执行时从快照重建的场景。
        """
        completed_max = max((item.task_index for item in self.task_results), default=0)
        self._task_counter = max(self._task_counter, completed_max) + 1
        return self._task_counter

    def filter_new_tasks(self, tasks: list[SubTask], *, max_tasks: int) -> list[SubTask]:
        """问题级去重并截断；task_id 冲突（恢复执行）同样跳过。"""
        existing_ids = {item.task_id for item in self.task_results}
        kept: list[SubTask] = []
        for task in tasks:
            if task["id"] in existing_ids:
                continue
            key = self._dedup_key(task["question"])
            if not key or key in self.seen_questions:
                continue
            self.seen_questions.add(key)
            kept.append(task)
        return kept[:max_tasks]

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
