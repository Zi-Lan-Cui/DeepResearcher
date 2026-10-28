"""Supervisor 的依赖、loop 上下文与状态合并规则。"""

import asyncio
from dataclasses import dataclass, field
from typing import Annotated, Any

from langchain.agents import AgentState

from deepresearcher.agents.reducers import (
    merge_active_set,
    merge_monotonic,
    merge_or,
    merge_registry_by_key,
    replace_channel,
    retain_if_absent,
)
from deepresearcher.agents.researcher import Researcher
from deepresearcher.config import AgentConfig
from deepresearcher.evidence.models import Evidence
from deepresearcher.observability.events import AgentEmit
from deepresearcher.observability.scope import AgentScope
from deepresearcher.schemas import (
    ResearcherResult,
    ResearchSynthesis,
    StopReason,
    TaskResult,
)
from deepresearcher.schemas.sources import source_domain
from deepresearcher.state import (
    RunState,
    merge_evidences,
    merge_task_results,
    merge_unique,
)


def _merge_stop_reason_by_rank(
    current: StopReason | None, incoming: StopReason | None
) -> StopReason | None:
    if current is None:
        return incoming
    if incoming is None:
        return current
    return incoming if incoming.rank >= current.rank else current


def _integer_field(record: object, field_name: str) -> int:
    """从对象或字典取整数编号，缺失按 0 计。

    通道里可能是模型对象，也可能是旧检查点留下的半结构化快照。
    """
    value = record.get(field_name) if isinstance(record, dict) else getattr(record, field_name, 0)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def pending_task_payload(entry: object) -> dict[str, object]:
    """取登记项里的 Task 载荷，缺失返回空字典。"""
    if isinstance(entry, dict):
        return dict(entry.get("payload") or {})
    return dict(getattr(entry, "payload", None) or {})


def _tool_call_position(messages: object, tool_call_id: str) -> int | None:
    """在发起本次执行的 AIMessage 中定位该派发在 ResearchDelegate 序列里的序。

    只数派发调用而非整批 tool_calls 下标：UpdateWorkbench 与派发同批时
    编号照样连续，且对同一 AIMessage 的重放仍确定。
    """
    entries = messages if isinstance(messages, list) else []
    for message in reversed(entries):
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            continue
        position = 0
        for call in tool_calls:
            if call.get("id") == tool_call_id:
                return position
            if call.get("name") == "ResearchDelegate":
                position += 1
        break
    return None


class SupervisorState(AgentState):
    """Supervisor 内层图的 state_schema：循环业务状态的持久化载体。"""

    evidences: Annotated[list[Evidence], merge_evidences]
    active_evidence_ids: Annotated[list[str], merge_active_set]
    source_refs: Annotated[list[str], merge_unique]
    task_results: Annotated[list[TaskResult], merge_task_results]
    failure_details: Annotated[list[str], merge_unique]
    working_set_revision: Annotated[int, merge_monotonic]
    research_synthesis: Annotated[ResearchSynthesis | None, replace_channel]
    completed_synthesis: Annotated[ResearchSynthesis | None, retain_if_absent]
    provider_exhausted: Annotated[bool, merge_or]
    stop_reason: Annotated[StopReason | None, _merge_stop_reason_by_rank]
    task_index_counter: Annotated[int, merge_monotonic]
    pending_tasks: Annotated[dict[str, dict], merge_registry_by_key]
    plan: Annotated[list[dict], replace_channel]


# 老检查点的任务结果没有短题时，卡片回退用长契约截断到此长度。
_TASK_FALLBACK_CHARS = 30


def _task_label(evidence: Evidence, task_title: str) -> str:
    """任务归属展示名：派发短题优先，无短题回退长契约截断。"""
    if task_title:
        return task_title
    topic = evidence.research_direction
    if len(topic) > _TASK_FALLBACK_CHARS:
        return topic[: _TASK_FALLBACK_CHARS - 1] + "…"
    return topic


def evidence_card(evidence: Evidence, task_title: str = "") -> dict[str, object]:
    """Supervisor 的唯一 Evidence 观察视图；不暴露 quote 与审计正文。"""
    profile = evidence.source_profile
    return {
        "evidence_id": evidence.evidence_id,
        "claim": evidence.claim,
        "task_title": _task_label(evidence, task_title),
        "support": evidence.support,
        "confidence": evidence.confidence,
        "source_title": evidence.source_title,
        "source_domain": source_domain(evidence.source_url),
        "source_type": profile.source_type,
        "authority_tier": profile.authority_tier,
        "publication_status": profile.publication_status,
        "primary_source": profile.primary_source,
        **({"published_at": evidence.published_at} if evidence.published_at else {}),
    }


def synthesis_card(synthesis: ResearchSynthesis | None) -> dict[str, object] | None:
    """综合稿的扁平内容卡；无稿为 None。"""
    if synthesis is None:
        return None
    return {
        # 模型可见面统一工作台措辞；schema 字段 based_on_working_set_revision 是持久化名，不随视图改名。
        "based_on_workbench_revision": synthesis.based_on_working_set_revision,
        "answer_goal": synthesis.answer_goal,
        "thesis": synthesis.thesis,
        "aspects": [aspect.model_dump(mode="json") for aspect in synthesis.aspects],
        "selected_evidence_ids": synthesis.selected_evidence_ids,
        "decision_rationale": synthesis.decision_rationale,
    }


def synthesis_snapshot(synthesis: ResearchSynthesis | None) -> dict[str, object]:
    """Supervisor 工具回执与轮次观察共用的视图：恒两键、结构不随有无综合稿而变。"""
    return {
        "synthesis_revision": synthesis.revision if synthesis is not None else 0,
        "research_synthesis": synthesis_card(synthesis),
    }


@dataclass(frozen=True)
class SupervisorDeps:
    """Supervisor 构造期的稳定零件；跨并发 loop 共享、只读，frozen 数据类保证这一点。"""

    config: AgentConfig
    research_agent: Researcher
    worker_limit: asyncio.Semaphore
    emit: AgentEmit


@dataclass
class SupervisorContext:
    """Supervisor 一次 round 内跨模型回合存活的注入载荷。"""

    scope: AgentScope
    deps: SupervisorDeps
    round_state: "SupervisorRoundState"
    research_query: str = ""
    current_round: int = 0
    task_handles: dict[str, "asyncio.Task[ResearcherResult]"] = field(default_factory=dict)

    def rebuild_state(self, values: dict[str, Any]) -> None:
        """每次模型回合前从通道现值重建状态对象，替代上一批的内存对象。

        两个标量派生自入口 State 且不进通道，只由本 context 承载。
        """
        self.round_state = SupervisorRoundState.from_channel_values(
            values,
            research_query=self.research_query,
            current_round=self.current_round,
            active_evidence_limit=self.deps.config.supervisor_max_active_evidences,
            max_research_rounds=self.deps.config.max_research_rounds,
        )


class SupervisorRoundState:
    """Supervisor 工具循环的可变业务状态：证据聚合、active 工作集、task result、
    coverage/failure、research synthesis、working_set_revision、stop reason、熔断认证位、轮次。
    """

    def __init__(
        self,
        state: RunState,
        *,
        current_round: int,
        active_evidence_limit: int,
        max_research_rounds: int = 0,
    ):
        # 计划清单：工作台的一个键，轮内清单不进外层持久化；dict 形态存储（PlanItem.model_dump）。
        self.plan = [dict(item) for item in (state.get("plan") or []) if isinstance(item, dict)]
        self.max_research_rounds = max_research_rounds
        self.evidences = list(state.get("evidences", []))
        active_ids = state.get("active_evidence_ids")
        self.active_evidence_ids = (
            set(active_ids)
            if active_ids is not None
            else {item.evidence_id for item in self.evidences}
        )
        self.active_evidence_limit = active_evidence_limit
        self.source_refs = list(state.get("source_refs", []))
        self.task_results = list(state.get("task_results", []))
        # 当前轮次只来源于构造参数：由 _run_agent_loop 从持久化的 SupervisorProgress 推一次。
        self.current_round: int = current_round
        # 执行失败详情，如截断后的异常文本。只进收尾诊断文案，不渲染成研究缺口清单。
        self.failure_details: list[str] = []
        self.research_query = str(state.get("clarified_query", state.get("query", "")))
        self.working_set_revision = int(state.get("working_set_revision", 0) or 0)
        self.research_synthesis = self._restore_synthesis(state.get("research_synthesis"))
        self.completed_synthesis: ResearchSynthesis | None = None
        self.stop_reason: StopReason | None = None
        # 账户级熔断认证位：absorb 见到任何带该标志的任务结果即置真。这类故障是系统性
        # 不可用而非单个任务偶发，一个任务足以为全局认证。ResearchComplete 据此允许通过
        # 无证据结束，执行故障轮不再被迫记成预算耗尽。
        self.provider_exhausted = False
        # 在途任务登记 {task_id: PendingTask dict}；通道为事实源，批内共享对象。
        self.pending_tasks: dict[str, dict] = dict(state.get("pending_tasks") or {})

        # 编号由 _used_task_index 从持久化事实派生，不持有进程内副本。
        self._task_counter: int = self._used_task_index()

    def _used_task_index(self) -> int:
        """已用最大编号：已结算结果与在途登记的并集。

        编号只能从持久化事实派生：另存计数器副本会在跨轮重建时归零，与已发出的
        task_id 重号，后写的结果按 id 覆盖前一轮结论。取最大而非取计数，才能容纳
        恢复路径上的编号空洞。
        """
        used_indexes = [_integer_field(item, "task_index") for item in self.task_results]
        for entry in self.pending_tasks.values():
            used_indexes.append(_integer_field(pending_task_payload(entry), "sequence"))
        return max(used_indexes, default=0)

    @staticmethod
    def _restore_synthesis(value: object) -> ResearchSynthesis | None:
        if value is None:
            return None
        if isinstance(value, ResearchSynthesis):
            return value
        return ResearchSynthesis.model_validate(value)

    @classmethod
    def from_channel_values(
        cls,
        values: dict[str, Any],
        *,
        research_query: str,
        current_round: int,
        active_evidence_limit: int,
        max_research_rounds: int = 0,
    ) -> "SupervisorRoundState":
        """从内层通道现值重建状态对象：仅依赖标量参数与通道，不持有外层 State。

        缺席键回落到空态默认(等同尚无工具写入的新建状态)。
        """
        # 传新建空 dict：__init__ 会就地写入入参，复用模块级常量会被交叉污染。
        view = cls(
            {},
            current_round=current_round,
            active_evidence_limit=active_evidence_limit,
            max_research_rounds=max_research_rounds,
        )
        view.research_query = research_query
        if "evidences" in values:
            view.evidences = [
                item if isinstance(item, Evidence) else Evidence.model_validate(item)
                for item in values["evidences"]
            ]
        if "active_evidence_ids" in values:
            # 通道值即当前活跃集(merge_active_set 合并增量后的排序列表)，直读。
            view.active_evidence_ids = set(values["active_evidence_ids"])
        if "source_refs" in values:
            view.source_refs = list(values["source_refs"])
        if "task_results" in values:
            view.task_results = [
                item if isinstance(item, TaskResult) else TaskResult.model_validate(item)
                for item in values["task_results"]
            ]
        if "failure_details" in values:
            view.failure_details = list(values["failure_details"])
        if "working_set_revision" in values:
            view.working_set_revision = int(values["working_set_revision"] or 0)
        if "research_synthesis" in values:
            view.research_synthesis = cls._restore_synthesis(values["research_synthesis"])
        if "completed_synthesis" in values:
            view.completed_synthesis = cls._restore_synthesis(values["completed_synthesis"])
        if "provider_exhausted" in values:
            view.provider_exhausted = bool(values["provider_exhausted"])
        if values.get("stop_reason") is not None:
            view.stop_reason = StopReason(values["stop_reason"])
        if "task_index_counter" in values:
            view._task_counter = max(int(values["task_index_counter"] or 0), view._task_counter)
        if "pending_tasks" in values:
            view.pending_tasks = dict(values.get("pending_tasks") or {})
        if "plan" in values:
            view.plan = [dict(item) for item in values["plan"] or [] if isinstance(item, dict)]
        return view

    def initial_channel_state(self) -> dict[str, Any]:
        """构造 ainvoke 的入口 input：外层 State 资产一次性写入内层通道。"""
        return {
            "evidences": list(self.evidences),
            "active_evidence_ids": {"add": sorted(self.active_evidence_ids), "remove": []},
            "source_refs": list(self.source_refs),
            "task_results": list(self.task_results),
            "failure_details": list(self.failure_details),
            "working_set_revision": self.working_set_revision,
            "research_synthesis": self.research_synthesis,
            "completed_synthesis": self.completed_synthesis,
            "provider_exhausted": self.provider_exhausted,
            "stop_reason": self.stop_reason,
            "task_index_counter": self._task_counter,
            "plan": [dict(item) for item in self.plan],
        }

    @property
    def task_index_counter(self) -> int:
        """派发批出口随 Command 提交的派发编号(merge_monotonic，重复提交无害)。"""
        return self._task_counter

    def set_stop_reason(self, reason: StopReason) -> None:
        """按 `StopReason.rank` 采纳终止原因；低权威度不覆盖已采纳的高权威度。

        写点只需调用本方法并选对 reason；冲突由声明在 StopReason.rank 上的权威度
        裁决，与写入先后无关。
        """
        current = self.stop_reason
        if current is None or reason.rank >= current.rank:
            self.stop_reason = reason

    def _bump_working_set(self) -> None:
        """工作集版本前进的唯一出口。

        任何改变 active_evidence_ids 成员的操作都必须经此推进 revision，否则
        synthesis_is_fresh 会误判综合稿仍然新鲜。
        """
        self.working_set_revision += 1

    def allocate_task_index(self) -> int:
        """顺次分配下一个编号；调用位置不可得时回退到本路径。

        正常派发使用 allocate_task_index_for_call。两条路径都以已用最大编号为基准。
        """
        self._task_counter = max(self._task_counter, self._used_task_index()) + 1
        return self._task_counter

    def allocate_task_index_for_call(self, origin_state: dict[str, Any], tool_call_id: str) -> int:
        """任务序号由发起消息内的调用位置派生，不把研究轮次编码进任务 ID。"""
        position = _tool_call_position(origin_state.get("messages", []), tool_call_id)
        if position is None:
            return self.allocate_task_index()
        base = max(
            int(origin_state.get("task_index_counter", 0) or 0),
            self._used_task_index(),
        )
        index = base + position + 1
        self._task_counter = max(self._task_counter, index)
        return index

    def absorb(self, execution: ResearcherResult) -> None:
        """把一次任务研究结果并入工作状态。"""
        self.evidences.extend(execution.evidences)
        available = {item.evidence_id for item in execution.evidences}
        selected = [item for item in execution.selected_evidence_ids if item in available]
        slots = max(0, self.active_evidence_limit - len(self.active_evidence_ids))
        self.active_evidence_ids.update(selected[:slots])
        self.source_refs.extend(execution.source_refs)
        self.task_results.append(execution.task_result)
        if execution.task_result.provider_exhausted:
            self.provider_exhausted = True
        self._bump_working_set()

    def release_evidence(self, evidence_ids: list[str]) -> list[str]:
        """从 Supervisor 当前工作集释放 Evidence；全量档案仍保留。"""
        existing = self.active_evidence_ids.intersection(evidence_ids)
        self.active_evidence_ids.difference_update(existing)
        if existing:
            self._bump_working_set()
        return sorted(existing)

    def restore_evidence(self, evidence_ids: list[str]) -> list[str]:
        """从全局 Evidence 档案恢复候选，同时遵守活跃工作集上限。"""
        archive_ids = {item.evidence_id for item in self.evidences}
        candidates = [
            item
            for item in dict.fromkeys(evidence_ids)
            if item in archive_ids and item not in self.active_evidence_ids
        ]
        slots = max(0, self.active_evidence_limit - len(self.active_evidence_ids))
        restored = candidates[:slots]
        self.active_evidence_ids.update(restored)
        if restored:
            self._bump_working_set()
        return restored

    def synthesis_is_fresh(self, synthesis: ResearchSynthesis | None) -> bool:
        return bool(
            synthesis is not None
            and synthesis.based_on_working_set_revision == self.working_set_revision
        )

    def active_evidences(self) -> list[Evidence]:
        return [item for item in self.evidences if item.evidence_id in self.active_evidence_ids]


_WORKBENCH_CONCLUSION_CHARS = 2000
_WORKBENCH_RESERVE_CARD_LIMIT = 20


def workbench_view(round_state: "SupervisorRoundState") -> dict[str, object]:
    """Supervisor 工作台的唯一视图构造处。"""
    titles = _task_titles(round_state)
    active = round_state.active_evidences()
    tasks: list[dict[str, object]] = []
    for result in round_state.task_results:
        getter = (
            result.get
            if isinstance(result, dict)
            else (lambda key, item=result: getattr(item, key, None))
        )
        conclusion = _clip_conclusion(str(getter("conclusion") or ""))
        task_id = str(getter("task_id") or "")
        tasks.append(
            {
                "task_id": task_id,
                "display_title": str(getter("display_title") or titles.get(task_id, "")),
                "execution_status": str(getter("execution_status") or ""),
                "evidence_count": int(getter("evidence_count") or 0),
                "conclusion": conclusion,
            }
        )
    active_ids = {item.evidence_id for item in active}
    reserve = [item for item in round_state.evidences if item.evidence_id not in active_ids]
    rounds_left = max(0, round_state.max_research_rounds - round_state.current_round)
    return {
        "workbench_revision": round_state.working_set_revision,
        "plan": [dict(item) for item in round_state.plan],
        "inflight": [
            {
                "task_id": str(entry.get("task_id") or key),
                "display_title": str(entry.get("display_title") or ""),
                "round": int(entry.get("round", 0) or 0),
            }
            for key, entry in sorted(round_state.pending_tasks.items())
        ],
        "tasks": tasks,
        "active_evidence": [evidence_card(item, titles.get(item.task_id, "")) for item in active],
        "active_evidence_count": len(active),
        "active_evidence_limit": round_state.active_evidence_limit,
        "reserve_evidence": [
            evidence_card(item, titles.get(item.task_id, ""))
            for item in reserve[:_WORKBENCH_RESERVE_CARD_LIMIT]
        ],
        "reserve_evidence_count": len(reserve),
        "flags": {
            "provider_exhausted": bool(round_state.provider_exhausted),
            "rounds_left": rounds_left,
        },
        "synthesis_revision": (
            round_state.research_synthesis.revision if round_state.research_synthesis else 0
        ),
        "research_synthesis": synthesis_card(round_state.research_synthesis),
    }


def _task_titles(round_state: "SupervisorRoundState") -> dict[str, str]:
    """task_id → 任务短题；通道恢复态的 task_results 可能是 dict 形态，双形态取值。"""
    titles: dict[str, str] = {}
    for result in round_state.task_results:
        if isinstance(result, dict):
            task_id = str(result.get("task_id", ""))
            title = str(result.get("display_title", ""))
        else:
            task_id = str(getattr(result, "task_id", ""))
            title = str(getattr(result, "display_title", ""))
        if task_id and title:
            titles[task_id] = title
    return titles


def _clip_conclusion(text: str) -> str:
    """超长结论在句末断开；句末太靠前则退回定长截断，免得只剩半句开头。"""
    if len(text) <= _WORKBENCH_CONCLUSION_CHARS:
        return text
    window = text[:_WORKBENCH_CONCLUSION_CHARS]
    cut = max(window.rfind(mark) for mark in "。；！？")
    return window[: cut + 1] + "…" if cut >= _WORKBENCH_CONCLUSION_CHARS // 3 else window + "…"
