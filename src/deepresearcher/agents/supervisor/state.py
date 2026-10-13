"""Supervisor 的运行态辅助对象与内层 agent 的状态通道。"""

import asyncio
from dataclasses import dataclass, field
from typing import Annotated, Any

from langchain.agents import AgentState

from deepresearcher.agents.channels import (
    merge_active_set,
    merge_monotonic,
    merge_or,
    merge_registry_by_key,
    replace_channel,
    retain_if_absent,
)
from deepresearcher.agents.researcher import ResearchAgent
from deepresearcher.config import AgentConfig
from deepresearcher.evidence.models import Evidence
from deepresearcher.observability.events import AgentEmit
from deepresearcher.observability.execution import AgentExecutionScope
from deepresearcher.schemas import (
    DirectionStopReason,
    ResearchDirectionResult,
    ResearchSynthesis,
    StopReason,
    SupervisorProgress,
)
from deepresearcher.schemas.sources import source_domain
from deepresearcher.state import (
    ResearchState,
    SubTask,
    merge_evidences,
    merge_task_results,
    merge_unique,
    section,
)


def _merge_stop_reason_by_rank(
    current: StopReason | None, incoming: StopReason | None
) -> StopReason | None:
    if current is None:
        return incoming
    if incoming is None:
        return current
    return incoming if incoming.rank >= current.rank else current


def _tool_call_position(messages: object, tool_call_id: str) -> int | None:
    """在发起本次执行的 AIMessage 的 tool_calls 序列中定位该调用的下标。"""
    entries = messages if isinstance(messages, list) else []
    for message in reversed(entries):
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            continue
        for index, call in enumerate(tool_calls):
            if call.get("id") == tool_call_id:
                return index
        break
    return None


class SupervisorAgentState(AgentState):
    """Supervisor 内层图的 state_schema:循环业务状态的持久化载体。

    reducer 按工具的串行/并发身份选择,共用 reducer 见 agents/channels.py:
    ResearchDelegate 同批并发,只写合并语义通道;Release、Restore、Revise、
    ResearchComplete 串行,可写 replace 通道。active 工作集通道存集合本身,
    工具出口只交 {add,remove} 差量,避免并发整字段快照互相覆盖。
    """

    evidences: Annotated[list[Evidence], merge_evidences]
    active_evidence_ids: Annotated[list[str], merge_active_set]
    source_refs: Annotated[list[str], merge_unique]
    task_results: Annotated[list[ResearchDirectionResult], merge_task_results]
    coverage_gaps: Annotated[list[str], merge_unique]
    failure_details: Annotated[list[str], merge_unique]
    working_set_revision: Annotated[int, merge_monotonic]
    research_synthesis: Annotated[ResearchSynthesis | None, replace_channel]
    completed_synthesis: Annotated[ResearchSynthesis | None, retain_if_absent]
    sufficient: Annotated[bool, merge_or]
    provider_exhausted: Annotated[bool, merge_or]
    stop_reason: Annotated[StopReason | None, _merge_stop_reason_by_rank]
    task_index_counter: Annotated[int, merge_monotonic]
    # 在途方向登记表 {task_id: PendingDirection.model_dump()}:派发批并发 add,
    # 到达吸收与丢失判定在独占批 remove。崩溃重放时通道值是"哪些方向欠账"的唯一事实源。
    pending_directions: Annotated[dict[str, dict], merge_registry_by_key]


# 老检查点的方向结果没有短题时,卡片回退用长契约截断到此长度。
_DIRECTION_FALLBACK_CHARS = 30


def _direction_label(evidence: Evidence, direction_title: str) -> str:
    """方向归属展示名：派发短题优先，无短题回退长契约截断。"""
    if direction_title:
        return direction_title
    topic = evidence.research_direction
    if len(topic) > _DIRECTION_FALLBACK_CHARS:
        return topic[: _DIRECTION_FALLBACK_CHARS - 1] + "…"
    return topic


def evidence_card(evidence: Evidence, direction_title: str = "") -> dict[str, object]:
    """Supervisor 的唯一 Evidence 观察视图；不暴露 quote 与审计正文。

    来源画像展平为顶层键（与 writer/reviewer 卡同形）；方向归属只带短题，
    完整契约在派发记录与方向报告里本来就有，不在每张卡上重复。
    """
    profile = evidence.source_profile
    return {
        "evidence_id": evidence.evidence_id,
        "claim": evidence.claim,
        "direction_title": _direction_label(evidence, direction_title),
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
    """ResearchSupervisor 构造期的稳定零件;跨并发 loop 共享、只读,frozen 数据类保证这一点。"""

    config: AgentConfig
    research_agent: ResearchAgent
    worker_limit: asyncio.Semaphore
    emit: AgentEmit


@dataclass
class SupervisorLoopContext:
    """Supervisor 一次工具 loop 的注入载荷。

    loop_state 是当前工具批的可变状态:每次模型回合前由 rebuild_loop_state 从通道重建,
    批内工具共享同一对象,工具出口把改动作为 Command 增量写回通道;它不进 checkpoint,
    持久状态以通道为准。research_query、current_round、active_evidence_limit 是重建
    所需的标量。业务实现在 services.py,tools.py 从本对象转交参数。

    消费方含中间件,删改字段前先 grep agents/middleware/:
      scope → observability:事件归因
    """

    scope: AgentExecutionScope
    deps: SupervisorDeps
    loop_state: "SupervisorLoopState"
    research_query: str = ""
    current_round: int = 0
    # 派发出去的在途方向后台任务 {task_id: Task[TaskExecution]};进程级,不进通道不进 checkpoint。
    # "哪些方向欠账"的事实源是 loop_state.pending_directions(通道),这里只是执行载体。
    direction_tasks: dict[str, "asyncio.Task[TaskExecution]"] = field(default_factory=dict)
    # ResearchComplete 因在途方向被拒后置真:下一个模型回合前等待一个到达而非空转。
    arrival_wait_requested: bool = False

    def rebuild_loop_state(self, values: dict[str, Any]) -> None:
        """每次模型回合前从通道现值重建状态对象,替代上一批的内存对象。

        外层 State 的业务资产已在入口经 initial_channel_state 种进通道;重建的两个
        标量派生自入口 State 且不进通道,只由本 context 承载,上限直接从 deps.config
        现取。
        """
        self.loop_state = SupervisorLoopState.from_channel_values(
            values,
            research_query=self.research_query,
            current_round=self.current_round,
            active_evidence_limit=self.deps.config.supervisor_max_active_evidences,
        )


@dataclass
class TaskExecution:
    """一次方向研究执行的完整产物：结果模型与 absorb 所需的 Evidence/来源清单。

    模型看到的方向报告由 services.absorb_arrival 在本对象产物上渲染,
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
            display_title=str(task.get("display_title") or ""),
            execution_status="failed",
            coverage_status="insufficient",
            evidence_count=0,
            source_count=0,
            failures=[error],
            stop_reason=DirectionStopReason.WORKER_EXCEPTION,
            stop_detail=error,
        )
        return cls(
            task_result=task_result,
            evidences=[],
            selected_evidence_ids=[],
            source_refs=[],
        )


class SupervisorLoopState:
    """Supervisor 工具循环的可变业务状态:证据聚合、active 工作集、task result、
    coverage/failure、research synthesis、working_set_revision、stop reason、熔断认证位、轮次。

    持久化由 SupervisorAgentState 通道承担:每次模型回合前从通道重建,工具批内原地改、
    出口经 Command 增量写回通道,loop 结束后由 SupervisorStateUpdate 转成顶层 State 增量。
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
        self.active_evidence_limit = active_evidence_limit
        self.source_refs = list(state.get("source_refs", []))
        self.task_results = list(state.get("task_results", []))
        # 当前轮次只来源于构造参数:由 _run_agent_loop 从持久化的 SupervisorProgress 推一次。
        # 这里读 SupervisorProgress 只为取持久化的 coverage_gaps,不决定轮次。
        persisted = section(state, "supervisor", SupervisorProgress)
        self.current_round: int = current_round
        self.coverage_gaps = list(persisted.coverage_gaps)
        # 执行失败详情,如截断后的异常文本。coverage_gaps 只装研究内容缺口,会渲染进用户
        # 报告的"未闭合缺口";两者的可见面与恢复策略不同,不能混用同一个字段。
        self.failure_details: list[str] = []
        self.research_query = str(state.get("clarified_query", state.get("query", "")))
        self.sufficient = False
        self.working_set_revision = int(state.get("working_set_revision", 0) or 0)
        self.research_synthesis = self._restore_synthesis(state.get("research_synthesis"))
        self.completed_synthesis: ResearchSynthesis | None = None
        self.stop_reason: StopReason | None = None
        # 账户级熔断认证位:absorb 见到任何带该标志的方向结果即置真。这类故障是系统性
        # 不可用而非单个方向偶发,一个方向足以为全局认证。ResearchComplete 据此放行
        # 无证据收束,执行故障轮不再被迫记成预算耗尽。
        self.provider_exhausted = False
        # 在途方向登记 {task_id: PendingDirection dict};通道为事实源,批内共享对象。
        self.pending_directions: dict[str, dict] = dict(state.get("pending_directions") or {})

        self._task_counter: int = 0

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
    ) -> "SupervisorLoopState":
        """从内层通道现值重建状态对象:仅依赖三个标量与通道,不持有外层 State。

        缺席键回落到空态默认(等同尚无工具写入的新建状态)。
        """
        # 传新建空 dict:__init__ 里的 section(...) 会就地写入入参,复用常量会被交叉污染。
        view = cls(
            {},
            current_round=current_round,
            active_evidence_limit=active_evidence_limit,
        )
        view.research_query = research_query
        if "evidences" in values:
            view.evidences = [
                item if isinstance(item, Evidence) else Evidence.model_validate(item)
                for item in values["evidences"]
            ]
        if "active_evidence_ids" in values:
            # 通道值即当前活跃集(merge_active_set 合并差量后的排序列表),直读。
            view.active_evidence_ids = set(values["active_evidence_ids"])
        if "source_refs" in values:
            view.source_refs = list(values["source_refs"])
        if "task_results" in values:
            view.task_results = [
                item
                if isinstance(item, ResearchDirectionResult)
                else ResearchDirectionResult.model_validate(item)
                for item in values["task_results"]
            ]
        if "coverage_gaps" in values:
            view.coverage_gaps = list(values["coverage_gaps"])
        if "failure_details" in values:
            view.failure_details = list(values["failure_details"])
        if "working_set_revision" in values:
            view.working_set_revision = int(values["working_set_revision"] or 0)
        if "research_synthesis" in values:
            view.research_synthesis = cls._restore_synthesis(values["research_synthesis"])
        if "completed_synthesis" in values:
            view.completed_synthesis = cls._restore_synthesis(values["completed_synthesis"])
        if "sufficient" in values:
            view.sufficient = bool(values["sufficient"])
        if "provider_exhausted" in values:
            view.provider_exhausted = bool(values["provider_exhausted"])
        if values.get("stop_reason") is not None:
            view.stop_reason = StopReason(values["stop_reason"])
        if "task_index_counter" in values:
            view._task_counter = max(int(values["task_index_counter"] or 0), view._task_counter)
        if "pending_directions" in values:
            view.pending_directions = dict(values["pending_directions"] or {})
        return view

    def initial_channel_state(self) -> dict[str, Any]:
        """构造 ainvoke 的入口 input:外层 State 资产一次性种进内层通道。"""
        return {
            "evidences": list(self.evidences),
            "active_evidence_ids": {"add": sorted(self.active_evidence_ids), "remove": []},
            "source_refs": list(self.source_refs),
            "task_results": list(self.task_results),
            "coverage_gaps": list(self.coverage_gaps),
            "failure_details": list(self.failure_details),
            "working_set_revision": self.working_set_revision,
            "research_synthesis": self.research_synthesis,
            "completed_synthesis": self.completed_synthesis,
            "sufficient": self.sufficient,
            "provider_exhausted": self.provider_exhausted,
            "stop_reason": self.stop_reason,
            "task_index_counter": self._task_counter,
        }

    @property
    def task_index_counter(self) -> int:
        """派发批出口随 Command 提交的编号水位(merge_monotonic,重复提交无害)。"""
        return self._task_counter

    def set_stop_reason(self, reason: StopReason) -> None:
        """按 `StopReason.rank` 采纳终止原因；低权威度不覆盖已采纳的高权威度。

        写点只需调用本方法并选对 reason;冲突由声明在 StopReason.rank 上的权威度
        裁决,与写入先后无关。
        """
        current = self.stop_reason
        if current is None or reason.rank >= current.rank:
            self.stop_reason = reason

    def _bump_working_set(self) -> None:
        """工作集版本前进的唯一出口。

        不变式:任何改变 `active_evidence_ids` 成员的操作都必须经此推进 revision,
        否则 `synthesis_is_fresh` 会误判综合稿仍然新鲜,放行基于过期证据的
        ResearchComplete。新增 mutation 方法时必须调用它:revision 的单调性
        无法由类型系统表达,只能靠此约定维持。
        """
        self.working_set_revision += 1

    def allocate_task_index(self) -> int:
        """活视图顺次分配;仅在调用位置不可得时兜底。

        正常派发走 allocate_task_index_for_call。此路径依赖批内共享对象,
        崩溃重放时不同生命周期会算出相同编号,merge_task_results 按 id
        去重会吞掉一个方向的研究结果,因此只能作为无持久位置信息时的退路。
        """
        completed_max = max((item.task_index for item in self.task_results), default=0)
        self._task_counter = max(self._task_counter, completed_max) + 1
        return self._task_counter

    def allocate_task_index_for_call(self, origin_state: dict[str, Any], tool_call_id: str) -> int:
        """任务序号由发起消息内的调用位置派生,不把研究轮次编码进任务 ID。

        批初 task_index_counter 与 AIMessage 的 tool_calls 顺序都已在通道里持久化:
        同一 tool_call 在任何生命周期重放到同一编号,journal 命中与重跑各得各位,
        嵌套 researcher 的 ns(键含 task id)因此跨重放稳定。
        """
        position = _tool_call_position(origin_state.get("messages", []), tool_call_id)
        if position is None:
            return self.allocate_task_index()
        base = int(origin_state.get("task_index_counter", 0) or 0)
        index = base + position + 1
        self._task_counter = max(self._task_counter, index)
        return index

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
        if execution.task_result.provider_exhausted:
            self.provider_exhausted = True
        self.coverage_gaps.extend(execution.task_result.remaining_gaps)
        self.coverage_gaps = list(dict.fromkeys(gap for gap in self.coverage_gaps if gap.strip()))
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


def working_set_snapshot(
    loop_state: "SupervisorLoopState", *, include_reserve: bool = True
) -> dict[str, object]:
    """Supervisor 工作集目录的唯一构造处。

    tools.py 的 ReadWorkingSet 与拒绝回执用全量,即 include_reserve=True;
    轮次观察只取 active 以省 token。两处共用同一卡片形状与 revision 字段,
    不要各自演化出不同的子集。
    """
    titles = _direction_titles(loop_state)
    active = loop_state.active_evidences()
    snapshot: dict[str, object] = {
        "working_set_revision": loop_state.working_set_revision,
        "active_evidence": [
            evidence_card(item, titles.get(item.subtask_id, "")) for item in active
        ],
        "active_evidence_count": len(active),
        "active_evidence_limit": loop_state.active_evidence_limit,
        # 在途方向清单恒有键:派发去重与收口判断都要求模型每回合看得到"还在跑什么"。
        "inflight": [
            {
                "task_id": str(entry.get("task_id") or key),
                "display_title": str(entry.get("display_title") or ""),
                "round": int(entry.get("round", 0) or 0),
            }
            for key, entry in sorted(loop_state.pending_directions.items())
        ],
    }
    if include_reserve:
        active_ids = {item.evidence_id for item in active}
        reserve = [item for item in loop_state.evidences if item.evidence_id not in active_ids]
        snapshot["reserve_evidence"] = [
            evidence_card(item, titles.get(item.subtask_id, "")) for item in reserve
        ]
        snapshot["reserve_evidence_count"] = len(reserve)
    return snapshot


def _direction_titles(loop_state: "SupervisorLoopState") -> dict[str, str]:
    """task_id → 方向短题；通道恢复态的 task_results 可能是 dict 形态，双形态取值。"""
    titles: dict[str, str] = {}
    for result in loop_state.task_results:
        if isinstance(result, dict):
            task_id = str(result.get("task_id", ""))
            title = str(result.get("display_title", ""))
        else:
            task_id = str(getattr(result, "task_id", ""))
            title = str(getattr(result, "display_title", ""))
        if task_id and title:
            titles[task_id] = title
    return titles
