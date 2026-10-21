"""Supervisor 业务实现：不感知 langgraph 的纯函数，async 与否按是否需要等待。

工具回执的字符串协议与消息配对留在协议层 tools.py，本模块只回 dict，不构造
ToolMessage。子 Agent 并发受 deps.worker_limit 限制；对 loop_state 的改写全部收在
不含 await 的同步段内，原子性由事件循环的单线程调度保证。
"""

import asyncio
from dataclasses import dataclass, field
from typing import cast

from pydantic import ValidationError

from deepresearcher.agents.supervisor.state import (
    SupervisorDeps,
    SupervisorLoopState,
    TaskExecution,
    evidence_card,
    synthesis_snapshot,
)
from deepresearcher.config import AgentConfig
from deepresearcher.evidence.models import Evidence
from deepresearcher.observability.events.names import EventName
from deepresearcher.observability.execution import AgentExecutionScope
from deepresearcher.routing import NodeName
from deepresearcher.schemas import (
    WORKBENCH_INTENT_ADAPTER,
    AspectIssue,
    PendingDirection,
    ReportBrief,
    ReportChapter,
    ResearchAgentResult,
    ResearchAspect,
    ResearchDirectionResult,
    ResearchSynthesis,
    ReviewProgress,
    RunStatus,
    StopReason,
    SupervisorProgress,
    SupervisorStateUpdate,
    WorkbenchAcknowledge,
    WorkbenchIntent,
    WorkbenchMoveEvidence,
    WorkbenchRemoveAspect,
    WorkbenchSetPlan,
    WorkbenchUpdateFraming,
    WorkbenchUpsertAspect,
    WriterDirective,
    WriterProgress,
)
from deepresearcher.schemas.limits import (
    EVIDENCE_REFERENCES_HARD_LIMIT,
    STRUCTURED_SUMMARY_HARD_LIMIT_CHARS,
)
from deepresearcher.state import ResearchState, SubTask, section

# 最小固定版综合稿里拼接的断言条数上限。
_FALLBACK_SUMMARY_CLAIM_LIMIT = 6


def dispatch_research(
    deps: SupervisorDeps,
    loop_state: SupervisorLoopState,
    scope: AgentExecutionScope,
    topic: str,
    display_title: str = "",
    *,
    tool_call_id: str = "",
    origin_state: dict[str, object] | None = None,
) -> tuple[SubTask | None, dict[str, object], dict[str, object]]:
    """受理一次 ResearchDelegate 派发请求：预算硬校验、任务编号、构造 SubTask。

    返回 (task, receipt, entry)；预算越界时 task 为 None。entry 是写进
    pending_directions 通道的登记项，与回执同批提交。方向本体由 tools 层后台执行，
    完成后经 absorb_arrival 吸收，派发批不等方向完成。
    """
    round_no = loop_state.current_round
    deps.emit(EventName.DELEGATE_STARTED, {"topic_chars": len(topic)})
    if round_no > deps.config.max_research_rounds:
        loop_state.set_stop_reason(StopReason.GLOBAL_ROUND_BUDGET_EXHAUSTED)
        blocked: dict[str, object] = {
            "status": "blocked",
            "reason": "round_budget_exhausted",
            "instruction": "研究轮次预算已耗尽；请修订最新研究综合稿。若达到完整标准则调用 ResearchComplete，否则直接结束，系统将按 partial 交付。",
        }
        # 被预算拒绝的调用不产生方向，即时补全 delegate 事件对，让无产出轮次在事件流可解释。
        deps.emit(
            EventName.DELEGATE_COMPLETED,
            {
                "status": "blocked",
                "reason": "round_budget_exhausted",
                "topic_chars": len(topic),
                "evidence_count": None,
                "source_count": None,
            },
        )
        return None, blocked, {}
    if tool_call_id and origin_state is not None:
        task_index = loop_state.allocate_task_index_for_call(origin_state, tool_call_id)
    else:
        task_index = loop_state.allocate_task_index()
    task: SubTask = {
        "id": f"task-{task_index:04d}",
        "run_id": scope.run_id,
        "question": topic,
        "display_title": display_title.strip(),
        "round": round_no,
        "sequence": task_index,
        "worker_id": f"research-agent-{task_index:04d}",
        "worker_index": task_index,
        "parent_task_id": "",
        "operation_id": f"research-task-{task_index:04d}",
    }
    entry = PendingDirection(
        task_id=task["id"],
        subtask=dict(task),
        display_title=task["display_title"],
        round=round_no,
    ).model_dump()
    receipt: dict[str, object] = {
        "status": "dispatched",
        "task_id": task["id"],
        "display_title": task["display_title"],
        "research_direction": topic,
        "workbench_revision": loop_state.working_set_revision,
        "instruction": (
            "方向已入队后台执行，本回执只代表受理。请勿对在途主题重复派发；"
            "可继续派发互补方向、修订综合稿，完成的结果将在回合开头注入，"
            "本场内主动等待用 WaitDirection。"
        ),
    }
    return task, receipt, entry


async def _execute_research_task(deps: SupervisorDeps, task: SubTask) -> TaskExecution:
    """后台任务体：等并发槽、执行单个方向研究，返回完整产物。

    除取消外不抛异常——worker 异常在此降级为 failed 结果，保证 await 方总能取值。
    完成/失败事件不在这里发："完成"以归入工作集的时刻为准，见 absorb_arrival。
    """
    round_no = int(task.get("round", 1))
    async with deps.worker_limit:
        deps.emit(
            EventName.RESEARCH_TASK_STARTED,
            {
                **_task_context(deps, task),
                "task_index": int(task.get("sequence", 0)),
                "component": "research_agent",
                "title": str(task.get("display_title") or ""),
                "question": task["question"][: deps.config.supervisor_preview_chars],
            },
        )
        try:
            result = await deps.research_agent.run(task)
            # 子 Agent 边界：结果写入审计事件或 State 前先过契约校验；
            # 同 run 内已是实例时不花校验开销，跨进程 checkpoint 恢复的 dict 由校验重建。
            agent_result = (
                result
                if isinstance(result, ResearchAgentResult)
                else ResearchAgentResult.model_validate(result)
            )
            task_result = agent_result.task_result
            return TaskExecution(
                task_result=task_result,
                evidences=list(agent_result.evidences),
                selected_evidence_ids=list(agent_result.selected_evidence_ids),
                source_refs=list(agent_result.source_refs),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return TaskExecution.failed_for(
                task,
                round_no,
                str(exc)[: deps.config.supervisor_preview_chars],
            )


def absorb_lost_registrations(
    deps: SupervisorDeps,
    loop_state: SupervisorLoopState,
    live_task_ids: set[str],
) -> list[tuple[dict[str, object], dict[str, object]]]:
    """崩溃恢复结算：通道有登记、进程无活任务体的方向一律判 failed 并移除登记。

    WaitDirection 在消费前先调用本函数，把未归属登记结算成收据；节点收尾 drain 在
    模型 nudge 上限用尽仍直接结束时以此作为回退。两处都必须调用，只在一处结算会漏掉
    另一处，在途方向就无声消失。返回逐项的方向报告与通道增量。
    """
    results: list[tuple[dict[str, object], dict[str, object]]] = []
    for task_id in sorted(list(loop_state.pending_directions)):
        if task_id in live_task_ids:
            continue
        entry = loop_state.pending_directions.get(task_id) or {}
        subtask = dict(entry.get("subtask") or {})
        subtask.setdefault("id", task_id)
        subtask.setdefault("question", str(subtask.get("display_title") or ""))
        typed_subtask = cast(SubTask, subtask)
        execution = TaskExecution.failed_for(
            typed_subtask,
            int(entry.get("round") or loop_state.current_round),
            "进程重启，在途方向丢失；主题仍缺口时按需重新派发。",
        )
        results.append(
            absorb_arrival(deps, loop_state, typed_subtask, execution, merged_via="lost")
        )
    return results


def _merge_channel_delta(delta: dict[str, object], part: dict[str, object]) -> None:
    """把逐项吸收增量并入总增量；两个增量通道按各自形态合并，其余键后写覆盖(值同源幂等)。"""
    for key, value in part.items():
        if key in ("active_evidence_ids", "pending_directions"):
            if key not in delta:
                delta[key] = {
                    "add": {} if key == "pending_directions" else [],
                    "remove": [],
                }
            existing = delta[key]
            assert isinstance(existing, dict)
            incoming = value if isinstance(value, dict) else {}
            add_key = "add" if key == "pending_directions" else "add"
            if key == "pending_directions":
                merged_add = existing.get(add_key)
                assert isinstance(merged_add, dict)
                add_update = incoming.get(add_key) or {}
                assert isinstance(add_update, dict)
                merged_add.update(add_update)
            else:
                merged_add_list = existing.get(add_key)
                assert isinstance(merged_add_list, list)
                existing[add_key] = sorted(
                    set(merged_add_list) | set((incoming.get(add_key) or []))
                )
            merged_remove = existing.get("remove")
            assert isinstance(merged_remove, list)
            existing["remove"] = sorted(set(merged_remove) | set(incoming.get("remove") or []))
        else:
            delta[key] = value


def receive_ready_directions(
    deps: SupervisorDeps,
    loop_state: SupervisorLoopState,
    registry: dict[str, asyncio.Task],
    *,
    merged_via: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """非阻塞接收：结算未归属登记，并吸收登记表中全部已完成且未领取的方向。

    进程内登记表弹出即领取标记，回合开头接收中间件与 WaitDirection 两个入口以它
    互斥：先执行的一方消费，后执行的一方只见剩余。报告与通道增量一并返回，增量
    只含 merge 语义通道。吸收原语按 id 幂等，两个入口不会重复计入。
    """
    reports: list[dict[str, object]] = []
    delta: dict[str, object] = {}
    for report, lost_delta in absorb_lost_registrations(deps, loop_state, set(registry)):
        reports.append(report)
        _merge_channel_delta(delta, lost_delta)

    for task_id in sorted(list(registry)):
        task = registry[task_id]
        if not task.done():
            continue
        execution = task.result()
        entry = loop_state.pending_directions.get(task_id) or {}
        subtask = cast(SubTask, dict(entry.get("subtask") or {"id": task_id, "question": ""}))
        registry.pop(task_id)
        report, task_delta = absorb_arrival(
            deps, loop_state, subtask, execution, merged_via=merged_via
        )
        reports.append(report)
        _merge_channel_delta(delta, task_delta)
    return reports, delta


async def wait_for_directions(
    deps: SupervisorDeps,
    loop_state: SupervisorLoopState,
    registry: dict[str, asyncio.Task],
    *,
    task_ids: list[str] | None = None,
    timeout: float | None = None,
) -> tuple[list[dict[str, object]], dict[str, object], bool]:
    """等待并收取在途方向：先非阻塞接收，无结果才阻塞。

    已有完成未领取的立即返回。timeout 到期无一完成时 timed_out 为真。任务体除取消外
    不抛；指定 id 全部不在途时以现有结果即返。
    """
    reports, delta = receive_ready_directions(deps, loop_state, registry, merged_via="wait_tool")

    targets = (
        [registry[tid] for tid in task_ids if tid in registry]
        if task_ids
        else list(registry.values())
    )
    live = [item for item in targets if not item.done()]
    timed_out = False
    if not reports and live:
        await asyncio.wait(live, return_when=asyncio.FIRST_COMPLETED, timeout=timeout)
        timed_out = timeout is not None and not any(item.done() for item in live)

    batch_reports, batch_delta = receive_ready_directions(
        deps, loop_state, registry, merged_via="wait_tool"
    )
    reports.extend(batch_reports)
    _merge_channel_delta(delta, batch_delta)
    if batch_reports:
        timed_out = False
    return reports, delta, timed_out


def compose_wait_receipt(
    reports: list[dict[str, object]],
    pending_task_ids: set[str],
    working_set_revision: int,
    *,
    timed_out: bool = False,
) -> dict[str, object]:
    """WaitDirection 回执：等待到期回 timed_out，不与无在途情形混同。

    无报告但在途非空时，只可能是指定 id 全不在册或等待到期，
    两种情形分别给出纠偏指令，避免回执自相矛盾。
    """
    if reports:
        status = "completed"
    elif pending_task_ids and timed_out:
        status = "timed_out"
    elif pending_task_ids:
        status = "unknown_task_ids"
    else:
        status = "no_inflight"
    payload: dict[str, object] = {
        "status": status,
        "directions": reports,
        "workbench_revision": working_set_revision,
        "inflight": sorted(pending_task_ids),
    }
    if status == "unknown_task_ids":
        payload["instruction"] = (
            "指定的 task_ids 没有一个在途(以本回执 inflight 为准)。核对 id 后重试，"
            "或留空 task_ids 等待任意一个方向完成。"
        )
    elif status == "timed_out":
        payload["instruction"] = (
            "等待时间已用尽，期间没有方向完成。可加长 timeout 再等，或先处理现有工作台。"
        )
    elif status == "no_inflight":
        payload["instruction"] = (
            "没有在途方向。请按工作台现值修订综合稿后 ResearchComplete，或继续派发补缺口。"
        )
    return payload


def _task_context(deps: SupervisorDeps, task: SubTask) -> dict[str, object]:
    return {
        "task_id": task["id"],
        "worker_id": task.get("worker_id", task["id"]),
        "worker_index": int(task.get("worker_index", task.get("sequence", 0))),
        "parent_task_id": task.get("parent_task_id", ""),
        "operation_id": task.get("operation_id", task["id"]),
        "concurrency_limit": deps.config.max_parallel_workers,
    }


def absorb_arrival(
    deps: SupervisorDeps,
    loop_state: SupervisorLoopState,
    task: SubTask,
    execution: TaskExecution,
    *,
    merged_via: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """方向完成后的同步吸收段：并入工作集、移除登记、发事件，返回报告与通道增量。

    调用方是 WaitDirection 与节点收尾 drain，都在循环线程上同步执行，段内禁 await。
    merged_via 记录吸收入口：model_turn、wait_tool、drain 或 lost。
    """
    task_context = _task_context(deps, task)
    task_result = execution.task_result
    active_before = set(loop_state.active_evidence_ids)
    loop_state.absorb(execution)
    loop_state.pending_directions.pop(task["id"], None)
    if task_result.execution_status == "failed":
        deps.emit(
            EventName.RESEARCH_TASK_FAILED,
            {**task_context, **task_result.model_dump()},
            component="research_agent",
        )
    else:
        deps.emit(
            EventName.RESEARCH_TASK_COMPLETED,
            {**task_context, **task_result.model_dump()},
            component="research_agent",
        )
    deps.emit(
        EventName.DIRECTION_ARRIVAL_MERGED,
        {
            **task_context,
            "display_title": task_result.display_title,
            "execution_status": task_result.execution_status,
            "evidence_count": task_result.evidence_count,
            "running_task_ids": sorted(loop_state.pending_directions),
            "merged_via": merged_via,
        },
    )
    deps.emit(
        EventName.DELEGATE_COMPLETED,
        {
            "status": task_result.execution_status,
            "reason": task_result.stop_reason.value,
            "topic_chars": len(task["question"]),
            "evidence_count": task_result.evidence_count,
            "source_count": task_result.source_count,
        },
    )
    direction_report: dict[str, object] = {
        "task_id": task_result.task_id,
        "status": task_result.execution_status,
        "research_direction": task_result.research_direction,
        "coverage_status": task_result.coverage_status,
        "evidence_count": task_result.evidence_count,
        "source_count": task_result.source_count,
        "remaining_gaps": task_result.remaining_gaps,
        "conclusion": task_result.conclusion,
        "failures": task_result.failures,
        "workbench_revision": loop_state.working_set_revision,
        "evidence": [
            evidence_card(item, task_result.display_title)
            for item in execution.evidences
            if item.evidence_id in loop_state.active_evidence_ids
        ],
    }
    if task_result.provider_exhausted:
        # 只给数据、不加控制流分支：Supervisor 模型读到系统性不可用后自然停止派发、收尾。
        direction_report["provider_exhausted"] = True
        direction_report["instruction"] = (
            "搜索服务账户级不可用（额度耗尽/密钥无效），系统性问题：再派新方向也会同样失败。"
            "停止派发 ResearchDelegate；把已有 Evidence 修订进综合稿，随后调用 ResearchComplete；"
            "若不足以成文则直接结束本轮，系统按 partial 交付。"
        )
    delta: dict[str, object] = {
        "evidences": list(loop_state.evidences),
        "active_evidence_ids": {
            "add": sorted(loop_state.active_evidence_ids - active_before),
            "remove": [],
        },
        "source_refs": list(loop_state.source_refs),
        "task_results": list(loop_state.task_results),
        "coverage_gaps": list(loop_state.coverage_gaps),
        "failure_details": list(loop_state.failure_details),
        "working_set_revision": loop_state.working_set_revision,
        "provider_exhausted": loop_state.provider_exhausted,
        "stop_reason": loop_state.stop_reason,
        "pending_directions": {"add": {}, "remove": [task["id"]]},
    }
    return direction_report, delta


def _candidate_synthesis(
    loop_state: SupervisorLoopState,
    framing_updates: dict[str, object],
    aspect_upserts: dict[str, ResearchAspect],
    aspect_removals: list[str],
) -> ResearchSynthesis | None:
    """把 framing 与 aspect 变更套到现稿上，产出候选稿；纯函数不写入磁盘。

    aspect 顺序：沿用现稿次序，同 id upsert 原位替换，新 id 追加。
    草稿期不要求任何字段齐备；变更后全空（无 aspect、三件与声明皆空）返回 None，
    调用方据此把稿子清回未开稿状态，清空草稿重新开始是合法动作。
    """
    current = loop_state.research_synthesis
    aspects: dict[str, ResearchAspect] = {}
    if current is not None:
        aspects = {aspect.aspect_id: aspect for aspect in current.aspects}
        order = [aspect.aspect_id for aspect in current.aspects]
    else:
        order = []
    for aspect_id in aspect_removals:
        if aspect_id in aspects:
            del aspects[aspect_id]
            order.remove(aspect_id)
    for aspect_id, aspect in aspect_upserts.items():
        if aspect_id not in aspects:
            order.append(aspect_id)
        aspects[aspect_id] = aspect
    payload: dict[str, object] = {
        "answer_goal": current.answer_goal if current else "",
        "thesis": current.thesis if current else "",
        "decision_rationale": current.decision_rationale if current else "",
    }
    payload.update(framing_updates)

    everything_empty = not aspects and all(
        payload.get(key) in (None, "") for key in ("answer_goal", "thesis", "decision_rationale")
    )
    if everything_empty:
        return None

    return ResearchSynthesis(
        revision=(current.revision if current else 0) + 1,
        based_on_working_set_revision=loop_state.working_set_revision,
        answer_goal=str(payload["answer_goal"] or ""),
        thesis=str(payload["thesis"] or ""),
        aspects=[aspects[aspect_id] for aspect_id in order],
        decision_rationale=str(payload["decision_rationale"] or ""),
    )


def _rejection(op_name: str, reason: str, **extra: object) -> dict[str, object]:
    return {"status": "rejected", "rejections": [{"key": op_name, "reason": reason, **extra}]}


_WRITABLE_OPS_HINT = (
    "不可写或未知的操作。可写操作：upsert_aspect | remove_aspect | framing | plan | "
    "evidence | acknowledge。只读段（directions/inflight/reserve/flags）由程序维护。"
)


@dataclass
class _SubmissionEdits:
    """一次 UpdateWorkbench 提交归并后的编辑集：各段候选输入，尚未写入磁盘。"""

    framing_updates: dict[str, object] = field(default_factory=dict)
    aspect_upserts: dict[str, ResearchAspect] = field(default_factory=dict)
    aspect_removals: list[str] = field(default_factory=list)
    plan_value: list[dict[str, object]] | None = None
    evidence_add: list[str] = field(default_factory=list)
    evidence_remove: list[str] = field(default_factory=list)
    acknowledge: bool = False

    @property
    def edits_draft(self) -> bool:
        return bool(self.framing_updates or self.aspect_upserts or self.aspect_removals)


def _intent_identity(intent: WorkbenchIntent) -> str:
    """去重身份按目标划：aspect 操作到单元粒度，其余 op 全局唯一。"""
    if isinstance(intent, WorkbenchUpsertAspect):
        return f"upsert_aspect:{intent.aspect.aspect_id}"
    if isinstance(intent, WorkbenchRemoveAspect):
        return f"remove_aspect:{intent.aspect_id}"
    return intent.op


_TAG_ERROR_TYPES = frozenset({"union_tag_invalid", "union_tag_not_found"})


def _project_submission(
    loop_state: SupervisorLoopState, changes: list[object]
) -> tuple[_SubmissionEdits, list[dict[str, object]]]:
    """判别并集解析 → 目标去重 → 逐条投影；返回(编辑集， 拒绝项)。"""
    edits = _SubmissionEdits()
    rejections: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in changes:
        if not isinstance(raw, dict):
            rejections.append(
                {
                    "key": "changes",
                    "reason": f"每个操作必须是带 op 字段的对象。{_WRITABLE_OPS_HINT}",
                }
            )
            continue
        try:
            intent = WORKBENCH_INTENT_ADAPTER.validate_python(raw)
        except ValidationError as exc:
            op = str(raw.get("op", ""))
            errors = exc.errors(include_url=False, include_input=False)
            if any(error["type"] in _TAG_ERROR_TYPES for error in errors):
                rejections.append({"key": op, "reason": _WRITABLE_OPS_HINT})
            else:
                rejections.append(
                    {
                        "key": op or "changes",
                        "reason": f"op={op} 的参数不满足该操作的契约。",
                        "issues": _issue_list(exc),
                    }
                )
            continue
        identity = _intent_identity(intent)
        if identity in seen:
            rejections.append(
                {
                    "key": identity,
                    "reason": "同一目标在一次提交内只允许一条操作；合并两次意图后重发。",
                }
            )
            continue
        seen.add(identity)
        problem = _project_intent(edits, loop_state, intent)
        if problem is not None:
            rejections.append(problem)
    return edits, rejections


def _project_intent(
    edits: _SubmissionEdits, loop_state: SupervisorLoopState, intent: WorkbenchIntent
) -> dict[str, object] | None:
    """把一条结构已合法的 intent 归并进编辑集；只保留存在性前置(删除目标须在、
    add/remove 不得同 id)，终态一致性在提交前统一判。"""
    if isinstance(intent, WorkbenchUpsertAspect):
        edits.aspect_upserts[intent.aspect.aspect_id] = intent.aspect
    elif isinstance(intent, WorkbenchRemoveAspect):
        if intent.aspect_id not in _aspect_ids(loop_state):
            return {
                "key": f"remove_aspect:{intent.aspect_id}",
                "reason": "要删除的 aspect_id 不存在。",
                "known_aspect_ids": _aspect_ids(loop_state),
            }
        edits.aspect_removals.append(intent.aspect_id)
    elif isinstance(intent, WorkbenchUpdateFraming):
        edits.framing_updates.update(intent.framing.model_dump(exclude_none=True))
    elif isinstance(intent, WorkbenchSetPlan):
        edits.plan_value = [item.model_dump() for item in intent.items]
    elif isinstance(intent, WorkbenchMoveEvidence):
        overlap = set(intent.add) & set(intent.remove)
        if overlap:
            return {
                "key": "evidence",
                "reason": "add 与 remove 不得包含同一 evidence_id；按期望终态二选一后重发。",
                "conflicting_evidence_ids": sorted(overlap),
            }
        edits.evidence_add = list(intent.add)
        edits.evidence_remove = list(intent.remove)
    elif isinstance(intent, WorkbenchAcknowledge):
        edits.acknowledge = True
    return None


def _simulate_active_slots(
    loop_state: SupervisorLoopState, evidence_add: list[str], evidence_remove: list[str]
) -> tuple[set[str], list[str], list[str]]:
    """在活跃集副本上算终态，先释放后找回，不写入磁盘。

    返回 (终态活跃集, release 增量, restore 候选)。超槽跳过由 restore_evidence 的上限
    守门，回执按写入磁盘后的现值标注。
    """
    simulated = set(loop_state.active_evidence_ids)
    release_ids = [
        evidence_id for evidence_id in dict.fromkeys(evidence_remove) if evidence_id in simulated
    ]
    simulated.difference_update(release_ids)
    archive_ids = {item.evidence_id for item in loop_state.evidences}
    restore_ids: list[str] = []
    for evidence_id in dict.fromkeys(evidence_add):
        if evidence_id in archive_ids and evidence_id not in simulated:
            restore_ids.append(evidence_id)
            if len(simulated) < loop_state.active_evidence_limit:
                simulated.add(evidence_id)
    return simulated, release_ids, restore_ids


def _commit_edits(
    loop_state: SupervisorLoopState,
    edits: _SubmissionEdits,
    candidate: ResearchSynthesis | None,
    release_ids: list[str],
    restore_ids: list[str],
) -> dict[str, object]:
    """把全部校验通过的编辑集写入磁盘并装配回执。"""
    applied: list[str] = []
    active_delta: dict[str, list[str]] = {"add": [], "remove": []}
    if release_ids:
        active_delta["remove"] = loop_state.release_evidence(release_ids)
        applied.append("evidence")
    if restore_ids:
        active_delta["add"] = loop_state.restore_evidence(
            [item for item in restore_ids if item not in loop_state.active_evidence_ids]
        )
        if "evidence" not in applied:
            applied.append("evidence")
    if edits.acknowledge and loop_state.research_synthesis is not None:
        loop_state.research_synthesis.based_on_working_set_revision = (
            loop_state.working_set_revision
        )
        applied.append("acknowledge")
    if edits.edits_draft:
        # candidate 为 None 表示变更后全空：稿子清回未开稿状态，清空草稿重新开始是合法动作。
        if candidate is not None:
            candidate.based_on_working_set_revision = loop_state.working_set_revision
        loop_state.research_synthesis = candidate
        if edits.framing_updates:
            applied.append("framing")
        applied.extend(f"upsert_aspect:{aspect_id}" for aspect_id in edits.aspect_upserts)
        applied.extend(f"remove_aspect:{aspect_id}" for aspect_id in edits.aspect_removals)
    if edits.plan_value is not None:
        loop_state.plan = edits.plan_value
        applied.append("plan")

    synthesis = loop_state.research_synthesis
    assigned = set(synthesis.selected_evidence_ids) if synthesis else set()
    receipt: dict[str, object] = {
        "status": "applied",
        "applied": sorted(set(applied)),
        "workbench_revision": loop_state.working_set_revision,
        "synthesis_revision": synthesis.revision if synthesis else 0,
        "active_evidence_count": len(loop_state.active_evidence_ids),
        "unassigned_active_evidence_ids": [
            item.evidence_id
            for item in loop_state.active_evidences()
            if item.evidence_id not in assigned
        ],
        "plan": [dict(item) for item in loop_state.plan],
        # 槽位实际写入磁盘增量，tools.py 据此写 active 通道(add/remove)。
        "slot_additions": active_delta["add"],
        "slot_removals": active_delta["remove"],
    }
    slots_full_skipped = [
        item for item in restore_ids if item not in loop_state.active_evidence_ids
    ]
    if slots_full_skipped:
        receipt["skipped_restore_evidence_ids"] = slots_full_skipped
    return receipt


def apply_workbench_changes(
    loop_state: SupervisorLoopState, changes: list[object]
) -> dict[str, object]:
    """UpdateWorkbench 业务：工作台唯一写入口，操作级整值替换。

    解析并集 → 表态冲突检查 → 在副本上算候选终态 → 过一次跨操作不变式
    （aspect 绑定 ⊆ 终态活跃集）→ 全部通过才写 loop_state。任一项失败整单拒绝并点名
    操作。稿子齐备性不在此校验，由 freeze_synthesis 负责。
    """
    edits, rejections = _project_submission(loop_state, changes)
    if rejections:
        return {"status": "rejected", "rejections": rejections}
    if edits.acknowledge and edits.edits_draft:
        return _rejection(
            "acknowledge", "同单已有 framing/aspect 变更时不需要 acknowledge，时间戳自动盖。"
        )
    if edits.acknowledge and loop_state.research_synthesis is None:
        return _rejection(
            "acknowledge", "尚无综合稿可表态；先用 UpdateWorkbench 记录 aspect 或 framing。"
        )

    final_active, release_ids, restore_ids = _simulate_active_slots(
        loop_state, edits.evidence_add, edits.evidence_remove
    )
    candidate: ResearchSynthesis | None = None
    if edits.edits_draft:
        candidate = _candidate_synthesis(
            loop_state, edits.framing_updates, edits.aspect_upserts, edits.aspect_removals
        )
        if candidate is not None:
            dangling = sorted(
                {evidence_id for aspect in candidate.aspects for evidence_id in aspect.evidence_ids}
                - final_active
            )
            if dangling:
                return _rejection(
                    "aspect",
                    "aspect 只能绑定终态活跃证据；下列 id 未活跃（在 reserve 或不存在），可先 restore 或改绑。",
                    invalid_evidence_ids=dangling,
                )
    return _commit_edits(loop_state, edits, candidate, release_ids, restore_ids)


def _aspect_ids(loop_state: SupervisorLoopState) -> list[str]:
    synthesis = loop_state.research_synthesis
    return [aspect.aspect_id for aspect in synthesis.aspects] if synthesis else []


def _issue_list(exc: ValidationError) -> list[dict[str, str]]:
    return [
        {
            "field": ".".join(str(part) for part in error["loc"]),
            "message": error["msg"],
        }
        for error in exc.errors(include_url=False, include_input=False)
    ]


def freeze_synthesis(
    loop_state: SupervisorLoopState,
    reason: str,
) -> dict[str, object]:
    """ResearchComplete 业务：冻结通过校验的最新综合稿。

    稿子落后于工作台（有未处理的方向结果或槽位变化）即拒，提示用 UpdateWorkbench
    修订或 acknowledge。answer_goal、thesis、decision_rationale 三项非空且至少一块
    aspect 才能提交。接受时写入 completed_synthesis，下一跳退出循环。
    """
    synthesis = loop_state.research_synthesis
    accepted = bool(
        not loop_state.pending_directions
        and synthesis is not None
        and _completion_missing_fields(synthesis) == []
        and loop_state.synthesis_is_fresh(synthesis)
        and (synthesis.selected_evidence_ids or loop_state.provider_exhausted)
    )
    if accepted:
        assert synthesis is not None  # accepted 已包含该条件，供静态类型收窄。
        loop_state.completed_synthesis = synthesis
        loop_state.set_stop_reason(
            StopReason.PROVIDER_EXHAUSTED
            if not synthesis.selected_evidence_ids
            else StopReason.COMPLETED
        )
    receipt: dict[str, object] = {
        "status": "accepted" if accepted else "rejected",
        "reason": reason,
        "workbench_revision": loop_state.working_set_revision,
        **synthesis_snapshot(synthesis),
    }
    if not accepted:
        receipt["rejection_cause"] = _freeze_rejection_cause(loop_state, synthesis)
        if loop_state.pending_directions:
            receipt["directions_running"] = sorted(loop_state.pending_directions)
    return receipt


def _completion_missing_fields(synthesis: ResearchSynthesis) -> list[str]:
    """校验清单：缺哪个点名哪个；三件+至少一块 aspect 齐了才够格冻结。"""
    missing = [
        field_name
        for field_name in ("answer_goal", "thesis", "decision_rationale")
        if not getattr(synthesis, field_name)
    ]
    if not synthesis.aspects:
        missing.append("aspect")
    return missing


def _freeze_rejection_cause(
    loop_state: SupervisorLoopState,
    synthesis: ResearchSynthesis | None,
) -> str:
    """对照 freeze_synthesis 的接受条件逐项给出可行动的拒绝原因。"""
    if loop_state.pending_directions:
        running = "、".join(
            str(entry.get("display_title") or entry.get("task_id") or "")
            for entry in loop_state.pending_directions.values()
        )
        return (
            f"仍有在途方向未返回：{running}。等待其完成并处理结果后修订再冻结"
            "（结果将在回合开头注入，也可用 WaitDirection 主动等）；"
            "也可继续派发互补方向，但不要在途未清零时结束本轮。"
        )
    if synthesis is None:
        return "尚无研究综合稿；用 UpdateWorkbench 记录 aspect 或 framing 后再冻结。"
    missing = _completion_missing_fields(synthesis)
    if missing:
        return (
            f"综合稿未通过完整性校验，缺少：{'、'.join(missing)}。"
            "用 UpdateWorkbench 的 framing 补齐 answer_goal/thesis/decision_rationale"
            "(thesis 是全文要辩护的那一句核心主张)，用 upsert_aspect 立块，再冻结。"
        )
    if not loop_state.synthesis_is_fresh(synthesis):
        return (
            "工作台在成稿后又有变化（方向结果或槽位调整）未反映：请修订 framing/aspect，"
            "或确认无需改动时用 acknowledge 表态后再冻结。"
        )
    return (
        "研究综合稿未引用任何活跃 Evidence。未认证账户级搜索故障时无证据不可冻结："
        "继续派发研究，或若研究已无法进行，直接结束本轮（不调用工具），系统按 partial 交付。"
    )


def compose_final_update(
    state: ResearchState,
    loop_state: SupervisorLoopState,
    config: AgentConfig,
) -> SupervisorStateUpdate:
    """把工作状态转为 State 增量与路由决策，是终局合成的唯一出口。

    explicitly_completed 只在这里判定一次，research_status、generation_mode 与
    WriterDirective 的相关字段都由它派生。
    """
    full_synthesis = loop_state.completed_synthesis
    explicitly_completed = full_synthesis is not None
    # 默认终态只在整轮没有任何信号且未显式收尾时补写，不参与 rank 竞争：已采纳的
    # 终态即便权威度更低，也不该被"预算耗尽"覆盖。账户级熔断已认证时终态记故障
    # 本身，不能误标成预算耗尽。
    if not explicitly_completed and loop_state.stop_reason is None:
        loop_state.stop_reason = (
            StopReason.PROVIDER_EXHAUSTED
            if loop_state.provider_exhausted
            else StopReason.ENDED_WITHOUT_COMPLETION
        )
    latest_synthesis = loop_state.research_synthesis
    partial = (
        partial_synthesis(loop_state, latest_synthesis)
        if loop_state.stop_reason is not None and loop_state.stop_reason.allows_partial_report
        else None
    )
    selected_synthesis = full_synthesis or partial
    can_write = selected_synthesis is not None
    writer_directive = (
        build_writer_directive(
            state,
            loop_state,
            selected_synthesis,
            explicitly_completed=explicitly_completed,
            config=config,
        )
        if selected_synthesis is not None
        else None
    )
    research_status = "completed" if explicitly_completed else "incomplete"
    generation_mode = "full" if explicitly_completed else "partial" if can_write else "not_ready"
    can_continue_to_writer = can_write
    # evidences、source_refs、task_results 的 reducer 都按 id 去重且幂等，因此把
    # SupervisorLoopState 全量副本交给 channel，与只发新增等价。
    return SupervisorStateUpdate(
        evidences=cast(list[Evidence], loop_state.evidences),
        source_refs=cast(list[str], loop_state.source_refs),
        task_results=cast(list[ResearchDirectionResult], loop_state.task_results),
        working_set_revision=loop_state.working_set_revision,
        research_synthesis=loop_state.research_synthesis,
        writer_directive=writer_directive,
        active_evidence_ids=sorted(loop_state.active_evidence_ids),
        run=RunStatus(
            phase="writing" if can_continue_to_writer else "rendering",
            terminal_reason="" if can_continue_to_writer else str(loop_state.stop_reason or ""),
        ),
        supervisor=SupervisorProgress(
            status=research_status,
            current_round=loop_state.current_round,
            coverage_gaps=loop_state.coverage_gaps,
            generation_mode=generation_mode,
        ),
        writer=WriterProgress(
            status="not_started",
            feedback=(
                ""
                if explicitly_completed
                else describe_research_stop(
                    loop_state.stop_reason,
                    loop_state.coverage_gaps,
                    loop_state.failure_details,
                )
            ),
        ),
        supervisor_next=NodeName.WRITER if can_write else NodeName.RENDER_FINAL_REPORT,
    )


def partial_synthesis(
    loop_state: SupervisorLoopState,
    latest: ResearchSynthesis | None,
) -> ResearchSynthesis | None:
    """选择可部分交付的最新综合稿；稿缺校验件时补齐中性值，无稿时生成最小固定版。"""
    active_ids = set(loop_state.active_evidence_ids)
    if latest is not None and latest.selected_evidence_ids:
        if set(latest.selected_evidence_ids).issubset(active_ids):
            # partial 交付没有资格虚构总纲：缺什么补中性值，由 Writer 从材料自行归纳。
            return latest.model_copy(
                update={
                    "answer_goal": latest.answer_goal
                    or loop_state.research_query
                    or "回答用户的研究问题",
                    "thesis": latest.thesis or "",
                    "decision_rationale": latest.decision_rationale
                    or "系统在研究结束时按最新工作台部分交付。",
                }
            )
    evidences = loop_state.active_evidences()
    if not evidences:
        return None
    selected_ids = [item.evidence_id for item in evidences]
    claims = list(dict.fromkeys(item.claim.strip() for item in evidences if item.claim.strip()))
    summary = (
        "；".join(claims[:_FALLBACK_SUMMARY_CLAIM_LIMIT])
        or "已收集可追溯 Evidence，但未形成模型综合结论。"
    )
    gap = "研究结束时模型未显式声明完成；报告只能陈述已验证材料及其适用边界。"
    return ResearchSynthesis(
        revision=(latest.revision + 1 if latest is not None else 1),
        based_on_working_set_revision=loop_state.working_set_revision,
        answer_goal=loop_state.research_query or "回答用户的研究问题",
        aspects=[
            ResearchAspect(
                aspect_id="fallback-evidence",
                topic="已验证材料",
                role="保守回应用户问题",
                status="partial",
                conclusion=summary[:STRUCTURED_SUMMARY_HARD_LIMIT_CHARS],
                gaps=[AspectIssue(text=gap)],
                evidence_ids=selected_ids[:EVIDENCE_REFERENCES_HARD_LIMIT],
            )
        ],
        selected_evidence_ids=selected_ids,
        decision_rationale="系统在研究结束时基于当前活跃 Evidence 生成最小可交付综合稿。",
    )


def build_writer_directive(
    state: ResearchState,
    loop_state: SupervisorLoopState,
    synthesis: ResearchSynthesis,
    *,
    explicitly_completed: bool,
    config: AgentConfig,
) -> WriterDirective:
    """从冻结综合版本派生 Writer 唯一可见的写作指令。

    ReportBrief 在指令内部构造、随指令一起交接，State 顶层不保留平行的第二副本。
    explicitly_completed 是"研究是否走到模型显式收尾"的事实，由调用方判定一次后传入。
    """
    report_brief = report_brief_from_synthesis(synthesis, max_caveats=config.report_max_caveats)
    review = section(state, "review", ReviewProgress)
    # rejected_draft 是校验失败的备用稿：仅在无通过稿时作修订底稿。
    previous_draft = str(state.get("report_draft") or state.get("rejected_draft") or "")
    revision_instructions = (
        [
            *([review.feedback] if review.feedback else []),
            *(f"修复缺口：{gap}" for gap in review.gaps),
        ]
        if review.status == "rejected"
        else []
    )
    evidence_ids = list(
        dict.fromkeys(
            evidence_id for chapter in report_brief.chapters for evidence_id in chapter.evidence_ids
        )
    )
    if not evidence_ids:
        # 章节卡未绑定任何 evidence_ids 时，保留冻结综合稿的选择集。
        evidence_ids = list(synthesis.selected_evidence_ids)
    return WriterDirective(
        query=str(state.get("clarified_query") or state.get("query") or ""),
        report_brief=report_brief,
        research_status="completed" if explicitly_completed else "incomplete",
        generation_mode="full" if explicitly_completed else "partial",
        evidence_ids=evidence_ids,
        known_gaps=list(report_brief.caveats),
        revision_instructions=revision_instructions,
        previous_draft=previous_draft,
    )


def report_brief_from_synthesis(synthesis: ResearchSynthesis, *, max_caveats: int) -> ReportBrief:
    """从冻结综合版本派生报告任务书，避免 ResearchComplete 再提交第二事实源。

    章节卡是 aspect 的写作投影：剥离 aspect_id/status，空结论以首条 gap 替代；
    caveats 只保留各板块 disclose=true 的条目，披露裁量留在登记表。
    """
    chapters = [
        ReportChapter(
            topic=aspect.topic,
            role=aspect.role,
            required=aspect.required,
            conclusion=aspect.conclusion or next((issue.text for issue in aspect.gaps), ""),
            key_points=list(aspect.key_points),
            evidence_ids=list(aspect.evidence_ids),
        )
        for aspect in synthesis.aspects
    ]
    caveats: list[str] = []
    for aspect in synthesis.aspects:
        for label, issues in (("缺口", aspect.gaps), ("结论冲突", aspect.conflicts)):
            caveats.extend(
                f"{label}|板块「{aspect.topic}」:{issue.text}" for issue in issues if issue.disclose
            )
    return ReportBrief(
        answer_goal=synthesis.answer_goal,
        thesis=synthesis.thesis,
        chapters=chapters,
        caveats=caveats[:max_caveats],
    )


def describe_research_stop(
    stop_reason: StopReason | None,
    coverage_gaps: list[str],
    failure_details: list[str],
) -> str:
    """把研究无法继续的原因保留给最终不完整报告与事件诊断。

    文案单一来源是 StopReason.description，这里只补本次运行的具体细节。
    AGENT_FAILED 取 failure_details 的详情，其余原因取 coverage_gaps，两条通道不互串内容。
    """
    if stop_reason is StopReason.AGENT_FAILED:
        detail = next(
            (item for item in reversed(failure_details) if item.strip()),
            "未记录到异常详情，详见运行事件流。",
        )
        return f"{stop_reason.description} {detail}"
    # 诊断细节取 coverage_gaps 与 failure_details 并集的最后一条非空项；
    # failure_details 只进诊断文案，不进报告"未闭合缺口"清单，两通道各守边界。
    detail = next(
        (item for item in reversed([*coverage_gaps, *failure_details]) if item.strip()),
        "未形成可验证的完整覆盖。",
    )
    prefix = (
        stop_reason.description
        if stop_reason
        else "Supervisor 未确认现有材料足以形成完整研究报告。"
    )
    return f"{prefix} {detail}"
