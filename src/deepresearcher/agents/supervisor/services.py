"""Supervisor 业务实现:不感知 langgraph 的纯函数(async 与否按是否需要等待)。

delegate_research 执行一次 ResearchDelegate 工具请求:预算 hard check、任务
编号、派发 ResearchAgent、吸收结果,返回给模型的完整方向报告 dict;
工具回执与消息配对留在协议层(tools.py),本模块不认识 ToolMessage。
簿记并发由 bookkeeping_lock 保护,实际子 Agent 并发受 deps.worker_limit 限制。
"""

import asyncio
from typing import cast

from pydantic import ValidationError

from deepresearcher.agents.supervisor.state import (
    SupervisorDeps,
    SupervisorLoopState,
    TaskExecution,
    evidence_card,
    synthesis_snapshot,
    working_set_snapshot,
)
from deepresearcher.config import AgentConfig
from deepresearcher.evidence.models import Evidence
from deepresearcher.observability.events.names import EventName
from deepresearcher.observability.execution import AgentExecutionScope
from deepresearcher.routing import NodeName
from deepresearcher.schemas import (
    CoveredTopic,
    ReportBrief,
    ResearchAgentResult,
    ResearchAspect,
    ResearchDirectionResult,
    ResearchSynthesis,
    ReviewProgress,
    RunStatus,
    StopReason,
    SupervisorProgress,
    SupervisorStateUpdate,
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


async def delegate_research(
    deps: SupervisorDeps,
    loop_state: SupervisorLoopState,
    scope: AgentExecutionScope,
    bookkeeping_lock: asyncio.Lock,
    topic: str,
    display_title: str = "",
) -> dict[str, object]:
    """执行一次 ResearchDelegate 工具请求:预算 hard check、任务编号、
    派发 ResearchAgent、吸收结果、返回完整方向报告。

    display_title 是方向卡的展示短题(与 topic 完整契约分离);缺省时前端回退截断 question。
    程序不做主题去重:防重复靠提示词纪律,浪费靠 max_subtasks/轮次预算封顶。
    """
    round_no = loop_state.current_round

    def reported(result: dict[str, object]) -> dict[str, object]:
        # 规划器的工具调用若被静默消化（blocked），事件流里只会
        # 看到连续两个 model_turn——delegate_started/completed 让“空轮次”可解释。
        deps.emit(
            EventName.DELEGATE_COMPLETED,
            {
                "status": str(result.get("status", "")),
                "reason": str(result.get("reason", "")),
                "topic_chars": len(topic),
                "evidence_count": result.get("evidence_count"),
                "source_count": result.get("source_count"),
            },
        )
        return result

    deps.emit(EventName.DELEGATE_STARTED, {"topic_chars": len(topic)})
    if round_no > deps.config.max_research_rounds:
        loop_state.set_stop_reason(StopReason.GLOBAL_ROUND_BUDGET_EXHAUSTED)
        return reported(
            {
                "status": "blocked",
                "reason": "round_budget_exhausted",
                "instruction": "研究轮次预算已耗尽；请修订最新研究综合稿。若达到完整标准则调用 ResearchComplete，否则直接结束，系统将按 partial 交付。",
            }
        )
    async with bookkeeping_lock:
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
    execution = await _execute_research_task(deps, task)
    async with bookkeeping_lock:
        loop_state.absorb(execution)
    direction_report: dict[str, object] = {
        "status": execution.task_result.execution_status,
        "research_direction": execution.task_result.research_direction,
        "coverage_status": execution.task_result.coverage_status,
        "evidence_count": execution.task_result.evidence_count,
        "source_count": execution.task_result.source_count,
        "remaining_gaps": execution.task_result.remaining_gaps,
        "conclusion": execution.task_result.conclusion,
        "failures": execution.task_result.failures,
        "working_set_revision": loop_state.working_set_revision,
        "evidence": [
            evidence_card(item)
            for item in execution.evidences
            if item.evidence_id in loop_state.active_evidence_ids
        ],
    }
    if execution.task_result.provider_exhausted:
        # 数据提示（无分支控制流）：让 Supervisor 模型读到系统性不可用后自然停止派发、收尾。
        direction_report["provider_exhausted"] = True
        direction_report["instruction"] = (
            "搜索服务账户级不可用（额度耗尽/密钥无效），系统性问题：再派新方向也会同样失败。"
            "停止派发 ResearchDelegate；把已有 Evidence 修订进综合稿，随后调用 ResearchComplete；"
            "若不足以成文则直接结束本轮，系统按 partial 交付。"
        )
    return reported(direction_report)


async def _execute_research_task(deps: SupervisorDeps, task: SubTask) -> TaskExecution:
    """执行单个方向研究；worker 异常降级为 failed 结果，不中断整轮。"""
    round_no = int(task.get("round", 1))
    task_context = {
        "task_id": task["id"],
        "worker_id": task.get("worker_id", task["id"]),
        "worker_index": int(task.get("worker_index", task.get("sequence", 0))),
        "parent_task_id": task.get("parent_task_id", ""),
        "operation_id": task.get("operation_id", task["id"]),
        "concurrency_limit": deps.config.max_parallel_workers,
    }
    async with deps.worker_limit:
        deps.emit(
            EventName.RESEARCH_TASK_STARTED,
            {
                **task_context,
                "task_index": int(task.get("sequence", 0)),
                "component": "research_agent",
                "title": str(task.get("display_title") or ""),
                "question": task["question"][: deps.config.supervisor_preview_chars],
            },
        )
        try:
            result = await deps.research_agent.run(task)
            # 研究员是子 Agent 边界：在写入审计事件或 State 前先验证结果契约；
            # 同 run 内已是模型时零开销，防御性恢复覆盖跨进程 checkpoint。
            agent_result = (
                result
                if isinstance(result, ResearchAgentResult)
                else ResearchAgentResult.model_validate(result)
            )
            task_result = agent_result.task_result
            evidences = list(agent_result.evidences)
            selected_ids = list(agent_result.selected_evidence_ids)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            execution = TaskExecution.failed_for(
                task,
                round_no,
                str(exc)[: deps.config.supervisor_preview_chars],
            )
            deps.emit(
                EventName.RESEARCH_TASK_FAILED,
                {**task_context, **execution.task_result.model_dump()},
                component="research_agent",
            )
            return execution
        deps.emit(
            EventName.RESEARCH_TASK_COMPLETED,
            {**task_context, **task_result.model_dump()},
            component="research_agent",
        )
        return TaskExecution(
            task_result=task_result,
            evidences=evidences,
            selected_evidence_ids=selected_ids,
            source_refs=list(agent_result.source_refs),
        )


def freeze_synthesis(
    loop_state: SupervisorLoopState,
    synthesis_revision: int,
    reason: str,
) -> dict[str, object]:
    """ResearchComplete 业务:冻结最新且未过期的研究综合稿,返回给模型的完整回执。

    拒绝回执自带 requested vs current 两个 revision 与综合稿快照,模型同回合
    修正后重提即可;不写 failure_details——自愈的协议失误不是执行故障,
    反复不成时终局归属模型调用天花板。接受时 completed_synthesis 已落盘,
    SubmittedExitMiddleware 在下一跳静默出环。
    """
    synthesis = loop_state.research_synthesis
    accepted = bool(
        synthesis is not None
        and synthesis.revision == synthesis_revision
        and loop_state.synthesis_is_fresh(synthesis)
        and synthesis.selected_evidence_ids
    )
    if accepted:
        assert synthesis is not None  # accepted 已包含该条件,供静态类型收窄。
        loop_state.sufficient = (
            all(
                not aspect.required or aspect.status == "covered"
                for aspect in synthesis.aspects
            )
            and not synthesis.open_gaps
            and not synthesis.conflicts
        )
        loop_state.completed_synthesis = synthesis
        loop_state.set_stop_reason(
            StopReason.SUFFICIENT if loop_state.sufficient else StopReason.SUBMITTED_WITH_GAPS
        )
    return {
        "status": "accepted" if accepted else "rejected",
        "reason": reason,
        "requested_revision": synthesis_revision,
        "current_working_set_revision": loop_state.working_set_revision,
        **synthesis_snapshot(synthesis),
    }


def revise_synthesis(
    loop_state: SupervisorLoopState,
    *,
    expected_revision: int,
    expected_working_set_revision: int,
    answer_goal: str,
    overall_summary: str,
    aspects: list[ResearchAspect],
    open_gaps: list[str],
    conflicts: list[str],
    next_actions: list[str],
    decision_rationale: str,
) -> dict[str, object]:
    """ReviseResearchSynthesis 业务:版本对账、引用越权校验、构造并落盘新综合稿。

    返回回执 dict(stale/rejected/accepted 三态);接受前完成 loop_state 变更。
    """
    current_revision = (
        loop_state.research_synthesis.revision if loop_state.research_synthesis else 0
    )
    if (
        expected_revision != current_revision
        or expected_working_set_revision != loop_state.working_set_revision
    ):
        return {
            "status": "stale",
            # 回执给的是系统当前值;与入参 expected_* 方向相反,键名必须区分两域。
            "current_synthesis_revision": current_revision,
            "current_working_set_revision": loop_state.working_set_revision,
        }
    active_ids = set(loop_state.active_evidence_ids)
    referenced_ids = {evidence_id for aspect in aspects for evidence_id in aspect.evidence_ids}
    unknown_ids = sorted(referenced_ids - active_ids)
    if unknown_ids:
        return {
            "status": "rejected",
            "reason": "研究综合稿只能引用当前活跃 Evidence。",
            "invalid_evidence_ids": unknown_ids,
            **working_set_snapshot(loop_state),
        }
    try:
        synthesis = ResearchSynthesis(
            revision=current_revision + 1,
            based_on_working_set_revision=loop_state.working_set_revision,
            answer_goal=answer_goal,
            overall_summary=overall_summary,
            aspects=aspects,
            open_gaps=open_gaps,
            conflicts=conflicts,
            next_actions=next_actions,
            decision_rationale=decision_rationale,
        )
    except ValidationError as exc:
        issues = [
            {
                "field": ".".join(str(part) for part in error["loc"]),
                "message": error["msg"],
            }
            for error in exc.errors(include_url=False, include_input=False)
        ]
        return {
            "status": "rejected",
            "reason": "研究综合稿不满足提交契约，请按 issues 修正后重试。",
            "issues": issues,
            **working_set_snapshot(loop_state),
        }
    loop_state.research_synthesis = synthesis
    assigned_ids = set(synthesis.selected_evidence_ids)
    return {
        "status": "accepted",
        **synthesis_snapshot(synthesis),
        "unassigned_active_evidence_ids": [
            item.evidence_id
            for item in loop_state.active_evidences()
            if item.evidence_id not in assigned_ids
        ],
    }


def compose_final_update(
    state: ResearchState,
    loop_state: SupervisorLoopState,
    config: AgentConfig,
) -> SupervisorStateUpdate:
    """把工作状态转为 State 增量与路由决策(终局合成的唯一出口)。

    sufficient 只在本次计算读一次,派生 research_status/generation_mode 与
    WriterDirective 的两个映射,不再各算各的。
    """
    # 默认终态只是兜底,不是优先级判断:整轮没产生任何信号时才补。
    # 保持 `is None` + 直接赋值:地板不参与 rank 竞争,任何已采纳的终态
    # (哪怕权威度更低)都不该被"预算耗尽"的猜测覆盖。
    if not loop_state.sufficient and loop_state.stop_reason is None:
        loop_state.stop_reason = StopReason.ROUND_BUDGET_EXHAUSTED
    full_synthesis = loop_state.completed_synthesis
    latest_synthesis = loop_state.research_synthesis
    partial = (
        partial_synthesis(loop_state, latest_synthesis)
        if loop_state.stop_reason is not None and loop_state.stop_reason.allows_partial_report
        else None
    )
    selected_synthesis = full_synthesis or partial
    can_write = selected_synthesis is not None
    sufficient = loop_state.sufficient
    writer_directive = (
        build_writer_directive(
            state, loop_state, selected_synthesis, sufficient=sufficient, config=config
        )
        if selected_synthesis is not None
        else None
    )
    research_status = "completed" if sufficient else "incomplete"
    generation_mode = "full" if sufficient else "partial" if can_write else "not_ready"
    can_continue_to_writer = can_write
    # evidences / source_refs / task_results 的 reducer 幂等(merge_evidences /
    # merge_task_results / merge_unique),直接把 SupervisorLoopState 全量副本交给 channel;
    # reducer 按 id 折回原样,等价于只发新增。
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
            is_sufficient=sufficient,
        ),
        writer=WriterProgress(
            status="not_started",
            feedback=(
                ""
                if sufficient
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
    """选择可部分交付的最新综合稿；无综合稿时生成最小固定版。"""
    active_ids = set(loop_state.active_evidence_ids)
    if latest is not None and latest.selected_evidence_ids:
        if set(latest.selected_evidence_ids).issubset(active_ids):
            return latest
    evidences = loop_state.active_evidences()
    if not evidences:
        return None
    selected_ids = [item.evidence_id for item in evidences]
    claims = list(dict.fromkeys(item.claim.strip() for item in evidences if item.claim.strip()))
    summary = (
        "；".join(claims[:_FALLBACK_SUMMARY_CLAIM_LIMIT])
        or "已收集可追溯 Evidence，但未形成模型综合结论。"
    )
    gap = "研究未达到完整标准；报告只能陈述已验证材料及其适用边界。"
    return ResearchSynthesis(
        revision=(latest.revision + 1 if latest is not None else 1),
        based_on_working_set_revision=loop_state.working_set_revision,
        answer_goal=loop_state.research_query or "回答用户的研究问题",
        overall_summary=summary[:STRUCTURED_SUMMARY_HARD_LIMIT_CHARS],
        aspects=[
            ResearchAspect(
                aspect_id="fallback-evidence",
                topic="已验证材料",
                role="保守回应用户问题",
                status="partial",
                summary=summary[:STRUCTURED_SUMMARY_HARD_LIMIT_CHARS],
                evidence_ids=selected_ids[:EVIDENCE_REFERENCES_HARD_LIMIT],
                remaining_gap=gap,
            )
        ],
        selected_evidence_ids=selected_ids,
        open_gaps=[gap],
        conflicts=[],
        next_actions=[],
        decision_rationale="系统在研究结束时基于当前活跃 Evidence 生成最小可交付综合稿。",
    )


def build_writer_directive(
    state: ResearchState,
    loop_state: SupervisorLoopState,
    synthesis: ResearchSynthesis,
    *,
    sufficient: bool,
    config: AgentConfig,
) -> WriterDirective:
    """从冻结综合版本派生 Writer 唯一可见的写作指令。

    ReportBrief 在指令内部构造、随指令一起交接；State 顶层不再有平行的第二副本。
    sufficient 由调用方(compose_final_update)一次判定后传入。
    """
    report_brief = report_brief_from_synthesis(synthesis, max_caveats=config.report_max_caveats)
    review = section(state, "review", ReviewProgress)
    # rejected_draft 是校验失败的兜底稿:仅在无通过稿时作修订底稿。
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
            evidence_id
            for topic in report_brief.covered_topics
            for evidence_id in topic.evidence_ids
        )
    )
    if not evidence_ids:
        # 旧 checkpoint 的 CoveredTopic 没有 evidence_ids，保留冻结综合稿的选择集。
        evidence_ids = list(synthesis.selected_evidence_ids)
    return WriterDirective(
        query=str(state.get("clarified_query") or state.get("query") or ""),
        report_brief=report_brief,
        research_status="completed" if sufficient else "incomplete",
        generation_mode="full" if sufficient else "partial",
        evidence_ids=evidence_ids,
        known_gaps=list(dict.fromkeys([*synthesis.open_gaps, *synthesis.conflicts]))[
            : config.report_max_caveats
        ],
        revision_instructions=revision_instructions,
        previous_draft=previous_draft,
    )


def report_brief_from_synthesis(
    synthesis: ResearchSynthesis, *, max_caveats: int
) -> ReportBrief:
    """从冻结综合版本派生报告任务书，避免 Complete 再提交第二事实源。"""
    topics = [
        CoveredTopic(
            topic=aspect.topic,
            role=aspect.role,
            reason=aspect.summary or aspect.remaining_gap,
            required=aspect.required,
            evidence_ids=list(aspect.evidence_ids),
        )
        for aspect in synthesis.aspects
    ]
    return ReportBrief(
        answer_goal=synthesis.answer_goal,
        covered_topics=topics,
        required_points=[aspect.topic for aspect in synthesis.aspects if aspect.required],
        caveats=list(dict.fromkeys([*synthesis.open_gaps, *synthesis.conflicts]))[:max_caveats],
    )


def describe_research_stop(
    stop_reason: StopReason | None,
    coverage_gaps: list[str],
    failure_details: list[str],
) -> str:
    """把研究无法继续的原因保留给最终不完整报告与事件诊断。

    文案单一来源在 ``StopReason.description``；这里只补充逐次运行的
    具体细节，不再各自维护字符串清单。执行失败（AGENT_FAILED）取
    failure_details，其余取 coverage_gaps——两条通道不互串内容。
    """
    if stop_reason is StopReason.AGENT_FAILED:
        detail = next(
            (item for item in reversed(failure_details) if item.strip()),
            "未记录到异常详情，详见运行事件流。",
        )
        return f"{stop_reason.description} {detail}"
    # 诊断优先取研究内容缺口;无缺口可报时退回执行/协议尾注——failure_details
    # 只进诊断文案,不进报告"未闭合缺口"清单,两通道规矩不变。
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
