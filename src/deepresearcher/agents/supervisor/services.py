"""Supervisor 业务实现:不感知 langgraph 的纯函数,async 与否按是否需要等待。

工具回执的字符串协议与消息配对留在协议层 tools.py,本模块只回 dict,不构造
ToolMessage。子 Agent 并发受 deps.worker_limit 限制;对 loop_state 的改写全部收在
不含 await 的同步段内,原子性由事件循环的单线程调度保证。
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
    topic: str,
    display_title: str = "",
    *,
    tool_call_id: str = "",
    origin_state: dict[str, object] | None = None,
) -> dict[str, object]:
    """执行一次 ResearchDelegate 工具请求:预算硬校验、任务编号、
    派发 ResearchAgent、吸收结果、返回完整方向报告。

    display_title 是方向卡的展示短题,topic 仍承载完整契约;缺省时前端回退截断 question。
    程序不做主题去重:防重复交给提示词纪律,重复调用的浪费由 max_subtasks 与轮次预算封顶。
    tool_call_id 与 origin_state 提供时编号按调用位置派生(跨重放稳定),否则退回活视图分配。
    编号分配与 absorb 为同步临界段,原子性依赖单线程调度,段内不得引入 await。
    """
    round_no = loop_state.current_round

    def reported(result: dict[str, object]) -> dict[str, object]:
        # 被预算拒绝的工具调用不产生工具事件,事件流里只剩连续两个 model_turn;
        # 补发 delegate_started/completed,让这类无产出轮次可解释。
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
    execution = await _execute_research_task(deps, task)
    # absorb 是同步临界段:入档、槽位分配、推进版本一次完成;段内禁引入 await,
    # 否则同批其他 delegate 会看到半更新的工作集并在回执里引用它。
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
        # 只给数据、不加控制流分支:Supervisor 模型读到系统性不可用后自然停止派发、收尾。
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
            # 子 Agent 边界:结果写入审计事件或 State 前先过契约校验;
            # 同 run 内已是实例时不花校验开销,跨进程 checkpoint 恢复的 dict 由校验重建。
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

    拒绝回执附 rejection_cause 与 requested、current 两个 revision:版本过期这类
    协议漂移由模型同回合修正后重提即可,不写 failure_details——能自纠的协议失误不是
    执行故障,反复不成立时终局落在模型调用上限。无证据的综合稿仅在 absorb 认证过账户级
    熔断、即 loop_state.provider_exhausted 为真时接受,终态记
    StopReason.PROVIDER_EXHAUSTED。接受时 completed_synthesis 已写入通道,
    SubmittedExitMiddleware 在下一跳直接退出循环。
    """
    synthesis = loop_state.research_synthesis
    accepted = bool(
        synthesis is not None
        and synthesis.revision == synthesis_revision
        and loop_state.synthesis_is_fresh(synthesis)
        and (synthesis.selected_evidence_ids or loop_state.provider_exhausted)
    )
    if accepted:
        assert synthesis is not None  # accepted 已包含该条件,供静态类型收窄。
        loop_state.sufficient = (
            all(not aspect.required or aspect.status == "covered" for aspect in synthesis.aspects)
            and not synthesis.open_gaps
            and not synthesis.conflicts
        )
        loop_state.completed_synthesis = synthesis
        if not synthesis.selected_evidence_ids:
            loop_state.set_stop_reason(StopReason.PROVIDER_EXHAUSTED)
        else:
            loop_state.set_stop_reason(
                StopReason.SUFFICIENT if loop_state.sufficient else StopReason.SUBMITTED_WITH_GAPS
            )
    receipt: dict[str, object] = {
        "status": "accepted" if accepted else "rejected",
        "reason": reason,
        "requested_revision": synthesis_revision,
        "current_working_set_revision": loop_state.working_set_revision,
        **synthesis_snapshot(synthesis),
    }
    if not accepted:
        receipt["rejection_cause"] = _freeze_rejection_cause(
            loop_state, synthesis, synthesis_revision
        )
    return receipt


def _freeze_rejection_cause(
    loop_state: SupervisorLoopState,
    synthesis: ResearchSynthesis | None,
    synthesis_revision: int,
) -> str:
    """对照 freeze_synthesis 的接受条件逐项给出拒绝原因,让模型拿到可行动的下一步。

    缺原因时模型只能从 revision 数字猜拒绝理由,倾向再派新方向重试,
    反而推翻熔断回执中的停止派发指令。
    """
    if synthesis is None:
        return "尚无研究综合稿；先调用 ReviseResearchSynthesis 产出综合稿，再冻结。"
    if synthesis.revision != synthesis_revision:
        return (
            f"synthesis_revision 不一致：当前综合稿为 {synthesis.revision}，"
            "按 current_synthesis_revision 对齐后重提。"
        )
    if not loop_state.synthesis_is_fresh(synthesis):
        return "综合稿基于已过期的工作集版本；先按当前 working set 修订，再冻结。"
    return (
        "研究综合稿未引用任何活跃 Evidence。未认证账户级搜索故障时无证据不可冻结："
        "继续派发研究，或若研究已无法进行，直接结束本轮（不调用工具），系统按 partial 交付。"
    )


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
            # 回执给的是系统当前值,与模型提交的 expected_* 分属两域,键名必须区分开。
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
    """把工作状态转为 State 增量与路由决策,是终局合成的唯一出口。

    sufficient 只在这里判定一次,research_status、generation_mode 与
    WriterDirective 的相关字段都由它派生。
    """
    # 默认终态只在整轮没有任何信号时补写,不参与 rank 竞争:已采纳的终态即便权威度更低,
    # 也不该被"预算耗尽"覆盖。账户级熔断已认证时终态记故障本身,不能误标成预算耗尽。
    if not loop_state.sufficient and loop_state.stop_reason is None:
        loop_state.stop_reason = (
            StopReason.PROVIDER_EXHAUSTED
            if loop_state.provider_exhausted
            else StopReason.ROUND_BUDGET_EXHAUSTED
        )
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
    # evidences、source_refs、task_results 的 reducer 即 merge_evidences、merge_unique、
    # merge_task_results,都按 id 折重、幂等,所以直接把 SupervisorLoopState 全量副本交给
    # channel,效果等价于只发新增。
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

    ReportBrief 在指令内部构造、随指令一起交接,State 顶层不保留平行的第二副本。
    sufficient 由调用方 compose_final_update 判定一次后传入。
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


def report_brief_from_synthesis(synthesis: ResearchSynthesis, *, max_caveats: int) -> ReportBrief:
    """从冻结综合版本派生报告任务书,避免 ResearchComplete 再提交第二事实源。"""
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

    文案单一来源是 ``StopReason.description``,这里只补本次运行的具体细节。
    AGENT_FAILED 取 failure_details 的详情,其余原因取 coverage_gaps,两条通道不互串内容。
    """
    if stop_reason is StopReason.AGENT_FAILED:
        detail = next(
            (item for item in reversed(failure_details) if item.strip()),
            "未记录到异常详情，详见运行事件流。",
        )
        return f"{stop_reason.description} {detail}"
    # 诊断细节取 coverage_gaps 与 failure_details 并集的最后一条非空项;
    # failure_details 只进诊断文案,不进报告"未闭合缺口"清单,两通道各守边界。
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
