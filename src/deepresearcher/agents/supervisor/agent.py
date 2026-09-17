"""动态研究 Supervisor：以工具调用循环驱动方向级研究。

模型每轮通过 ResearchDelegate 派发方向级 ResearchAgent(子 Agent 抽象为工具),
或通过 ResearchComplete 宣布现有 Evidence 足以成文;轮次预算、任务/URL 去重、
并发上限与异常降级由本地程序强制,不依赖模型自觉。
"""

import asyncio
from typing import Any, cast
from urllib.parse import parse_qsl, urldefrag, urlencode, urlsplit, urlunsplit

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from deepresearcher.agents.middleware import (
    AGENT_RECURSION_LIMIT,
    LIMIT_MESSAGE_MARKER,
    MiddlewareProfile,
    build_agent_middleware,
)
from deepresearcher.agents.researcher import ResearchAgent
from deepresearcher.agents.supervisor.state import (
    RunUrlReservations,
    SupervisorRuntimeContext,
    TaskExecution,
    WorkingState,
    evidence_card,
    synthesis_snapshot,
)
from deepresearcher.agents.supervisor.tools import (
    build_supervisor_tools,
)
from deepresearcher.config import AgentConfig, language_directive
from deepresearcher.context.execution import AgentExecutionScope
from deepresearcher.context.runtime import get_runtime_environment
from deepresearcher.evidence.models import Evidence
from deepresearcher.llm import LLMConfigurationError, LLMInvoker
from deepresearcher.observability.events import JsonlSink, emit_agent_event
from deepresearcher.observability.logger import get_logger
from deepresearcher.prompts import json_data_section, load_prompt
from deepresearcher.routing import NodeName
from deepresearcher.schemas import (
    CoveredTopic,
    ReportBrief,
    ResearchAgentResult,
    ResearchAspect,
    ResearchDirectionResult,
    ResearchProgress,
    ResearchSynthesis,
    ReviewProgress,
    RunLifecycle,
    StopReason,
    SupervisorStateUpdate,
    WriterDirective,
    WriterProgress,
)
from deepresearcher.schemas.limits import (
    EVIDENCE_REFERENCES_HARD_LIMIT,
    STRUCTURED_SUMMARY_HARD_LIMIT_CHARS,
)
from deepresearcher.state import ResearchState, SubTask, section

_SUPERVISOR_SYSTEM_PROMPT = load_prompt("supervisor")
_FALLBACK_SUMMARY_CLAIM_LIMIT = 6


def _model_call_limit_hit(messages: list[BaseMessage]) -> bool:
    """判断本次 Agent 运行是否被 ModelCallLimitMiddleware 掐断而非模型正常收尾。"""
    return any(
        isinstance(message, AIMessage) and LIMIT_MESSAGE_MARKER in str(message.content)
        for message in messages[-3:]
    )


class ResearchSupervisor:
    """维护研究工具循环、覆盖判断、任务派发与有界并发。"""

    def __init__(
        self,
        llm: LLMInvoker,
        config: AgentConfig,
        *,
        research_agent: ResearchAgent,
        event_sink: JsonlSink | None = None,
        context_window_tokens: int = 32_768,
    ):
        if llm is None:
            raise LLMConfigurationError("ResearchSupervisor 需要已装配的 LLMInvoker。")
        if research_agent is None:
            raise ValueError("ResearchSupervisor 需要 ResearchAgent。")
        self.llm = llm
        self.config = config
        self.research_agent = research_agent
        self.event_sink = event_sink
        self.logger = get_logger("deepresearcher.agents.supervisor")
        self._worker_limit = asyncio.Semaphore(config.max_parallel_workers)
        self._agent_loop = create_agent(
            model=cast(Any, llm),
            tools=build_supervisor_tools(),
            system_prompt=_SUPERVISOR_SYSTEM_PROMPT
            + "\n"
            + language_directive(config.output_language),
            context_schema=SupervisorRuntimeContext,
            middleware=cast(
                Any,
                build_agent_middleware(
                    MiddlewareProfile(
                        agent_name="Supervisor",
                        model=getattr(self.llm, "chat_model", None),
                        # 一次节点访问 = 一轮；ModelCallLimit 只是防失控天花板：
                        # 一轮最多 max_subtasks_per_round 次委托 + 读工作集/决策/收尾的余量。
                        # 轮次配额由 remaining_rounds 提示 + delegate() 的本地 hard check 执行。
                        max_turns=config.max_subtasks_per_round + 10,
                        context_window_tokens=context_window_tokens,
                        retry_tools=[(["ResearchDelegate"], "ResearchDelegate")],
                        serial_tools={
                            "ReadWorkingSet",
                            "ReleaseEvidence",
                            "RestoreEvidence",
                            "ReviseResearchSynthesis",
                            "ResearchComplete",
                        },
                        tool_call_limits=[("ResearchDelegate", config.max_subtasks_per_round)],
                        emit=self._emit_audit_event,
                    )
                ),
            ),
            name="supervisor",
        )

    @staticmethod
    def _build_supervisor_context(
        state: ResearchState,
    ) -> list[BaseMessage]:
        """从图 State 恢复 Supervisor 私有上下文；首次运行时写入初始委托。"""
        history = list(state.get("supervisor_messages", []))
        if history:
            return history
        return [
            HumanMessage(
                content=(
                    json_data_section("运行时环境", get_runtime_environment().payload())
                    + "\n\n---\n\n"
                    + json_data_section(
                        "研究委托",
                        {
                            "research_question": state.get(
                                "clarified_query", state.get("query", "")
                            ),
                            "research_brief": state.get("research_brief", ""),
                        },
                    )
                )
            )
        ]

    async def run(self, state: ResearchState) -> dict[str, object]:
        """注入审阅回流（如有），然后执行有界的研究工具调用循环。

        改写还是补研究不由独立决策判定，而由工具循环里的模型直接表达：
        ResearchComplete 冻结最新综合版本进入改写，ResearchDelegate 继续补研究。
        """
        history = self._build_supervisor_context(state)
        # 首次运行时 _build_supervisor_context 会补入初始 System/Human 消息；
        # 快照必须取 State 入口长度，确保这些消息也能持久化到上下文历史。
        history_start = len(state.get("supervisor_messages", []))

        review = section(state, "review", ReviewProgress)
        if review.status == "rejected":
            if review.attempts > self.config.max_post_review_recovery_cycles:
                update = SupervisorStateUpdate(
                    run=RunLifecycle(
                        phase="rendering", terminal_reason="review_recovery_exhausted"
                    ),
                    research=section(state, "research", ResearchProgress),
                    writer=section(state, "writer", WriterProgress),
                )
                return {**update.state_update(), "supervisor_messages": history[history_start:]}
            self._append_review_rejection(state, history)

        update = await self._run_agent_loop(state, history)
        return {**update.state_update(), "supervisor_messages": history[history_start:]}

    async def _run_agent_loop(
        self,
        state: ResearchState,
        history: list[BaseMessage],
    ) -> SupervisorStateUpdate:
        """运行 Supervisor 标准 Agent；工具通过运行时上下文修改 WorkingState。"""
        url_reservations = RunUrlReservations(
            state.get("attempted_source_urls", []),
            normalize_url=self._normalize_source_url,
        )
        working = WorkingState(
            state,
            dedup_key=self._task_deduplication_key,
            active_evidence_limit=self.config.supervisor_max_active_evidences,
        )
        research = section(state, "research", ResearchProgress)
        round_no = research.current_round + 1
        working.current_round = round_no
        self._append_research_observation(
            history,
            {
                "remaining_rounds": max(
                    0, self.config.max_research_rounds - research.current_round
                ),
                "working_set_revision": working.working_set_revision,
                "working_set": self._working_set_snapshot(working),
                "research_synthesis": self._research_synthesis_observation(
                    working.research_synthesis
                ),
            },
        )

        async def delegate(topic: str) -> dict[str, object]:
            def _reported(result: dict[str, object]) -> dict[str, object]:
                # 规划器的工具调用若被静默消化（blocked/skipped），事件流里只会
                # 看到连续两个 model_turn——delegate_started/completed 让“空轮次”可解释。
                self._emit_audit_event(
                    "delegate_completed",
                    {
                        "status": str(result.get("status", "")),
                        "reason": str(result.get("reason", "")),
                        "topic_chars": len(topic),
                        "evidence_count": result.get("evidence_count"),
                        "source_count": result.get("source_count"),
                    },
                )
                return result

            self._emit_audit_event("delegate_started", {"topic_chars": len(topic)})
            if round_no > self.config.max_research_rounds:
                working.set_stop_reason(StopReason.GLOBAL_ROUND_BUDGET_EXHAUSTED)
                return _reported(
                    {
                        "status": "blocked",
                        "reason": "round_budget_exhausted",
                        "instruction": "研究轮次预算已耗尽；请修订最新研究综合稿。若达到完整标准则调用 ResearchComplete，否则直接结束，系统将按 partial 交付。",
                    }
                )
            async with runtime.tool_lock:
                task_index = working.allocate_task_index()
                task: SubTask = {
                    "id": f"task-{task_index:04d}",
                    "run_id": str(state.get("run_id") or ""),
                    "question": topic,
                    "round": round_no,
                    "sequence": task_index,
                    "type": "search",
                    "status": "pending",
                    "assigned_agent": "research_agent",
                    "worker_id": f"research-agent-{task_index:04d}",
                    "worker_index": task_index,
                    "parent_task_id": "",
                    "operation_id": f"research-task-{task_index:04d}",
                }
                new_tasks = working.filter_new_tasks(
                    [task], max_tasks=self.config.max_subtasks_per_round
                )
            if not new_tasks:
                working.set_stop_reason(StopReason.NO_NEW_TASKS)
                return _reported(
                    {"status": "skipped", "reason": "duplicate_or_budget", "topic": topic}
                )
            execution = await self._execute_research_task(
                new_tasks[0],
                tool_call_id=f"delegate-{new_tasks[0]['id']}",
                url_reservations=url_reservations,
            )
            async with runtime.tool_lock:
                working.absorb(execution)
            direction_report: dict[str, object] = {
                "status": execution.task_result.execution_status,
                "research_direction": execution.task_result.research_direction,
                "coverage_status": execution.task_result.coverage_status,
                "evidence_count": execution.task_result.evidence_count,
                "source_count": execution.task_result.source_count,
                "remaining_gaps": execution.task_result.remaining_gaps,
                "conclusion": execution.task_result.conclusion,
                "failures": execution.task_result.failures,
                "working_set_revision": working.working_set_revision,
                "evidence": [
                    evidence_card(item)
                    for item in execution.evidences
                    if item.evidence_id in working.active_evidence_ids
                ],
            }
            if execution.task_result.provider_exhausted:
                # 数据提示（无分支控制流）：让 Supervisor 模型读到系统性不可用后自然停止派发、收尾。
                direction_report["provider_exhausted"] = True
                direction_report["instruction"] = (
                    "搜索服务账户级不可用（额度耗尽/密钥无效），系统性问题：再派新方向也会同样失败。"
                    "停止派发 ResearchDelegate；把已有 Evidence 修订进综合稿，随后 "
                    "ResearchComplete（足以成文）或 ResearchReady（保存部分报告）收尾。"
                )
            return _reported(direction_report)

        runtime = SupervisorRuntimeContext(
            scope=AgentExecutionScope(
                run_id=str(state.get("run_id") or ""),
                agent_name="Supervisor",
            ),
            working=working,
            url_reservations=url_reservations,
            delegate_research=delegate,
            round_no=round_no,
        )
        prepared = history
        try:
            result = await cast(Any, self._agent_loop).ainvoke(
                cast(Any, {"messages": prepared}),
                context=runtime,
                config={"recursion_limit": AGENT_RECURSION_LIMIT},
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            working.set_stop_reason(StopReason.AGENT_FAILED)
            working.coverage_gaps.append(str(exc)[: self.config.supervisor_preview_chars])
        else:
            generated = result.get("messages", []) if isinstance(result, dict) else []
            history.extend(generated[len(prepared) :])
            if not working.sufficient and _model_call_limit_hit(generated):
                # 真实终止原因是模型调用天花板；set_stop_reason 的声明式 rank 保证它压过
                # 更弱的瞬时信号（如某条 delegate 撞去重留下的 NO_NEW_TASKS），
                # 又不会盖过模型的显式收尾决定。
                working.set_stop_reason(StopReason.MODEL_CALL_LIMIT_EXCEEDED)
        self._emit_round_completed(
            round_no,
            len([item for item in working.task_results if item.round == round_no]),
            working,
            outcome=working.stop_reason or "agent_loop_completed",
        )
        return self._final_update(state, working, url_reservations)

    def _append_review_rejection(self, state: ResearchState, history: list[BaseMessage]) -> None:
        """把审阅拒绝作为消息注入历史；如何响应留给工具循环里的模型。"""
        review = section(state, "review", ReviewProgress)
        history.append(
            HumanMessage(
                content=(
                    json_data_section(
                        "审阅回流",
                        {
                            "review_feedback": review.feedback,
                            "fatal_gaps": list(review.gaps),
                            "decision_rules": {
                                "rewrite": "Evidence 已覆盖核心问题，问题仅是措辞、范围、组织或已知材料利用不足；"
                                "确认综合稿仍是最新版本后调用 ResearchComplete，进入改写。",
                                "research": "核心结论缺少直接证据、来源矛盾，或必须补定义、比较对象或关键事实；"
                                "调用 ResearchDelegate 补充方向。",
                            },
                        },
                    )
                )
            )
        )

    def _append_research_observation(
        self,
        history: list[BaseMessage],
        payload: dict[str, object],
    ) -> None:
        """把轮次预算等管理信息作为轻量观察写入 Supervisor 历史。"""
        history.append(HumanMessage(content=json_data_section("研究管理观察", payload)))

    async def _execute_research_task(
        self,
        task: SubTask,
        *,
        tool_call_id: str,
        url_reservations: RunUrlReservations,
    ) -> TaskExecution:
        """执行单个方向研究；worker 异常降级为 failed 结果，不中断整轮。"""
        round_no = int(task.get("round", 1))
        task_context = {
            "task_id": task["id"],
            "worker_id": task.get("worker_id", task["id"]),
            "worker_index": int(task.get("worker_index", task.get("sequence", 0))),
            "parent_task_id": task.get("parent_task_id", ""),
            "operation_id": task.get("operation_id", task["id"]),
            "concurrency_limit": self.config.max_parallel_workers,
        }
        async with self._worker_limit:
            self._emit_audit_event(
                "research_task_started",
                {
                    **task_context,
                    "task_index": int(task.get("sequence", 0)),
                    "component": "research_agent",
                    "question": task["question"][: self.config.supervisor_preview_chars],
                    "type": task["type"],
                },
            )
            try:
                result = await self.research_agent.run(
                    task,
                    claim_url=url_reservations.reserve,
                    on_url_already_attempted=lambda url: self._emit_audit_event(
                        "source_duplicate_skipped",
                        {
                            "task_id": task["id"],
                            "url": self._normalize_source_url(url),
                            "dedup_scope": "research_run",
                        },
                    ),
                )
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
                    str(exc)[: self.config.supervisor_preview_chars],
                    tool_call_id=tool_call_id,
                )
                self._emit_audit_event(
                    "research_task_failed",
                    {**task_context, **execution.task_result.model_dump()},
                    component="research_agent",
                )
                return execution
            self._emit_audit_event(
                "research_task_completed",
                {**task_context, **task_result.model_dump()},
                component="research_agent",
            )
            selected_set = set(selected_ids)
            selected_evidences = [item for item in evidences if item.evidence_id in selected_set]
            return TaskExecution(
                task_result=task_result,
                evidences=evidences,
                selected_evidence_ids=selected_ids,
                source_refs=list(agent_result.source_refs),
                message=TaskExecution._result_message(
                    task,
                    task_result,
                    selected_evidences,
                    tool_call_id=tool_call_id,
                ),
            )

    def _emit_research_stopped(self, round_no: int, working: WorkingState) -> None:
        self._emit_audit_event(
            "research_stopped",
            {
                "round": round_no,
                "reason": str(working.stop_reason or ""),
                "evidence_count": len(working.evidences),
            },
        )

    def _emit_round_completed(
        self,
        round_no: int,
        task_count: int,
        working: WorkingState,
        *,
        outcome: str,
    ) -> None:
        self._emit_audit_event(
            "research_round_completed",
            {
                "round": round_no,
                "task_count": task_count,
                "outcome": outcome,
                "completed_tasks": sum(
                    item.execution_status == "completed"
                    for item in working.task_results
                    if item.round == round_no
                ),
                "failed_tasks": sum(
                    item.execution_status == "failed"
                    for item in working.task_results
                    if item.round == round_no
                ),
                "evidence_added": sum(
                    item.evidence_count for item in working.task_results if item.round == round_no
                ),
                "total_evidence_count": len(working.evidences),
            },
        )

    def _final_update(
        self,
        state: ResearchState,
        working: WorkingState,
        url_reservations: RunUrlReservations,
    ) -> SupervisorStateUpdate:
        """把工作状态转为 State 增量与路由决策。"""
        # 地板兜底,不是优先级判断:整轮没产生任何信号时才补一个默认终态。
        # 故意保持 `is None` + 直接赋值,不走 set_stop_reason——ROUND_BUDGET 的
        # rank 高于 NO_NEW_TASKS,若走 setter 会误盖掉"这轮全是重复 topic"的真信号。
        if not working.sufficient and working.stop_reason is None:
            working.stop_reason = StopReason.ROUND_BUDGET_EXHAUSTED
        full_synthesis = working.completed_synthesis
        latest_synthesis = working.research_synthesis
        partial_synthesis = (
            self._partial_synthesis(working, latest_synthesis)
            if working.stop_reason is not None and working.stop_reason.allows_partial_report
            else None
        )
        selected_synthesis = full_synthesis or partial_synthesis
        can_write = selected_synthesis is not None
        report_brief = (
            self._report_brief_from_synthesis(selected_synthesis)
            if selected_synthesis is not None
            else None
        )
        writer_directive = (
            self._build_writer_directive(state, working, selected_synthesis, report_brief)
            if selected_synthesis is not None and report_brief is not None
            else None
        )
        deltas = working.deltas()
        research_status = "completed" if working.sufficient else "incomplete"
        generation_mode = "full" if working.sufficient else "partial" if can_write else "not_ready"
        can_continue_to_writer = can_write
        return SupervisorStateUpdate(
            evidences=cast(list[Evidence], deltas["evidences"]),
            source_refs=cast(list[str], deltas["source_refs"]),
            task_results=cast(list[ResearchDirectionResult], deltas["task_results"]),
            attempted_source_urls=url_reservations.newly_attempted,
            working_set_revision=working.working_set_revision,
            research_synthesis=working.research_synthesis,
            report_brief=report_brief,
            writer_directive=writer_directive,
            active_evidence_ids=sorted(working.active_evidence_ids),
            run=RunLifecycle(
                phase="writing" if can_continue_to_writer else "rendering",
                terminal_reason="" if can_continue_to_writer else str(working.stop_reason or ""),
            ),
            research=ResearchProgress(
                status=research_status,
                current_round=working.current_round,
                coverage_gaps=working.coverage_gaps,
                generation_mode=generation_mode,
                is_sufficient=working.sufficient,
            ),
            writer=WriterProgress(
                status="not_started",
                feedback=(
                    ""
                    if working.sufficient
                    else self._describe_research_stop(working.stop_reason, working.coverage_gaps)
                ),
            ),
            supervisor_next=NodeName.WRITER if can_write else NodeName.RENDER_FINAL_REPORT,
        )

    def _partial_synthesis(
        self,
        working: WorkingState,
        latest: ResearchSynthesis | None,
    ) -> ResearchSynthesis | None:
        """选择可部分交付的最新综合稿；无综合稿时生成最小固定版。"""
        active_ids = set(working.active_evidence_ids)
        if latest is not None and latest.selected_evidence_ids:
            if set(latest.selected_evidence_ids).issubset(active_ids):
                return latest
        evidences = working.active_evidences()
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
            based_on_working_set_revision=working.working_set_revision,
            answer_goal=working.research_query or "回答用户的研究问题",
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

    def _build_writer_directive(
        self,
        state: ResearchState,
        working: WorkingState,
        synthesis: ResearchSynthesis,
        report_brief: ReportBrief,
    ) -> WriterDirective:
        """从冻结综合版本派生 Writer 唯一可见的写作指令。"""
        review = section(state, "review", ReviewProgress)
        previous_draft = str(state.get("report_draft") or state.get("writer_draft") or "")
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
            research_status="completed" if working.sufficient else "incomplete",
            generation_mode="full" if working.sufficient else "partial",
            evidence_ids=evidence_ids,
            known_gaps=list(dict.fromkeys([*synthesis.open_gaps, *synthesis.conflicts]))[
                : self.config.report_max_caveats
            ],
            revision_instructions=revision_instructions,
            previous_draft=previous_draft,
        )

    @staticmethod
    def _working_set_snapshot(working: WorkingState) -> dict[str, object]:
        """构造 Supervisor 工作集摘要，不把完整 quote 重复注入上下文。"""
        active = working.active_evidences()
        return {
            "active_evidence": [evidence_card(item) for item in active],
            "active_evidence_count": len(active),
        }

    @staticmethod
    def _research_synthesis_observation(
        synthesis: ResearchSynthesis | None,
    ) -> dict[str, object] | None:
        """每轮固定注入当前综合稿，避免上下文压缩后丢失研究认知。"""
        if synthesis is None:
            return None
        return synthesis_snapshot(synthesis)

    def _report_brief_from_synthesis(self, synthesis: ResearchSynthesis) -> ReportBrief:
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
            caveats=list(dict.fromkeys([*synthesis.open_gaps, *synthesis.conflicts]))[
                : self.config.report_max_caveats
            ],
        )

    @staticmethod
    def _describe_research_stop(stop_reason: StopReason | None, coverage_gaps: list[str]) -> str:
        """把研究无法继续的原因保留给最终不完整报告与事件诊断。

        文案单一来源在 ``StopReason.description``；这里只补充逐次运行的
        具体缺口细节（coverage_gaps），不再各自维护字符串清单。
        """
        detail = next(
            (gap for gap in reversed(coverage_gaps) if gap.strip()), "未形成可验证的完整覆盖。"
        )
        prefix = (
            stop_reason.description
            if stop_reason
            else "Supervisor 未确认现有材料足以形成完整研究报告。"
        )
        return f"{prefix} {detail}"

    def _emit_audit_event(
        self,
        event_type: str,
        payload: dict[str, object],
        *,
        component: str = "supervisor",
    ) -> None:
        emit_agent_event(
            self.event_sink,
            self.logger,
            event_type,
            payload,
            component=component,
            node_fallback="supervisor",
        )

    @staticmethod
    def _normalize_source_url(url: str) -> str:
        url, _ = urldefrag(url.strip())
        parts = urlsplit(url)
        query = urlencode(
            [
                (key, value)
                for key, value in parse_qsl(parts.query, keep_blank_values=True)
                if not key.casefold().startswith("utm_")
            ],
            doseq=True,
        )
        return urlunsplit(
            (parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), query, "")
        )

    @staticmethod
    def _task_deduplication_key(question: str) -> str:
        return "".join(question.lower().split())
