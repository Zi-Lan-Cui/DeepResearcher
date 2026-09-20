"""动态研究 Supervisor：以工具调用循环驱动方向级研究。

模型每轮通过 ResearchDelegate 派发方向级 ResearchAgent(子 Agent 抽象为工具),
或通过 ResearchComplete 宣布现有 Evidence 足以成文;轮次预算、任务/URL 去重、
并发上限与异常降级由本地程序强制,不依赖模型自觉。
"""

import asyncio
from typing import Any, cast

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
    SupervisorDeps,
    SupervisorLoopContext,
    SupervisorLoopState,
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
from deepresearcher.prompts import load_prompt, render_data_section
from deepresearcher.routing import NodeName
from deepresearcher.schemas import (
    CoveredTopic,
    ReportBrief,
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
from deepresearcher.state import ResearchState, section

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
        # 构造期定稿的稳定零件；跨并发 loop 共享只读，loop 期不再新增任何依赖。
        self._deps = SupervisorDeps(
            config=self.config,
            research_agent=self.research_agent,
            worker_limit=self._worker_limit,
            emit=self._emit_audit_event,
        )
        self._agent_loop = create_agent(
            model=cast(Any, llm),
            tools=build_supervisor_tools(),
            system_prompt=_SUPERVISOR_SYSTEM_PROMPT
            + "\n"
            + language_directive(config.output_language),
            context_schema=SupervisorLoopContext,
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
                    render_data_section("运行时环境", get_runtime_environment().payload())
                    + "\n\n---\n\n"
                    + render_data_section(
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
                    run=RunStatus(phase="rendering", terminal_reason="review_recovery_exhausted"),
                    supervisor=section(state, "supervisor", SupervisorProgress),
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
        """运行 Supervisor 标准 Agent；工具通过 loop 上下文修改 SupervisorLoopState。"""
        supervisor_progress = section(state, "supervisor", SupervisorProgress)
        round_no = supervisor_progress.current_round + 1
        loop_state = SupervisorLoopState(
            state,
            current_round=round_no,
            dedup_key=self._task_deduplication_key,
            active_evidence_limit=self.config.supervisor_max_active_evidences,
        )
        self._append_research_observation(
            history,
            {
                "remaining_rounds": max(
                    0, self.config.max_research_rounds - supervisor_progress.current_round
                ),
                "working_set_revision": loop_state.working_set_revision,
                "working_set": self._working_set_snapshot(loop_state),
                "research_synthesis": self._research_synthesis_observation(
                    loop_state.research_synthesis
                ),
            },
        )

        loop_context = SupervisorLoopContext(
            scope=AgentExecutionScope(
                run_id=str(state.get("run_id") or ""),
                agent_name="Supervisor",
            ),
            deps=self._deps,
            loop_state=loop_state,
        )
        prepared = history
        try:
            result = await cast(Any, self._agent_loop).ainvoke(
                cast(Any, {"messages": prepared}),
                context=loop_context,
                config={"recursion_limit": AGENT_RECURSION_LIMIT},
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            loop_state.set_stop_reason(StopReason.AGENT_FAILED)
            # 异常文本进 failure_details 而非 coverage_gaps：后者会被渲染为用户报告的
            # "未闭合缺口"，执行故障不得伪装成研究缺口；原文上限走截断配置。
            loop_state.failure_details.append(str(exc)[: self.config.supervisor_preview_chars])
        else:
            generated = result.get("messages", []) if isinstance(result, dict) else []
            history.extend(generated[len(prepared) :])
            if not loop_state.sufficient and _model_call_limit_hit(generated):
                # 真实终止原因是模型调用天花板；set_stop_reason 的声明式 rank 保证它压过
                # 更弱的瞬时信号（如某条 delegate 撞去重留下的 NO_NEW_TASKS），
                # 又不会盖过模型的显式收尾决定。
                loop_state.set_stop_reason(StopReason.MODEL_CALL_LIMIT_EXCEEDED)
        self._emit_round_completed(
            round_no,
            len([item for item in loop_state.task_results if item.round == round_no]),
            loop_state,
            outcome=loop_state.stop_reason or "agent_loop_completed",
        )
        return self._final_update(state, loop_state)

    def _append_review_rejection(self, state: ResearchState, history: list[BaseMessage]) -> None:
        """把审阅拒绝作为消息注入历史；如何响应留给工具循环里的模型。"""
        review = section(state, "review", ReviewProgress)
        history.append(
            HumanMessage(
                content=(
                    render_data_section(
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
        history.append(HumanMessage(content=render_data_section("研究管理观察", payload)))

    def _emit_research_stopped(self, round_no: int, loop_state: SupervisorLoopState) -> None:
        self._emit_audit_event(
            "research_stopped",
            {
                "round": round_no,
                "reason": str(loop_state.stop_reason or ""),
                "evidence_count": len(loop_state.evidences),
            },
        )

    def _emit_round_completed(
        self,
        round_no: int,
        task_count: int,
        loop_state: SupervisorLoopState,
        *,
        outcome: str,
    ) -> None:
        self._emit_audit_event(
            "research_round_completed",
            {
                "round": round_no,
                "task_count": task_count,
                "outcome": outcome,
                # 内部事件允许携带截断后的异常详情；用户报告通道不放。
                "failure_details": list(loop_state.failure_details),
                "completed_tasks": sum(
                    item.execution_status == "completed"
                    for item in loop_state.task_results
                    if item.round == round_no
                ),
                "failed_tasks": sum(
                    item.execution_status == "failed"
                    for item in loop_state.task_results
                    if item.round == round_no
                ),
                "evidence_added": sum(
                    item.evidence_count
                    for item in loop_state.task_results
                    if item.round == round_no
                ),
                "total_evidence_count": len(loop_state.evidences),
            },
        )

    def _final_update(
        self,
        state: ResearchState,
        loop_state: SupervisorLoopState,
    ) -> SupervisorStateUpdate:
        """把工作状态转为 State 增量与路由决策。"""
        # 地板兜底,不是优先级判断:整轮没产生任何信号时才补一个默认终态。
        # 故意保持 `is None` + 直接赋值,不走 set_stop_reason——ROUND_BUDGET 的
        # rank 高于 NO_NEW_TASKS,若走 setter 会误盖掉"这轮全是重复 topic"的真信号。
        if not loop_state.sufficient and loop_state.stop_reason is None:
            loop_state.stop_reason = StopReason.ROUND_BUDGET_EXHAUSTED
        full_synthesis = loop_state.completed_synthesis
        latest_synthesis = loop_state.research_synthesis
        partial_synthesis = (
            self._partial_synthesis(loop_state, latest_synthesis)
            if loop_state.stop_reason is not None and loop_state.stop_reason.allows_partial_report
            else None
        )
        selected_synthesis = full_synthesis or partial_synthesis
        can_write = selected_synthesis is not None
        writer_directive = (
            self._build_writer_directive(state, loop_state, selected_synthesis)
            if selected_synthesis is not None
            else None
        )
        research_status = "completed" if loop_state.sufficient else "incomplete"
        generation_mode = (
            "full" if loop_state.sufficient else "partial" if can_write else "not_ready"
        )
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
                is_sufficient=loop_state.sufficient,
            ),
            writer=WriterProgress(
                status="not_started",
                feedback=(
                    ""
                    if loop_state.sufficient
                    else self._describe_research_stop(
                        loop_state.stop_reason,
                        loop_state.coverage_gaps,
                        loop_state.failure_details,
                    )
                ),
            ),
            supervisor_next=NodeName.WRITER if can_write else NodeName.RENDER_FINAL_REPORT,
        )

    def _partial_synthesis(
        self,
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

    def _build_writer_directive(
        self,
        state: ResearchState,
        loop_state: SupervisorLoopState,
        synthesis: ResearchSynthesis,
    ) -> WriterDirective:
        """从冻结综合版本派生 Writer 唯一可见的写作指令。

        ReportBrief 在指令内部构造、随指令一起交接；State 顶层不再有平行的第二副本。
        """
        report_brief = self._report_brief_from_synthesis(synthesis)
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
            research_status="completed" if loop_state.sufficient else "incomplete",
            generation_mode="full" if loop_state.sufficient else "partial",
            evidence_ids=evidence_ids,
            known_gaps=list(dict.fromkeys([*synthesis.open_gaps, *synthesis.conflicts]))[
                : self.config.report_max_caveats
            ],
            revision_instructions=revision_instructions,
            previous_draft=previous_draft,
        )

    @staticmethod
    def _working_set_snapshot(loop_state: SupervisorLoopState) -> dict[str, object]:
        """构造 Supervisor 工作集摘要，不把完整 quote 重复注入上下文。"""
        active = loop_state.active_evidences()
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
    def _describe_research_stop(
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
    def _task_deduplication_key(question: str) -> str:
        return "".join(question.lower().split())
