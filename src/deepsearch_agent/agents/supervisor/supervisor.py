"""动态研究 Supervisor：以工具调用循环驱动方向级研究。

模型每轮通过 ResearchDelegate 派发方向级 ResearchAgent(子 Agent 抽象为工具),
或通过 ResearchComplete 宣布现有 Evidence 足以成文;轮次预算、任务/URL 去重、
并发上限与异常降级由本地程序强制,不依赖模型自觉。
"""

import asyncio
import json
from typing import Any, cast
from urllib.parse import parse_qsl, urldefrag, urlencode, urlsplit, urlunsplit

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from deepsearch_agent.agents.middleware import (
    AGENT_RECURSION_LIMIT,
    LIMIT_MESSAGE_MARKER,
    MiddlewareProfile,
    build_agent_middleware,
)
from deepsearch_agent.agents.researcher import ResearchAgent
from deepsearch_agent.agents.supervisor.state import (
    RunUrlReservations,
    SupervisorRuntimeContext,
    TaskExecution,
    WorkingState,
)
from deepsearch_agent.agents.supervisor.tools import (
    build_supervisor_tools,
)
from deepsearch_agent.config import AgentConfig
from deepsearch_agent.evidence.models import Evidence
from deepsearch_agent.llm import LLMConfigurationError, LLMInvoker
from deepsearch_agent.observability.events import JsonlSink, emit_agent_event
from deepsearch_agent.observability.logger import get_logger
from deepsearch_agent.routing import NodeName
from deepsearch_agent.schemas import (
    CoveredTopic,
    ReportBrief,
    ResearchAgentResult,
    ResearchDirectionResult,
    ResearchProgress,
    ReviewProgress,
    RunLifecycle,
    StopReason,
    SupervisorStateUpdate,
    WriterDirective,
    WriterProgress,
)
from deepsearch_agent.state import ResearchState, SubTask, section

_SUPERVISOR_SYSTEM_PROMPT = """【身份】你是深度研究系统的 Supervisor，管理可并发的方向级 ResearchAgent。
【职责】你决定何时派发互补方向、何时现有 Evidence 足以进入写作，以及 Reflection 拒绝后应改写或补研究。
你不直接撰写报告、不伪造 Evidence，也不把来源数量当作充分性。系统会持续提供方向结果和审阅回流；
请基于完整管理历史作决定。
【工具】你有五个工具：
- ResearchDelegate：派发一个方向级研究任务。每次可派发 1 到 N 个互补方向（不超过并行上限）；
 方向必须具体、可检索、可验证，不能重述原问题，也不能重复历史已做过的方向。
 每个任务描述必须明确：研究对象、范围、待回答的局部问题、与历史任务的边界、排除项和完成标准。
 补缺时只针对当前 Evidence 暴露出的一个或几个明确缺口，缩小范围；不要重新派发一个覆盖整段历史、
 整个流派或全部对象的宽泛任务。任务是否与历史方向重复由你根据研究语义判断，不要依赖程序替你判断。
  每次调用的结果会带着该方向带回的 Evidence 事实与结论注入历史。
- ResearchComplete：宣布现有 Evidence 已足以成文，必须同时给出 report_brief
  （成文目标、覆盖主题、必要结论和限定）。只有确实充分时才调用。
- ResearchReady：记录现有 Evidence 虽未完整覆盖、但已经能够形成一篇基本成立的部分报告，必须同时给出
  report_brief。它不是停止信号；只要还有研究轮次，仍应继续补齐并优先争取 ResearchComplete。
  只有轮次耗尽或没有新方向时，才把这份部分报告交给 Writer；Writer 必须诚实说明未覆盖主题、证据限制和剩余缺口。
 - ReadWorkingSet：查看当前活跃 Evidence 的轻量摘要和数量；不返回完整 quote。
 - ForgetEvidence：将重复、偏题或当前阶段不需要的 Evidence 从活跃工作集中释放；不删除全局 Evidence 档案。
【预算约束】ResearchDelegate 受本轮派发配额与总轮次预算双重限制。若工具返回 status=blocked、
reason=round_budget_exhausted，或被本轮配额拦截：不要再尝试派发或读取工作集，
立即基于现有 Evidence 调用 ResearchComplete（足以成文）或 ResearchReady（可出部分报告）收尾。
ResearchComplete 与 ResearchReady 不能在同一轮同时使用；如果连一篇有证据支撑的基本报告都无法形成，继续派发 ResearchDelegate。
ResearchAgent 返回的 remaining_gaps 只是局部观察，不是全局结论。你必须综合原问题、所有方向结果和全部
Evidence 自己判断覆盖度；核心主题均覆盖时调用 ResearchComplete；核心主题尚未全部覆盖但已有清晰论证主线时调用 ResearchReady。"""

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
        self.logger = get_logger("deepsearch_agent.agents.supervisor")
        self._worker_limit = asyncio.Semaphore(config.max_parallel_workers)
        self._agent_loop = create_agent(
            model=cast(Any, llm),
            tools=build_supervisor_tools(),
            system_prompt=_SUPERVISOR_SYSTEM_PROMPT,
            context_schema=SupervisorRuntimeContext,
            middleware=cast(Any, build_agent_middleware(MiddlewareProfile(
                agent_name="Supervisor",
                model=getattr(self.llm, "chat_model", None),
                # 一次节点访问 = 一轮；ModelCallLimit 只是防失控天花板：
                # 一轮最多 max_subtasks_per_round 次委托 + 读工作集/决策/收尾的余量。
                # 轮次配额由 remaining_rounds 提示 + delegate() 的本地 hard check 执行。
                max_turns=config.max_subtasks_per_round + 10,
                context_window_tokens=context_window_tokens,
                retry_tools=[(["ResearchDelegate"], "ResearchDelegate")],
                serial_tools={"ReadWorkingSet", "ForgetEvidence", "ResearchComplete", "ResearchReady"},
                tool_call_limits=[("ResearchDelegate", config.max_subtasks_per_round)],
                emit=self._emit_audit_event,
            ))),
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
                    "【研究委托】\n"
                    + json.dumps(
                        {
                            "research_question": state.get(
                                "clarified_query", state.get("query", "")
                            ),
                            "research_brief": state.get("research_brief", ""),
                        },
                        ensure_ascii=False,
                    )
                )
            )
        ]

    async def run(self, state: ResearchState) -> dict[str, object]:
        """注入审阅回流（如有），然后执行有界的研究工具调用循环。

        改写还是补研究不由独立决策判定，而由工具循环里的模型直接表达：
        ResearchComplete 进入改写，ResearchDelegate 继续补研究。
        """
        history = self._build_supervisor_context(state)
        # 首次运行时 _build_supervisor_context 会补入初始 System/Human 消息；
        # 快照必须取 State 入口长度，确保这些消息也能持久化到上下文历史。
        history_start = len(state.get("supervisor_messages", []))

        review = section(state, "review", ReviewProgress)
        if review.status == "rejected":
            if review.attempts > self.config.max_post_review_recovery_cycles:
                update = SupervisorStateUpdate(
                    run=RunLifecycle(phase="rendering", terminal_reason="review_recovery_exhausted"),
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
        working = WorkingState(state, dedup_key=self._task_deduplication_key)
        research = section(state, "research", ResearchProgress)
        round_no = research.current_round + 1
        working.current_round = round_no
        self._append_research_observation(
            history,
            {"remaining_rounds": max(0, self.config.max_research_rounds - research.current_round)},
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
                working.stop_reason = StopReason.GLOBAL_ROUND_BUDGET_EXHAUSTED
                return _reported(
                    {
                        "status": "blocked",
                        "reason": "round_budget_exhausted",
                        "instruction": "研究轮次预算已耗尽；请基于现有 Evidence 立即调用 ResearchComplete 或 ResearchReady。",
                    }
                )
            async with runtime.tool_lock:
                task_index = working.allocate_task_index()
                task: SubTask = {
                    "id": f"task-{task_index:04d}",
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
                working.stop_reason = StopReason.NO_NEW_TASKS
                return _reported({"status": "skipped", "reason": "duplicate_or_budget", "topic": topic})
            execution = await self._execute_research_task(
                new_tasks[0],
                tool_call_id=f"delegate-{new_tasks[0]['id']}",
                url_reservations=url_reservations,
            )
            async with runtime.tool_lock:
                working.absorb(execution)
            return _reported(
                {
                    "status": execution.task_result.execution_status,
                    "research_direction": execution.task_result.research_direction,
                    "coverage_status": execution.task_result.coverage_status,
                    "evidence_count": execution.task_result.evidence_count,
                    "source_count": execution.task_result.source_count,
                    "answered_points": execution.task_result.answered_points,
                    "remaining_gaps": execution.task_result.remaining_gaps,
                    "conclusion": execution.task_result.conclusion,
                    "failures": execution.task_result.failures,
                    "evidence": [item.claim for item in execution.evidences],
                }
            )

        runtime = SupervisorRuntimeContext(
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
            working.stop_reason = StopReason.AGENT_FAILED
            working.coverage_gaps.append(str(exc)[: self.config.supervisor_preview_chars])
        else:
            generated = result.get("messages", []) if isinstance(result, dict) else []
            history.extend(generated[len(prepared) :])
            if (
                not working.sufficient
                and working.stop_reason is None
                and _model_call_limit_hit(generated)
            ):
                # 真实终止原因是模型调用天花板，不得再兜底标成轮次耗尽。
                working.stop_reason = StopReason.MODEL_CALL_LIMIT_EXCEEDED
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
                    "【审阅回流】\n"
                    + json.dumps(
                        {
                            "review_feedback": review.feedback,
                            "fatal_gaps": list(review.gaps),
                            "decision_rules": {
                                "rewrite": "Evidence 已覆盖核心问题，问题仅是措辞、范围、组织或已知材料利用不足；"
                                "调用 ResearchComplete 并提供 report_brief，进入改写。",
                                "research": "核心结论缺少直接证据、来源矛盾，或必须补定义、比较对象或关键事实；"
                                "调用 ResearchDelegate 补充方向。",
                            },
                        },
                        ensure_ascii=False,
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
        history.append(
            HumanMessage(content="【研究管理观察】\n" + json.dumps(payload, ensure_ascii=False))
        )

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
            return TaskExecution(
                task_result=task_result,
                evidences=evidences,
                source_refs=list(agent_result.source_refs),
                message=TaskExecution._result_message(
                    task,
                    task_result,
                    evidences,
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
        if not working.sufficient and working.stop_reason is None:
            working.stop_reason = StopReason.ROUND_BUDGET_EXHAUSTED
        meets_material_floor = self._meets_partial_report_threshold(working)
        # ResearchReady 提供语义判断，但不能绕过本地最低材料安全线；预算耗尽时，
        # 达到安全线即可兜底进入 Writer，并由 Writer 明确披露未完成部分。
        can_generate_partial = meets_material_floor and (
            working.partial_ready
            or (working.stop_reason is not None and working.stop_reason.allows_partial_report)
        )
        if working.partial_ready and not meets_material_floor:
            working.coverage_gaps.append(
                "Supervisor 判断可以形成部分报告，但当前材料未达到本地最低生成门槛。"
            )
        if can_generate_partial and working.report_brief is None:
            working.report_brief = self._build_partial_report_brief(working)
        can_write = working.sufficient or can_generate_partial
        writer_directive = self._build_writer_directive(state, working) if can_write else None
        deltas = working.deltas()
        research_status = "completed" if working.sufficient else "incomplete"
        generation_mode = (
            "full" if working.sufficient else "partial" if can_generate_partial else "not_ready"
        )
        can_continue_to_writer = can_write
        return SupervisorStateUpdate(
            evidences=cast(list[Evidence], deltas["evidences"]),
            source_refs=cast(list[str], deltas["source_refs"]),
            task_results=cast(list[ResearchDirectionResult], deltas["task_results"]),
            attempted_source_urls=url_reservations.newly_attempted,
            report_brief=working.report_brief,
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

    def _build_writer_directive(
        self,
        state: ResearchState,
        working: WorkingState,
    ) -> WriterDirective:
        """把 Supervisor 的判断整理成 Writer 唯一可见的写作指令。"""
        if working.report_brief is None:
            raise RuntimeError("WriterDirective 需要 ReportBrief。")
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
        return WriterDirective(
            query=str(state.get("clarified_query") or state.get("query") or ""),
            report_brief=working.report_brief,
            research_status="completed" if working.sufficient else "incomplete",
            generation_mode="full" if working.sufficient else "partial",
            evidence_ids=[item.evidence_id for item in working.active_evidences()],
            known_gaps=list(dict.fromkeys(working.coverage_gaps))[: self.config.report_max_caveats],
            revision_instructions=revision_instructions,
            previous_draft=previous_draft,
        )

    def _meets_partial_report_threshold(self, working: WorkingState) -> bool:
        """判断材料是否足以写一份明确标注缺口的部分报告。"""
        active_evidences = working.active_evidences()
        if len(active_evidences) < self.config.partial_report_min_evidences:
            return False
        source_count = len({item.source_url for item in active_evidences if item.source_url})
        return source_count >= self.config.partial_report_min_sources

    @staticmethod
    def _working_set_snapshot(working: WorkingState) -> dict[str, object]:
        """构造 Supervisor 工作集摘要，不把完整 quote 重复注入上下文。"""
        active = working.active_evidences()
        return {
            "active_evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "claim": item.claim,
                    "support": item.support,
                    "confidence": item.confidence,
                }
                for item in active
            ],
            "active_evidence_count": len(active),
        }

    def _build_partial_report_brief(self, working: WorkingState) -> ReportBrief:
        """预算耗尽且没有 ResearchComplete 时，为 Writer 生成保守任务书。"""
        topics = [
            CoveredTopic(
                topic=result.research_direction or result.question,
                role="已获得证据的局部方向",
                reason="该方向已返回可用于成文的 Evidence。",
                required=False,
            )
            for result in working.task_results
            if result.evidence_count > 0
        ]
        if not topics:
            topics = [
                CoveredTopic(
                    topic=working.research_query or "已有研究材料",
                    role="部分证据",
                    reason="当前仅允许基于已获得材料进行有限回答。",
                    required=False,
                )
            ]
        return ReportBrief(
            answer_goal=(
                working.research_query
                or "基于已获得 Evidence 形成一份明确标注范围和缺口的部分研究报告。"
            ),
            covered_topics=topics[: self.config.report_max_topics],
            required_points=[],
            caveats=list(dict.fromkeys(working.coverage_gaps))[: self.config.report_max_caveats],
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
        prefix = stop_reason.description if stop_reason else "Supervisor 未确认现有材料足以形成完整研究报告。"
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
