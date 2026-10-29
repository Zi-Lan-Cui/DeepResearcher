"""动态研究 Supervisor：以工具调用循环驱动任务级研究。"""

import asyncio
from typing import Any, cast

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from deepresearcher.agents.middleware import (
    AGENT_RECURSION_LIMIT,
    LIMIT_MESSAGE_MARKER,
    MiddlewareProfile,
    StateRebuildMiddleware,
    SubmissionGuard,
    build_agent_middleware,
)
from deepresearcher.agents.researcher import Researcher
from deepresearcher.agents.supervisor import services
from deepresearcher.agents.supervisor.middleware import (
    DirectionResultReceiverMiddleware,
)
from deepresearcher.agents.supervisor.state import (
    SupervisorContext,
    SupervisorDeps,
    SupervisorRoundState,
    SupervisorState,
    pending_task_payload,
    workbench_view,
)
from deepresearcher.agents.supervisor.tools import (
    build_supervisor_tools,
)
from deepresearcher.config import DEFAULT_CONTEXT_WINDOW_TOKENS, AgentConfig
from deepresearcher.observability.events import JsonlSink, emit_agent_event
from deepresearcher.observability.events.names import EventName
from deepresearcher.observability.logging_config import get_logger
from deepresearcher.observability.scope import AgentScope
from deepresearcher.prompts import (
    get_runtime_environment,
    language_directive,
    load_prompt,
    render_data_section,
)
from deepresearcher.schemas import (
    RenderOutcome,
    ReviewProgress,
    RunStatus,
    StopReason,
    SupervisorProgress,
    SupervisorStateUpdate,
    Task,
    WriterProgress,
)
from deepresearcher.state import RunState, section

_SUPERVISOR_SYSTEM_PROMPT = load_prompt("supervisor")

# 整字段快照互覆风险的写工具必须单独一轮，如 ResearchComplete 的 completed_synthesis。
# ResearchDelegate 与 UpdateWorkbench 通道互不相交，可以同批发出，先写计划再派任务。
SUPERVISOR_SERIAL_TOOLS: frozenset[str] = frozenset(
    {
        "ReadWorkbench",
        "ResearchComplete",
        "WaitTask",
    }
)


def _may_conclude(context: Any) -> bool:
    """纯文本收尾的合法态：没有在途任务，且工作台已有可交付的草稿。

    既无草稿也无任何派发时不算收尾，退回让模型在等待、派发、建稿之间选择。
    """
    round_state = getattr(context, "round_state", None)
    if round_state is None:
        return True
    if round_state.pending_tasks:
        return False
    if round_state.research_synthesis is None and not round_state.task_results:
        return bool(
            round_state.provider_exhausted
            or round_state.stop_reason
            in {StopReason.GLOBAL_ROUND_BUDGET_EXHAUSTED, StopReason.ENDED_WITHOUT_COMPLETION}
        )
    return True


# 模型调用上限消息恒在会话尾部数条内；扫描窗口多留几条冗余，防 after-hook 顺序变化漏检。
_LIMIT_MESSAGE_SCAN_TAIL = 3

# submission_guard 驳回时注入的文案；测试据此判断「已被打回」，改文案只改这一处。
SUBMISSION_REJECTION_MESSAGE = (
    "当前不应收尾。在途任务未返回就调用 WaitTask 或继续派发；工作台为空应先 "
    "ResearchDelegate 或 UpdateWorkbench 建稿。"
    "在途未清零、无草稿无派发时不应以纯文本结束本轮。"
)


def _model_call_limit_hit(messages: list[BaseMessage]) -> bool:
    """判断本次 Agent 运行是否终止于 ModelCallLimitMiddleware 上限，而非模型正常收尾。"""
    return any(
        isinstance(message, AIMessage) and LIMIT_MESSAGE_MARKER in str(message.content)
        for message in messages[-_LIMIT_MESSAGE_SCAN_TAIL:]
    )


class Supervisor:
    """维护研究工具循环、覆盖判断、任务派发与有界并发。"""

    def __init__(
        self,
        llm: BaseChatModel,
        config: AgentConfig,
        *,
        research_agent: Researcher,
        event_sink: JsonlSink | None = None,
        context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS,
    ):
        self.llm = llm
        self.config = config
        self.research_agent = research_agent
        self.event_sink = event_sink
        self.logger = get_logger("deepresearcher.agents.supervisor")
        self._worker_limit = asyncio.Semaphore(config.supervisor_max_concurrent_tasks)
        self._deps = SupervisorDeps(
            config=self.config,
            research_agent=self.research_agent,
            worker_limit=self._worker_limit,
            emit=self._emit_audit_event,
        )
        self._agent_loop = create_agent(
            model=llm,
            tools=build_supervisor_tools(),
            system_prompt=_SUPERVISOR_SYSTEM_PROMPT
            + "\n"
            + language_directive(config.output_language),
            context_schema=SupervisorContext,
            state_schema=SupervisorState,
            middleware=cast(
                Any,
                [
                    StateRebuildMiddleware(),
                    DirectionResultReceiverMiddleware(),
                    *build_agent_middleware(
                        MiddlewareProfile(
                            agent_name="Supervisor",
                            event_slug="supervisor",
                            model=self.llm,
                            # 主动等待与补派发都占用模型回合，余量按派发上限翻倍再留收尾空间。
                            max_turns=2 * config.max_tasks_per_round + 10,
                            context_window_tokens=context_window_tokens,
                            retry_tools=[["ResearchDelegate"]],
                            serial_tools=set(SUPERVISOR_SERIAL_TOOLS),
                            tool_call_limits=[("ResearchDelegate", config.max_tasks_per_round)],
                            submission_guard=SubmissionGuard(
                                missing_submission_message=SUBMISSION_REJECTION_MESSAGE,
                                submitted_probe=_may_conclude,
                                max_rejections=2,
                            ),
                            # 结束信号读通道 completed_synthesis
                            exit_probe=lambda _ctx, state: bool(
                                state.get("completed_synthesis") is not None
                            ),
                            emit=self._emit_audit_event,
                        )
                    ),
                ],
            ),
            name="supervisor",
        )

    @staticmethod
    def _supervisor_history_from_state(
        state: RunState,
    ) -> list[BaseMessage]:
        """从图 State 恢复 Supervisor 私有上下文；历史为空时以初始委托构造首个列表。"""
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
                            "用户输入": state.get("clarified_query", state.get("query", "")),
                            "澄清问答时用户选择": state.get("clarification_log", []),
                            "Clarifier 对意图的理解摘要": state.get("intent_summary", ""),
                            "系统采用的假设": state.get("assumptions", []),
                        },
                    )
                    + "\n\n任务规划、缺口判断与派发由你基于用户原话与证据自行作出；"
                    "理解摘要与假设只是辅助信息，不构成必答清单。"
                )
            )
        ]

    async def run(self, state: RunState) -> dict[str, object]:
        """运行 Supervisor Agent 循环，直到模型收尾或异常退出。"""
        history = self._supervisor_history_from_state(state)
        # 首次运行时 _supervisor_history_from_state 会补入初始委托消息；
        # 快照长度必须取 State 入口处的历史长度，补入的消息才能一并持久化到上下文历史。
        history_start = len(state.get("supervisor_messages", []))

        # 审阅否决：轮次用尽即交付最后一版草稿，未用尽则把否决意见回流给模型重新组织。
        review = section(state, "review", ReviewProgress)
        if review.status == "rejected":
            if review.attempts >= self.config.max_review_attempts:
                update = SupervisorStateUpdate(
                    run=RunStatus(
                        phase="rendering",
                        terminal_reason=RenderOutcome.REVIEW_RECOVERY_EXHAUSTED,
                    ),
                    supervisor=section(state, "supervisor", SupervisorProgress),
                    writer=section(state, "writer", WriterProgress),
                )
                return {**update.state_update(), "supervisor_messages": history[history_start:]}
            self._append_review_rejection(state, history)
        # 核心循环
        update = await self._run_agent_loop(state, history)
        return {**update.state_update(), "supervisor_messages": history[history_start:]}

    async def _run_agent_loop(
        self,
        state: RunState,
        history: list[BaseMessage],
    ) -> SupervisorStateUpdate:
        """运行 Supervisor 标准 Agent；工具通过 loop 上下文修改 SupervisorRoundState。"""
        supervisor_progress = section(state, "supervisor", SupervisorProgress)
        round_no = supervisor_progress.current_round + 1
        round_state = SupervisorRoundState(
            state,
            current_round=round_no,
            active_evidence_limit=self.config.supervisor_max_active_evidences,
            max_research_rounds=self.config.max_research_rounds,
        )
        # 初始提供工作台快照
        research_query = round_state.research_query
        self._append_research_observation(
            history,
            {"workbench": workbench_view(round_state)},
        )

        loop_context = SupervisorContext(
            scope=AgentScope(
                run_id=str(state.get("run_id") or ""),
                agent_name="Supervisor",
            ),
            deps=self._deps,
            round_state=round_state,
            research_query=research_query,
            current_round=round_no,
        )
        prepared = history
        try:
            result = await cast(Any, self._agent_loop).ainvoke(
                cast(Any, {"messages": prepared, **round_state.initial_channel_state()}),
                context=loop_context,
                config={"recursion_limit": AGENT_RECURSION_LIMIT},
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - 回合边界，异常写回状态后按失败收尾
            # 异常路径取不到通道现值(没有返回 result)，按最后一个批内对象收尾；
            # 该对象含上一批合并后的状态与本批已完成工具的改动。
            round_state = loop_context.round_state
            round_state.set_stop_reason(StopReason.AGENT_FAILED)
            # 异常文本进 failure_details：它只作为收尾诊断语，不会渲染成研究内容缺口。
            # 文本长度由截断配置控制。
            round_state.failure_details.append(str(exc)[: self.config.supervisor_preview_chars])
        else:
            generated = result.get("messages", []) if isinstance(result, dict) else []
            history.extend(generated[len(prepared) :])
            round_state = SupervisorRoundState.from_channel_values(
                result if isinstance(result, dict) else {},
                research_query=research_query,
                current_round=round_no,
                active_evidence_limit=self.config.supervisor_max_active_evidences,
                max_research_rounds=self.config.max_research_rounds,
            )
            if round_state.completed_synthesis is None and _model_call_limit_hit(generated):
                # 真实终止原因是模型调用上限；set_stop_reason 的声明式 rank 保证它强于
                # 更弱的预算类信号，又不会盖过 COMPLETED 的显式收尾决定。
                round_state.set_stop_reason(StopReason.MODEL_CALL_LIMIT_EXCEEDED)
            await self._drain_task_handles(loop_context, round_state)
        finally:
            await self._stop_residual_tasks(loop_context)
        self._emit_round_completed(
            round_no,
            len([item for item in round_state.task_results if item.round == round_no]),
            round_state,
            outcome=round_state.stop_reason or "agent_loop_completed",
        )
        return services.compose_final_update(state, round_state, self.config)

    async def _drain_task_handles(
        self,
        loop_context: SupervisorContext,
        round_state: SupervisorRoundState,
    ) -> None:
        """节点异常终止，将已注册的在途任务结算到 round_state；不再派发新任务。"""
        registry = loop_context.task_handles
        # 驳回封顶后模型仍直接结束、或崩溃恢复后循环不再进模型：未结算的在途任务在收尾处统一结算。
        for _report, _delta in services.absorb_lost_registrations(
            loop_context.deps, round_state, set(registry)
        ):
            pass
        for task_id in sorted(list(registry)):
            handle = registry.pop(task_id)
            try:
                execution = await handle
            except asyncio.CancelledError:
                registry[task_id] = handle
                raise
            entry = round_state.pending_tasks.get(task_id) or {}
            task = cast(Task, pending_task_payload(entry) or {"id": task_id, "question": ""})
            services.absorb_arrival(
                loop_context.deps, round_state, task, execution, merged_via="drain"
            )

    async def _stop_residual_tasks(self, loop_context: SupervisorContext) -> None:
        """异常/取消退出路径的孤儿任务终止：取消并等待结束，任务不得逃逸节点生命周期。"""
        leftovers = [item for item in loop_context.task_handles.values() if not item.done()]
        for item in leftovers:
            item.cancel()
        if not leftovers:
            return
        try:
            await asyncio.wait(leftovers)
        except asyncio.CancelledError:
            # 退出期间的再次取消：任务已收到取消请求，让它们自行结束，不再上抛打断解栈。
            pass
        loop_context.task_handles.clear()

    def _append_review_rejection(self, state: RunState, history: list[BaseMessage]) -> None:
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
                                "调用 ResearchDelegate 补充任务。",
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
        """把工作台现值作为轮次观察追加进 Supervisor 历史。"""
        history.append(HumanMessage(content=render_data_section("研究管理观察", payload)))

    def _emit_round_completed(
        self,
        round_no: int,
        task_count: int,
        round_state: SupervisorRoundState,
        *,
        outcome: str,
    ) -> None:
        self._emit_audit_event(
            EventName.RESEARCH_ROUND_COMPLETED,
            {
                "round": round_no,
                "task_count": task_count,
                "outcome": outcome,
                # 内部事件允许携带截断后的异常详情；用户报告通道不放。
                "failure_details": list(round_state.failure_details),
                "completed_tasks": sum(
                    item.execution_status == "completed"
                    for item in round_state.task_results
                    if item.round == round_no
                ),
                "failed_tasks": sum(
                    item.execution_status == "failed"
                    for item in round_state.task_results
                    if item.round == round_no
                ),
                "evidence_added": sum(
                    item.evidence_count
                    for item in round_state.task_results
                    if item.round == round_no
                ),
                "total_evidence_count": len(round_state.evidences),
            },
        )

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
