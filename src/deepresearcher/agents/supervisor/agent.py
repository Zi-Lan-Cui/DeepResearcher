"""动态研究 Supervisor：以工具调用循环驱动方向级研究。

模型每轮通过 ResearchDelegate 派发方向级 ResearchAgent(子 Agent 抽象为工具),
或通过 ResearchComplete 宣布现有 Evidence 足以成文;轮次预算、任务/URL 去重、
并发上限与异常降级由本地程序强制,不依赖模型自觉。
"""

import asyncio
from typing import Any, cast

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from deepresearcher.agents.middleware import (
    AGENT_RECURSION_LIMIT,
    LIMIT_MESSAGE_MARKER,
    MiddlewareProfile,
    build_agent_middleware,
)
from deepresearcher.agents.researcher import ResearchAgent
from deepresearcher.agents.supervisor import services
from deepresearcher.agents.supervisor.state import (
    SupervisorDeps,
    SupervisorLoopContext,
    SupervisorLoopState,
    synthesis_card,
    working_set_snapshot,
)
from deepresearcher.agents.supervisor.tools import (
    build_supervisor_tools,
)
from deepresearcher.config import DEFAULT_CONTEXT_WINDOW_TOKENS, AgentConfig
from deepresearcher.llm import LLMConfigurationError
from deepresearcher.observability.events import JsonlSink, emit_agent_event
from deepresearcher.observability.events.names import EventName
from deepresearcher.observability.execution import AgentExecutionScope
from deepresearcher.observability.logging_config import get_logger
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
    WriterProgress,
)
from deepresearcher.state import ResearchState, section

_SUPERVISOR_SYSTEM_PROMPT = load_prompt("supervisor")


# 天花板消息恒在会话尾部数条内;窗口留一小段冗余防 after-hook 顺序变化。
_LIMIT_MESSAGE_SCAN_TAIL = 3


def _model_call_limit_hit(messages: list[BaseMessage]) -> bool:
    """判断本次 Agent 运行是否被 ModelCallLimitMiddleware 掐断而非模型正常收尾。"""
    return any(
        isinstance(message, AIMessage) and LIMIT_MESSAGE_MARKER in str(message.content)
        for message in messages[-_LIMIT_MESSAGE_SCAN_TAIL:]
    )


class ResearchSupervisor:
    """维护研究工具循环、覆盖判断、任务派发与有界并发。"""

    def __init__(
        self,
        llm: BaseChatModel,
        config: AgentConfig,
        *,
        research_agent: ResearchAgent,
        event_sink: JsonlSink | None = None,
        context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS,
    ):
        if llm is None:
            raise LLMConfigurationError("ResearchSupervisor 需要已装配的模型。")
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
            model=llm,
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
                        event_slug="supervisor",
                        model=self.llm,
                        max_turns=config.max_subtasks_per_round + 10,
                        context_window_tokens=context_window_tokens,
                        retry_tools=[["ResearchDelegate"]],
                        serial_tools={
                            "ReadWorkingSet",
                            "ReleaseEvidence",
                            "RestoreEvidence",
                            "ReviseResearchSynthesis",
                            "ResearchComplete",
                        },
                        tool_call_limits=[("ResearchDelegate", config.max_subtasks_per_round)],
                        exit_probe=lambda ctx, _state: bool(
                            ctx is not None and ctx.loop_state.completed_synthesis is not None
                        ),
                        emit=self._emit_audit_event,
                    )
                ),
            ),
            name="supervisor",
        )

    @staticmethod
    def _supervisor_history_from_state(
        state: ResearchState,
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
        history = self._supervisor_history_from_state(state)
        # 首次运行时 _supervisor_history_from_state 会补入初始 System/Human 消息；
        # 快照必须取 State 入口长度，确保这些消息也能持久化到上下文历史。
        history_start = len(state.get("supervisor_messages", []))

        review = section(state, "review", ReviewProgress)
        if review.status == "rejected":
            if review.attempts > self.config.max_post_review_recovery_cycles:
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
            active_evidence_limit=self.config.supervisor_max_active_evidences,
        )
        # 轮次预算不注入观察,由 services.delegate_research 的硬校验在执行越界时以回执告知。
        self._append_research_observation(
            history,
            {
                "working_set_revision": loop_state.working_set_revision,
                "working_set": working_set_snapshot(loop_state, include_reserve=False),
                "research_synthesis": synthesis_card(loop_state.research_synthesis),
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
                # 更弱的预算类信号，又不会盖过模型的显式收尾决定。
                loop_state.set_stop_reason(StopReason.MODEL_CALL_LIMIT_EXCEEDED)
        self._emit_round_completed(
            round_no,
            len([item for item in loop_state.task_results if item.round == round_no]),
            loop_state,
            outcome=loop_state.stop_reason or "agent_loop_completed",
        )
        return services.compose_final_update(state, loop_state, self.config)

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
        """把当前工作集与综合稿快照作为轮次观察追加进 Supervisor 历史。"""
        history.append(HumanMessage(content=render_data_section("研究管理观察", payload)))

    def _emit_round_completed(
        self,
        round_no: int,
        task_count: int,
        loop_state: SupervisorLoopState,
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
