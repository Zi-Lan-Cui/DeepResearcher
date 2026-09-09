"""Clarifier Agent：用工具调用表达询问或完成。"""

from typing import Any, cast

from langchain.agents import create_agent

from deepsearch_agent.agents.clarifier.state import (
    ClarifierAgentState,
    ClarifierGraphState,
    ClarifierRuntimeContext,
)
from deepsearch_agent.agents.clarifier.tools import (
    MAX_CLARIFICATION_ROUNDS,
    build_clarifier_tools,
)
from deepsearch_agent.agents.middleware.factory import (
    AGENT_RECURSION_LIMIT,
    build_agent_middleware,
)
from deepsearch_agent.agents.middleware.profile import MiddlewareProfile
from deepsearch_agent.config import AgentConfig, language_directive
from deepsearch_agent.context.execution import AgentExecutionScope
from deepsearch_agent.llm import LLMConfigurationError, LLMInvoker
from deepsearch_agent.prompts import load_prompt

_SYSTEM_PROMPT = load_prompt("clarifier")


class Clarifier:
    def __init__(
        self,
        llm: LLMInvoker,
        config: AgentConfig,
        *,
        context_window_tokens: int = 32_768,
    ):
        if llm is None:
            raise LLMConfigurationError("Clarifier 需要已装配的 LLMInvoker。")
        self.graph = create_agent(
            model=cast(Any, llm),
            tools=build_clarifier_tools(),
            system_prompt=_SYSTEM_PROMPT + "\n" + language_directive(config.output_language),
            state_schema=ClarifierAgentState,
            context_schema=ClarifierRuntimeContext,
            middleware=cast(
                Any,
                build_agent_middleware(
                    MiddlewareProfile(
                        agent_name="Clarifier",
                        model=getattr(llm, "chat_model", None),
                        max_turns=MAX_CLARIFICATION_ROUNDS + 4,
                        context_window_tokens=context_window_tokens,
                        serial_tools={"AskClarification", "ClarificationComplete"},
                    )
                ),
            ),
            name="clarifier",
        )

    async def run(self, state: ClarifierGraphState) -> dict[str, object]:
        """为每次 Agent 决策注入独立工具锁；上下文不写入 checkpoint。"""
        return await self.graph.ainvoke(
            cast(Any, state),
            context=ClarifierRuntimeContext(
                scope=AgentExecutionScope(
                    run_id=str(state.get("run_id") or ""),
                    agent_name="Clarifier",
                )
            ),
            config={"recursion_limit": AGENT_RECURSION_LIMIT},
        )
