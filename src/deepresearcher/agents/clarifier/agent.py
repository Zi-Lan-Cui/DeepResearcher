"""Clarifier Agent：用工具调用表达询问或完成。"""

from typing import Any, cast

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel

from deepresearcher.agents.clarifier.state import (
    ClarifierAgentState,
    ClarifierGraphState,
    ClarifierLoopContext,
)
from deepresearcher.agents.clarifier.tools import (
    MAX_CLARIFICATION_ROUNDS,
    build_clarifier_tools,
)
from deepresearcher.agents.middleware import (
    AGENT_RECURSION_LIMIT,
    MiddlewareProfile,
    build_agent_middleware,
)
from deepresearcher.config import DEFAULT_CONTEXT_WINDOW_TOKENS, AgentConfig
from deepresearcher.llm import LLMConfigurationError
from deepresearcher.observability.execution import AgentExecutionScope
from deepresearcher.prompts import language_directive, load_prompt

_SYSTEM_PROMPT = load_prompt("clarifier")


class Clarifier:
    def __init__(
        self,
        llm: BaseChatModel,
        config: AgentConfig,
        *,
        context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS,
    ):
        if llm is None:
            raise LLMConfigurationError("Clarifier 需要已装配的模型。")
        self.graph = create_agent(
            model=llm,
            tools=build_clarifier_tools(),
            system_prompt=_SYSTEM_PROMPT + "\n" + language_directive(config.output_language),
            state_schema=ClarifierAgentState,
            context_schema=ClarifierLoopContext,
            middleware=cast(
                Any,
                build_agent_middleware(
                    MiddlewareProfile(
                        agent_name="Clarifier",
                        event_slug="clarifier",
                        model=llm,
                        # +4 为防失控余量,不承担业务配额:轮次上限由
                        # clarification_rounds 与程序侧守卫执行。
                        max_turns=MAX_CLARIFICATION_ROUNDS + 4,
                        context_window_tokens=context_window_tokens,
                        serial_tools={"AskClarification", "ClarificationComplete"},
                    )
                ),
            ),
            name="clarifier",
        )

    async def run(self, state: ClarifierGraphState) -> dict[str, object]:
        """注入本轮独立的调度栅栏(ToolExecutionGate)；上下文不写入 checkpoint。"""
        return await self.graph.ainvoke(
            cast(Any, state),
            context=ClarifierLoopContext(
                scope=AgentExecutionScope(
                    run_id=str(state.get("run_id") or ""),
                    agent_name="Clarifier",
                )
            ),
            config={"recursion_limit": AGENT_RECURSION_LIMIT},
        )
