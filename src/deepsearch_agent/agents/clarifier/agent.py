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
from deepsearch_agent.llm import LLMConfigurationError, LLMInvoker

_SYSTEM_PROMPT = """
你是研究意图澄清 Agent。用户原问题必须原样保留，不得擅自缩窄或改写。
你只能通过工具表达决定：必要歧义调用 AskClarification，信息足够调用 ClarificationComplete。
以下任一情况应调用 AskClarification：
1. 缺少会实质改变答案方向、且无法合理并列处理的必要选择；
2. 研究边界不确定，例如时间范围、地域、研究对象、目标受众、技术层级、对比范围或交付范围不明确，
   并且不同边界会明显改变检索材料、研究计划或最终答案。
如果边界可以从用户原话可靠推断，或可以在报告中并列覆盖并明确假设，则不必追问。
范围宽、多维分析、价值判断，或可由研究给出工作定义，都不是追问理由。
AskClarification 每次只问一题，必须给出恰好三个互斥选项，不包含 Other。
用户回答后，你自己判断关键歧义是否已解决；空洞或答非所问才可追问。
最多询问两次；额度用尽后在 assumptions 明示合理假设并调用 ClarificationComplete。
直接输出文字不算完成。
""".strip()


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
                        tool_call_limits=[
                            ("AskClarification", MAX_CLARIFICATION_ROUNDS),
                            ("ClarificationComplete", 1),
                        ],
                    )
                ),
            ),
            name="clarifier",
        )

    async def run(self, state: ClarifierGraphState) -> dict[str, object]:
        """为每次 Agent 决策注入独立工具锁；上下文不写入 checkpoint。"""
        return await self.graph.ainvoke(
            cast(Any, state),
            context=ClarifierRuntimeContext(),
            config={"recursion_limit": AGENT_RECURSION_LIMIT},
        )
