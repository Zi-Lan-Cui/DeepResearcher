"""Agent 模型调用前的上下文准备策略。"""

from collections.abc import Sequence
from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from deepsearch_agent.context.budget import MessageBudget
from deepsearch_agent.context.protocol import message_groups, validate_tool_call_pairs
from deepsearch_agent.llm import LLMInvoker

AgentName = Literal["supervisor", "researcher", "writer"]


class ContextPolicy:
    """在不改变事实 State 的前提下准备一次模型调用的消息。

    超出预算时优先保留最近的完整工具事务；被移除的旧消息可由独立文本调用
    压缩成一条工作记忆摘要。摘要不是事实来源，失败时只降级为裁剪。
    """

    def __init__(
        self,
        *,
        max_tokens: int | None = None,
        summarizer: LLMInvoker | None = None,
        summary_budget_tokens: int = 1_000,
    ) -> None:
        self.max_tokens = max_tokens if max_tokens and max_tokens > 0 else None
        self.summarizer = summarizer
        self.summary_budget_tokens = max(128, summary_budget_tokens)
        self.budget = MessageBudget()

    def prepare(
        self,
        messages: Sequence[BaseMessage],
        *,
        agent: AgentName,
    ) -> list[BaseMessage]:
        del agent  # 为后续 Agent 专属策略保留统一入口。
        prepared = list(messages)
        validate_tool_call_pairs(prepared)
        if self.max_tokens is None or self.budget.count(prepared) <= self.max_tokens:
            return prepared
        return self._trim_preserving_protocol(prepared)

    async def aprepare(
        self,
        messages: list[BaseMessage],
        *,
        agent: AgentName,
    ) -> list[BaseMessage]:
        """异步准备上下文；超预算时额外生成一次旧消息摘要。"""
        del agent
        validate_tool_call_pairs(messages)
        if self.max_tokens is None or self.budget.count(messages) <= self.max_tokens:
            return messages
        compacted, removed = self._select_recent_messages(messages)
        summary = await self._summarize(removed)
        if summary:
            system_count = len([item for item in compacted if isinstance(item, SystemMessage)])
            compacted.insert(system_count, HumanMessage(content=f"【历史工作记忆摘要】\n{summary}"))
        validate_tool_call_pairs(compacted)
        messages[:] = compacted
        return messages

    def _trim_preserving_protocol(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        result, _ = self._select_recent_messages(messages)
        validate_tool_call_pairs(result)
        return result

    def _select_recent_messages(
        self, messages: list[BaseMessage]
    ) -> tuple[list[BaseMessage], list[BaseMessage]]:
        system = [message for message in messages if isinstance(message, SystemMessage)]
        non_system = [message for message in messages if not isinstance(message, SystemMessage)]
        groups = message_groups(non_system)
        selected: list[list[BaseMessage]] = []
        used = self.budget.count(system)
        assert self.max_tokens is not None
        target = max(1, int(self.max_tokens * 0.7))
        for group in reversed(groups):
            group_tokens = self.budget.count(group)
            if selected and used + group_tokens > target:
                break
            selected.append(group)
            used += group_tokens
            if used >= target:
                break
        selected_messages = [message for group in reversed(selected) for message in group]
        kept = system + selected_messages
        removed = [message for group in groups[: len(groups) - len(selected)] for message in group]
        return kept, removed

    async def _summarize(self, messages: Sequence[BaseMessage]) -> str:
        if not messages or self.summarizer is None:
            return ""
        prompt = [
            SystemMessage(
                content=(
                    "将下面的旧 Agent 工作记录压缩成简短的工作记忆。只保留已发生的任务、"
                    "决策、工具结果、失败原因和待处理事项；不得新增事实，不要复述完整网页或 Evidence quote。"
                )
            ),
            HumanMessage(content="\n\n".join(str(item.content) for item in messages)),
        ]
        try:
            response = await self.summarizer.ainvoke_text(
                prompt,
                request_kwargs={"max_tokens": self.summary_budget_tokens},
            )
        except Exception:
            return ""
        content = getattr(response, "content", response)
        return content.strip() if isinstance(content, str) else str(content).strip()
