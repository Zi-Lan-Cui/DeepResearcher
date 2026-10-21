"""Clarifier 的询问/提交工具；无外部副作用。"""

import json

from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from langgraph.types import Command

from deepresearcher.agents.clarifier.state import ClarifierAgentState, ClarifierLoopContext
from deepresearcher.schemas import AskClarificationArgs, ClarificationCompleteArgs
from deepresearcher.schemas.limits import (
    CLARIFICATION_ASSUMPTION_LIMIT,
    CLARIFICATION_OPTION_COUNT,
    MAX_CLARIFICATION_ROUNDS,
    RUN_HEADLINE_MAX_CHARS,
)


def build_clarifier_tools():
    @tool("AskClarification", args_schema=AskClarificationArgs)
    async def ask_clarification(
        question: str,
        options: list[str],
        runtime: ToolRuntime[ClarifierLoopContext, ClarifierAgentState],
    ) -> Command | str:
        """暂停当前 Run，向用户提出一个关键问题，回答后由 Clarifier 判断是否已足够。"""
        choices = list(dict.fromkeys(item.strip() for item in options if item.strip()))
        if len(choices) != CLARIFICATION_OPTION_COUNT:
            return json.dumps(
                {"status": "rejected", "error": "必须给出恰好三个不重复选项。"},
                ensure_ascii=False,
            )
        rounds = int(runtime.state.get("clarification_rounds") or 0)
        if rounds >= MAX_CLARIFICATION_ROUNDS:
            query = str(runtime.state.get("query") or "")
            return Command(
                update={
                    "intent_summary": query,
                    "assumptions": ["澄清轮次已用尽，按原问题字面含义处理，保留多种合理解释。"],
                    "clarification_completed": True,
                    "messages": [
                        _tool_message(runtime, {"status": "limit_reached"}, "AskClarification")
                    ],
                },
            )
        return Command(
            update={
                "pending_question": question.strip(),
                "pending_options": choices,
                "clarification_rounds": rounds + 1,
                "messages": [
                    _tool_message(runtime, {"status": "question_staged"}, "AskClarification")
                ],
            },
        )

    @tool("ClarificationComplete", args_schema=ClarificationCompleteArgs)
    async def clarification_complete(
        headline: str,
        intent_summary: str,
        assumptions: list[str],
        runtime: ToolRuntime[ClarifierLoopContext, ClarifierAgentState],
    ) -> Command:
        """确认意图与研究边界已足以开始研究，提交标题、摘要与假设；这是 Clarifier 唯一完成信号。"""
        return Command(
            update={
                "run_headline": headline.strip()[:RUN_HEADLINE_MAX_CHARS],
                "intent_summary": intent_summary.strip(),
                "assumptions": [item.strip() for item in assumptions if item.strip()][
                    :CLARIFICATION_ASSUMPTION_LIMIT
                ],
                "clarification_completed": True,
                "messages": [
                    _tool_message(runtime, {"status": "accepted"}, "ClarificationComplete")
                ],
            },
        )

    return [ask_clarification, clarification_complete]


def _tool_message(runtime, payload: dict[str, str], tool_name: str) -> dict[str, str]:
    return {
        "role": "tool",
        "content": json.dumps(payload, ensure_ascii=False),
        "name": tool_name,
        "tool_call_id": runtime.tool_call_id,
    }
