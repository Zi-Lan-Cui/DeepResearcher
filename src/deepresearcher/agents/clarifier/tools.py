"""Clarifier 的询问/提交工具；无外部副作用。"""

import json
import re

from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from langgraph.types import Command
from pydantic import BaseModel, Field, field_validator

from deepresearcher.agents.clarifier.constants import (
    CLARIFICATION_ASSUMPTION_LIMIT,
    CLARIFICATION_FOCUS_LIMIT,
    CLARIFICATION_INTENT_MAX_CHARS,
    CLARIFICATION_OPTION_COUNT,
    CLARIFICATION_QUESTION_MAX_CHARS,
    MAX_CLARIFICATION_ROUNDS,
)
from deepresearcher.agents.clarifier.state import ClarifierAgentState, ClarifierLoopContext


def _normalize_string_list(value: object) -> object:
    """修复模型把 JSON 字符串数组再次编码成字符串的常见偏差。"""
    if not isinstance(value, str):
        return value
    raw = value.strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        return parsed
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1].strip()
    # 兼容内部引号未转义、但数组元素分隔符仍明确的模型输出。
    parts = re.split(r'["”]\s*,\s*["“]', raw)
    return [item.strip().strip('"“”') for item in parts if item.strip().strip('"“”')]


class AskClarificationArgs(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=CLARIFICATION_QUESTION_MAX_CHARS,
        description=(
            "向用户提出的一个简短、具体问题，只解决一个会改变研究方向或研究边界的关键决策。"
            "不得询问能够通过检索自行回答的问题。"
        ),
    )
    options: list[str] = Field(
        min_length=CLARIFICATION_OPTION_COUNT,
        max_length=CLARIFICATION_OPTION_COUNT,
        description=(
            "恰好三个互斥、具体、可直接选择的默认答案；覆盖最可能的三种意图。"
            "不要加入‘其他/Other’，界面会自动提供自由输入。"
        ),
    )

    @field_validator("options", mode="before")
    @classmethod
    def normalize_options(cls, value: object) -> object:
        return _normalize_string_list(value)


class ClarificationCompleteArgs(BaseModel):
    intent_summary: str = Field(
        min_length=1,
        max_length=CLARIFICATION_INTENT_MAX_CHARS,
        description="对已确认用户意图和研究边界的准确摘要；不得改变或缩窄用户已经明确的要求。",
    )
    research_focus: list[str] = Field(
        default_factory=list,
        max_length=CLARIFICATION_FOCUS_LIMIT,
        description="交给 Supervisor 的核心研究重点，最多四项；只写用户已明确或回答确认的重点。",
    )
    assumptions: list[str] = Field(
        default_factory=list,
        max_length=CLARIFICATION_ASSUMPTION_LIMIT,
        description="仍需由系统采用的显式假设，最多三项；没有假设时传空列表。",
    )

    @field_validator("research_focus", "assumptions", mode="before")
    @classmethod
    def normalize_string_lists(cls, value: object) -> object:
        return _normalize_string_list(value)


def build_clarifier_tools():
    @tool("AskClarification", args_schema=AskClarificationArgs)
    async def ask_clarification(
        question: str,
        options: list[str],
        runtime: ToolRuntime[ClarifierLoopContext, ClarifierAgentState],
    ) -> Command | str:
        """暂停当前 Run 并向真人询问一个关键问题。

        当必要选择缺失，或时间、地域、对象、受众、技术层级、对比范围、交付范围等
        研究边界不确定，而且不同答案会明显改变检索材料、研究计划或最终结论时使用。
        不要用它询问可通过研究自行解决的事实，也不要因为问题宽泛或包含多个维度就追问。
        调用后系统会保存 checkpoint，展示三个默认选项和 Other，并在用户回答后再次交给
        Clarifier 判断是否已经足够；它不是提交研究任务或输出最终答案的工具。
        """
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
                    "assumptions": ["澄清轮次已用尽，按原问题并列覆盖合理解释。"],
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
        intent_summary: str,
        research_focus: list[str],
        assumptions: list[str],
        runtime: ToolRuntime[ClarifierLoopContext, ClarifierAgentState],
    ) -> Command:
        """确认用户意图与研究边界已经足以规划研究，并提交结构化研究简报。

        当原问题本身已经明确，或用户回答已消除关键歧义时使用。可在 assumptions 中记录
        不影响继续研究的合理假设。不得用普通文本结束；这是 Clarifier 的唯一完成信号。
        """
        return Command(
            update={
                "intent_summary": intent_summary.strip(),
                "research_focus": [item.strip() for item in research_focus if item.strip()][
                    :CLARIFICATION_FOCUS_LIMIT
                ],
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
