"""Clarifier 子图与 Agent 的持久化 State。"""

import asyncio
from dataclasses import dataclass, field
from typing import Annotated, TypedDict

from langchain.agents import AgentState
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from deepresearcher.context.execution import AgentExecutionScope
from deepresearcher.schemas import RunLifecycle


@dataclass
class ClarifierRuntimeContext:
    """不进入 checkpoint 的 Clarifier 工具执行上下文。"""

    scope: AgentExecutionScope
    tool_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class ClarifierAgentState(AgentState, total=False):
    query: str
    intent_summary: str
    research_focus: list[str]
    assumptions: list[str]
    clarification_completed: bool
    clarification_rounds: int
    pending_question: str
    pending_options: list[str]


class ClarifierGraphState(TypedDict, total=False):
    run_id: str
    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    intent_summary: str
    research_focus: list[str]
    assumptions: list[str]
    clarification_completed: bool
    clarification_rounds: int
    pending_question: str
    pending_options: list[str]
    clarified_query: str
    research_brief: str
    answer_mode: str
    run: RunLifecycle
