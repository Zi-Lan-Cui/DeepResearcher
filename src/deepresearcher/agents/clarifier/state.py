"""Clarifier 子图与 Agent 的持久化 State。"""

from dataclasses import dataclass, field
from typing import Annotated

from langchain.agents import AgentState
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

# AgentState 由 typing_extensions 的 _TypedDictMeta 构造;混入 TypedDict 基
# 必须同族,否则多继承撞元类。
from typing_extensions import TypedDict

from deepresearcher.agents.middleware.concurrency import ToolExecutionGate
from deepresearcher.observability.execution import AgentExecutionScope
from deepresearcher.schemas import RunStatus


@dataclass
class ClarifierLoopContext:
    """不进入 checkpoint 的 Clarifier 工具执行上下文。

    读者:observability 取 scope(事件归因);serial_tools 取 tool_gate
    (AskClarification/ClarificationComplete 独占,与一切在飞工具互斥)。
    """

    scope: AgentExecutionScope
    tool_gate: ToolExecutionGate = field(default_factory=ToolExecutionGate)


class ClarifierDialogue(TypedDict, total=False):
    """内外图共享的澄清对话字段;字段名就是两图间 wire 契约,只定义一遍。

    单边改动只改一处、静默丢字段;两 State 各继承一次。
    """

    query: str
    intent_summary: str
    research_focus: list[str]
    assumptions: list[str]
    clarification_completed: bool
    clarification_rounds: int
    pending_question: str
    pending_options: list[str]


class ClarifierAgentState(AgentState, ClarifierDialogue, total=False):
    pass


class ClarifierGraphState(ClarifierDialogue, total=False):
    run_id: str
    messages: Annotated[list[AnyMessage], add_messages]
    # finalize 节点产出、交还主图的出口字段。
    clarified_query: str
    research_brief: str
    answer_mode: str
    run: RunStatus
