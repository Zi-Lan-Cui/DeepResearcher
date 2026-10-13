"""Clarifier 子图与 Agent 的持久化 State。"""

from dataclasses import dataclass
from operator import add
from typing import Annotated

from langchain.agents import AgentState
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

# AgentState 由 typing_extensions 的 _TypedDictMeta 构造;混入的 TypedDict 基类
# 必须来自同一来源,否则多继承会因元类冲突报错。
from typing_extensions import TypedDict

from deepresearcher.observability.execution import AgentExecutionScope
from deepresearcher.schemas import RunStatus


@dataclass
class ClarifierLoopContext:
    """不进入 checkpoint 的 Clarifier 工具执行上下文。

    消费方:observability 取 scope 做事件归因。AskClarification 与 ClarificationComplete
    是串行工具,必须单独一轮,与其他工具同批会被 SerialToolMiddleware 整批驳回。
    """

    scope: AgentExecutionScope


class ClarifierDialogue(TypedDict, total=False):
    """内外图共享的澄清对话字段,两个 State 类各继承一次;字段名就是两图间的数据契约,只定义一处。

    若只改单边而字段名对不上,跨图传递会静默丢字段,所以两侧改动必须同时核对这里的名字。
    """

    query: str
    intent_summary: str
    assumptions: list[str]
    clarification_completed: bool
    clarification_rounds: int
    pending_question: str
    pending_options: list[str]
    # 逐轮澄清问答留痕 {question, options, answer};reducer 只在外层图的
    # Annotated 通道生效,内层 Agent 不写它。
    clarification_log: list[dict[str, object]]
    # ClarificationComplete 提交的历史浓缩标题;必须在此数据契约中声明,
    # 未声明的键会被子图按未知通道静默丢弃,主图取不到值,headline 无法落库。
    run_headline: str


class ClarifierAgentState(AgentState, ClarifierDialogue, total=False):
    pass


class ClarifierGraphState(ClarifierDialogue, total=False):
    run_id: str
    messages: Annotated[list[AnyMessage], add_messages]
    # ask 节点逐轮追加问答留痕，需要外层图的 add reducer。
    clarification_log: Annotated[list[dict[str, object]], add]
    # finalize 节点产出、交还主图的出口字段。
    clarified_query: str
    answer_mode: str
    run: RunStatus
