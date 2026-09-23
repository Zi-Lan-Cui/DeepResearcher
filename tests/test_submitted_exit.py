"""SubmittedExit 契约回归:真跑 create_agent 内层循环。

clarifier/writer/researcher 的既有单测把 agent 循环整体 stub 掉,"提交后多跑一圈、
Command(goto=END) 被 pregel 丢弃"这类循环层缺陷在 stub 测法下不可见;本文件是唯一
真跑内层循环的地方,覆盖三种行为:staged 即出环(不多跑)、被拒留环、改对后出环。
"""

import asyncio
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentState
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.types import Command
from typing_extensions import NotRequired

from deepresearcher.agents.clarifier import Clarifier
from deepresearcher.agents.clarifier.state import ClarifierLoopContext
from deepresearcher.agents.middleware.tool_loop_guard import SubmittedExitMiddleware
from deepresearcher.config import AgentConfig
from deepresearcher.observability.execution import AgentExecutionScope


class ScriptedModel:
    """按脚本逐回合返回响应并统计调用次数;超出脚本后重复末项(用于断言不该再被调)。"""

    def __init__(self, replies: list[Any]):
        self._replies = list(replies)
        self.calls = 0

    def bind_tools(self, _tools, **_kwargs):
        return self

    async def ainvoke(self, _messages, **_kwargs):
        self.calls += 1
        reply = self._replies[min(self.calls - 1, len(self._replies) - 1)]
        return reply() if callable(reply) else reply


class StageState(AgentState, total=False):
    pending_question: NotRequired[str]
    committed: NotRequired[bool]


def test_staged_commit_exits_loop_without_extra_model_call():
    """提交写入 state 通道后,下一跳 before_model 直接出环——模型一次都不许多跑。

    修复前的形态(Command(goto=END) 无 return_direct)在此会多跑一圈:goto 被 pregel
    丢弃、边级规则把消息送回模型。
    """

    @tool("stage_question")
    async def stage_question(runtime: ToolRuntime[Any]) -> Command:
        """把问题写入 state 通道表示已提交。"""
        return Command(
            update={
                "pending_question": "研究对象是什么？",
                "messages": [
                    {
                        "role": "tool",
                        "content": "question_staged",
                        "tool_call_id": runtime.tool_call_id,
                        "name": "stage_question",
                    }
                ],
            }
        )

    model = ScriptedModel(
        [
            AIMessage(content="", tool_calls=[{"name": "stage_question", "args": {}, "id": "c1"}]),
            AIMessage(content="IDLE_SPIN_MUST_NOT_HAPPEN"),
        ]
    )
    agent = create_agent(
        model=model,
        tools=[stage_question],
        state_schema=StageState,
        middleware=[
            SubmittedExitMiddleware(lambda ctx, state: bool(state.get("pending_question")))
        ],
    )
    result = asyncio.run(agent.ainvoke({"messages": [HumanMessage("go")]}))

    assert model.calls == 1
    assert result["pending_question"] == "研究对象是什么？"
    assert "IDLE_SPIN" not in str(result["messages"][-1].content)


def test_rejected_receipt_stays_in_loop_until_accepted():
    """被拒回执(纯 str、不写 state)送回模型改正;改正后写 state,下一跳出环。"""

    @tool("commit")
    async def commit(runtime: ToolRuntime[Any]) -> Any:
        """历史里没有 accepted 痕迹时先拒一次,再接受。"""
        messages = runtime.state["messages"]
        already_rejected = any(
            str(getattr(message, "content", "")) == "rejected:fix-ids" for message in messages
        )
        if not already_rejected:
            return "rejected:fix-ids"
        return Command(
            update={
                "committed": True,
                "messages": [
                    {
                        "role": "tool",
                        "content": "accepted",
                        "tool_call_id": runtime.tool_call_id,
                        "name": "commit",
                    }
                ],
            }
        )

    model = ScriptedModel(
        [lambda: AIMessage(content="", tool_calls=[{"name": "commit", "args": {}, "id": "c1"}])]
    )
    agent = create_agent(
        model=model,
        tools=[commit],
        state_schema=StageState,
        middleware=[SubmittedExitMiddleware(lambda ctx, state: bool(state.get("committed")))],
    )
    result = asyncio.run(agent.ainvoke({"messages": [HumanMessage("go")]}))

    assert model.calls == 2  # 拒绝恰好多花一次(改正),接受后不再多跑
    assert result["committed"] is True


def test_clarifier_stages_question_with_single_model_call():
    """真实 Clarifier 内环接线:AskClarification staged 后不得再叫模型。

    这是修复前必然失败、修复后稳定的接线级回归测试;ask 节点复位 pending_question 放行
    后续轮次由子图路由测试保障,不在本文件范围。
    """

    def ask_call(tool_call_id: str) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "AskClarification",
                    "args": {"question": "要研究哪个版本？", "options": ["v1", "v2", "v3"]},
                    "id": tool_call_id,
                }
            ],
        )

    model = ScriptedModel([ask_call("c1"), AIMessage(content="IDLE_SPIN_MUST_NOT_HAPPEN")])
    clarifier = Clarifier(model, AgentConfig(), context_window_tokens=32_000)
    context = ClarifierLoopContext(
        scope=AgentExecutionScope(run_id="run-d", agent_name="Clarifier")
    )
    result = asyncio.run(
        clarifier.graph.ainvoke(
            {"messages": [HumanMessage("研究一下")]},
            context=context,  # type: ignore[arg-type]
        )
    )

    assert model.calls == 1
    assert result.get("pending_question") == "要研究哪个版本？"
    assert result.get("clarification_rounds") == 1
    assert "IDLE_SPIN" not in str(result["messages"][-1].content)
