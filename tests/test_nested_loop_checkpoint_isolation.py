"""嵌套循环 checkpointer 的继承合同:内层从自己的回合边界续跑,而非整体重放。

create_agent 的 checkpointer 默认 None,表示继承外层运行的 thread/checkpointer。
合同内容:
  1. 内层循环把业务状态写进自己的 state 通道后,进程崩溃重放从内层最后一个已提交
     superstep 续跑:已完成的工具不重跑、其通道增量不重复施加。两级(外层节点直接
     调内层)与三级(外层节点→supervisor create_agent→Delegate 工具→researcher
     create_agent)皆然;
  2. 内层已跑完、外层节点尚未提交时,重放再次以 seed 调用内层不会重新开场:既不重跑
     内层模型,也不把 seed 再写一遍通道,故入口无需额外的 resume 守卫;langgraph 的
     pending-writes WAL 已覆盖该窗口;
  3. checkpointer=False 时内层不落检查点,重放整体重跑本节点,消息与业务状态同时重置,
     是尚未通道化循环的过渡语义。

supervisor、researcher、writer 已将业务状态并入各自通道,使用继承(None)。调整这些循环时
不得把 clarifier 一并改为不继承:其 interrupt 在 awaiting_input 澄清后的恢复依赖继承链,
文件末尾有单独校验用例。
"""

import asyncio

import pytest
from langchain.agents import AgentState, create_agent
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from typing_extensions import Annotated, TypedDict

from deepresearcher.agents.clarifier import Clarifier
from deepresearcher.agents.researcher import ResearchAgent
from deepresearcher.agents.supervisor import ResearchSupervisor
from deepresearcher.agents.writer import ReportWriter
from deepresearcher.config import AgentConfig


def _reducer_sum(current, incoming):
    return (current or 0) + (incoming or 0)


def _reducer_append(current, incoming):
    return [*(current or []), *(incoming or [])]


class _CountedState(AgentState):
    """内层业务通道:evidence 计数与一条裸 append 列表,用于暴露 seed 是否被二次施加。"""

    evidence_count: Annotated[int, _reducer_sum]
    seeded: Annotated[list, _reducer_append]


class _CrashModel:
    """第一轮派工具、工具结果回来后第二轮抛异常,模拟模型回合中途进程终止。"""

    def __init__(self, history_shapes: list[int], role="child") -> None:
        self.history_shapes = history_shapes
        self.role = role

    def bind_tools(self, _tools, **_kwargs):
        return self

    async def ainvoke(self, messages):
        self.history_shapes.append(len(messages))
        if len(messages) == 1:
            return AIMessage(
                content="",
                tool_calls=[{"name": "AddEvidence", "args": {}, "id": "call-ev-1"}],
            )
        raise asyncio.CancelledError("simulated mid-turn process kill")


def _build_child(checkpointer, evidence_runs: list[int], history_shapes: list[int]):
    """内层循环:一个把 evidence 写进自己通道的工具 + 派一次工具后就崩溃的模型。"""

    @tool("AddEvidence")
    async def add_evidence(runtime: ToolRuntime) -> Command:
        """取证并把计数写进内层 state 通道。"""
        evidence_runs.append(1)
        return Command(
            update={
                "evidence_count": 1,
                "messages": [
                    ToolMessage("evidence-1", tool_call_id=runtime.tool_call_id, name="AddEvidence")
                ],
            }
        )

    return create_agent(
        model=_CrashModel(history_shapes),
        tools=[add_evidence],
        state_schema=_CountedState,
        checkpointer=checkpointer,
        name="inner",
    )


# 两级嵌套:外层节点函数调用内层 create_agent。


def _build_pair(child_checkpointer):
    """外层单节点图带 thread 检查点,节点函数调内层 create_agent,与生产同形。"""
    history_shapes: list[int] = []
    evidence_runs: list[int] = []
    child = _build_child(child_checkpointer, evidence_runs, history_shapes)

    class _Outer(TypedDict):
        done: bool

    async def outer_node(_state):
        # 与生产调用点同形:只显式传 recursion_limit,thread 由上下文继承。
        await child.ainvoke(
            {"messages": [HumanMessage(content="go")], "seeded": ["x"]},
            config={"recursion_limit": 50},
        )
        return {"done": True}

    outer = StateGraph(_Outer)
    outer.add_node("sup", outer_node)
    outer.add_edge(START, "sup")
    outer.add_edge("sup", END)
    saver = MemorySaver()
    return outer.compile(checkpointer=saver), saver, history_shapes, evidence_runs


def test_inherited_child_checkpointer_resumes_history_on_replay():
    """继承合同(两级):重放从子图检查点续跑,已完成工具不重跑、seed 通道不重复施加。"""
    app, saver, history_shapes, evidence_runs = _build_pair(None)
    config = {"configurable": {"thread_id": "inherit-case"}}

    with pytest.raises(BaseException):
        asyncio.run(app.ainvoke({"done": False}, config))
    assert history_shapes == [1, 3]
    assert len(evidence_runs) == 1

    with pytest.raises(BaseException):
        asyncio.run(app.ainvoke(None, config))

    # 续跑证据:重放的模型回合直接看到已含 ToolMessage 的旧历史(3 条起步),
    # 已完成工具没有重跑(pending writes 记有结果)。
    assert history_shapes[2:] and history_shapes[2] >= 3
    assert len(evidence_runs) == 1
    # seed 的裸 append 通道未被二次施加:重放后 seeded 仍为单元素。
    seeded_sizes = [
        len((checkpoint_tuple.checkpoint or {}).get("channel_values", {}).get("seeded", []))
        for checkpoint_tuple in saver.list(config)
        if "seeded" in (checkpoint_tuple.checkpoint or {}).get("channel_values", {})
    ]
    assert seeded_sizes and max(seeded_sizes) == 1


def test_disabled_child_checkpointer_replays_whole_node():
    """禁用合同:checkpointer=False 时子图不落检查点,重放整体重跑,状态自洽。"""
    app, saver, history_shapes, evidence_runs = _build_pair(False)
    config = {"configurable": {"thread_id": "isolated-case"}}

    with pytest.raises(BaseException):
        asyncio.run(app.ainvoke({"done": False}, config))
    assert history_shapes == [1, 3]

    with pytest.raises(BaseException):
        asyncio.run(app.ainvoke(None, config))

    # 整体重跑证据:第二次尝试从初始消息重新开场,取证工具重新执行;
    # 消息历史与业务状态同时重置,不存在不一致窗口。
    third = history_shapes.index(1, 2)
    assert history_shapes[third:] == [1, 3]
    assert len(evidence_runs) == 2
    nested_ns = [
        (tuple_.config or {}).get("configurable", {}).get("checkpoint_ns")
        for tuple_ in saver.list(config)
    ]
    assert not any(nested_ns)


def test_completed_child_not_reinjected_when_outer_uncommitted():
    """内层已完整、外层未提交的窗口:重放不再跑内层模型、不再施加 seed,入口守卫因此多余。

    把内层模型换成"派工具后正常收尾"的可完成模型,外层节点在内层返回之后、return
    之前抛错;重放外层节点会再次调用内层,但内层在其 ns 已完成,langgraph 从 WAL 直接
    取结果:模型不再被调用、evidence 不再增长、seeded 通道保持单元素。
    """
    history_shapes: list[int] = []
    evidence_runs: list[int] = []

    @tool("AddEvidence")
    async def add_evidence(runtime: ToolRuntime) -> Command:
        """取证一次并把计数写进通道,然后让模型正常收尾。"""
        evidence_runs.append(1)
        return Command(
            update={
                "evidence_count": 1,
                "messages": [
                    ToolMessage("ok", tool_call_id=runtime.tool_call_id, name="AddEvidence")
                ],
            }
        )

    class _FinishModel:
        def bind_tools(self, _tools, **_kwargs):
            return self

        async def ainvoke(self, messages):
            history_shapes.append(len(messages))
            if len(messages) == 1:
                return AIMessage(
                    content="",
                    tool_calls=[{"name": "AddEvidence", "args": {}, "id": "call-ev-1"}],
                )
            return AIMessage(content="all done")

    child = create_agent(
        model=_FinishModel(),
        tools=[add_evidence],
        state_schema=_CountedState,
        checkpointer=None,
        name="inner",
    )

    class _Outer(TypedDict):
        done: bool

    armed = {"value": True}

    async def outer_node(_state):
        await child.ainvoke(
            {"messages": [HumanMessage(content="go")], "seeded": ["x"]},
            config={"recursion_limit": 50},
        )
        if armed["value"]:
            raise RuntimeError("simulated crash after inner completed, before outer commit")
        return {"done": True}

    outer = StateGraph(_Outer)
    outer.add_node("sup", outer_node)
    outer.add_edge(START, "sup")
    outer.add_edge("sup", END)
    saver = MemorySaver()
    app = outer.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "completed-window"}}

    with pytest.raises(RuntimeError):
        asyncio.run(app.ainvoke({"done": False}, config))
    assert history_shapes == [1, 3]
    assert len(evidence_runs) == 1

    armed["value"] = False
    final = asyncio.run(app.ainvoke(None, config))
    assert final["done"] is True
    # 内层完成态被 WAL 复用:重放没有再调用内层模型,也没有重复取证。
    assert history_shapes == [1, 3]
    assert len(evidence_runs) == 1
    seeded_sizes = [
        len((checkpoint_tuple.checkpoint or {}).get("channel_values", {}).get("seeded", []))
        for checkpoint_tuple in saver.list(config)
        if "seeded" in (checkpoint_tuple.checkpoint or {}).get("channel_values", {})
    ]
    assert seeded_sizes and max(seeded_sizes) == 1


# 三级嵌套:外层节点、supervisor create_agent、Delegate 工具、researcher create_agent。


def test_delegate_resumed_inner_researcher_keeps_evidence():
    """三级继承合同:生产嵌套形状下,researcher 从自己的 superstep 续跑,取证工具不重跑。

    outer→supervisor(None)→Delegate 工具→researcher。researcher 内层把 evidence 写进
    自己的通道、第二回合崩溃;重放外层后,若 researcher 续跑则取证只发生一次,若整体
    重跑则两次。
    """
    for researcher_checkpointer, expected_runs in ((None, 1), (False, 2)):
        app, counters = _build_triple(researcher_checkpointer)
        config = {"configurable": {"thread_id": f"triple-{researcher_checkpointer}"}}
        for attempt in range(2):
            try:
                asyncio.run(app.ainvoke({"done": False} if attempt == 0 else None, config))
            except BaseException:
                pass
        assert counters["evidence"] == expected_runs, (
            f"researcher checkpointer={researcher_checkpointer!r} 取证次数应为 {expected_runs}"
        )


def _build_triple(researcher_checkpointer):
    counters = {"evidence": 0}

    @tool("AddEvidence")
    async def add_evidence(runtime: ToolRuntime) -> Command:
        """最内层取证一次。"""
        counters["evidence"] += 1
        return Command(
            update={
                "evidence_count": 1,
                "messages": [
                    ToolMessage("e", tool_call_id=runtime.tool_call_id, name="AddEvidence")
                ],
            }
        )

    researcher_shapes: list[int] = []

    @tool("Delegate")
    async def delegate(runtime: ToolRuntime) -> Command:
        """派最内层 researcher,与生产 delegate 工具同形,仅传部分 config。"""
        researcher = create_agent(
            model=_CrashModel(researcher_shapes),
            tools=[add_evidence],
            state_schema=_CountedState,
            checkpointer=researcher_checkpointer,
            name="researcher",
        )
        await researcher.ainvoke(
            {"messages": [HumanMessage(content="research this direction")]},
            config={"recursion_limit": 50},
        )
        return Command(
            update={
                "messages": [
                    ToolMessage("delegated", tool_call_id=runtime.tool_call_id, name="Delegate")
                ]
            }
        )

    supervisor_shapes: list[int] = []

    class _SupModel:
        def bind_tools(self, _tools, **_kwargs):
            return self

        async def ainvoke(self, messages):
            supervisor_shapes.append(len(messages))
            if len(messages) == 1:
                return AIMessage(
                    content="",
                    tool_calls=[{"name": "Delegate", "args": {}, "id": "d-1"}],
                )
            return AIMessage(content="done")

    supervisor = create_agent(
        model=_SupModel(), tools=[delegate], checkpointer=None, name="supervisor"
    )

    class _Outer(TypedDict):
        done: bool

    async def outer_node(_state):
        await supervisor.ainvoke(
            {"messages": [HumanMessage(content="plan")]}, config={"recursion_limit": 50}
        )
        return {"done": True}

    outer = StateGraph(_Outer)
    outer.add_node("sup", outer_node)
    outer.add_edge(START, "sup")
    outer.add_edge("sup", END)
    return outer.compile(checkpointer=MemorySaver()), counters


# 生产校验:各循环构造时的 checkpointer 取值。


def test_production_loops_compile_with_inherited_checkpointer():
    """生产三个 create_agent 循环均已通道化,业务状态并入各自通道,checkpointer 使用继承(None)。

    supervisor/researcher/writer 把可变簿记写进各自 state_schema 通道,启用继承后,
    崩溃重放从各内层最后一个已提交 superstep 续跑,见上方继承合同用例。
    """
    supervisor = ResearchSupervisor(object(), AgentConfig(), research_agent=object())
    assert supervisor._agent_loop.checkpointer is None

    researcher = ResearchAgent(
        object(),
        AgentConfig(),
        search_tool=object(),
        reader_tool=object(),
    )
    assert researcher._agent_loop.checkpointer is None

    writer = ReportWriter(
        object(),
        AgentConfig(),
        render_incomplete=lambda _state: "x",
    )
    assert writer._agent_loop.checkpointer is None


def test_clarifier_keeps_checkpointer_inheritance_for_interrupt():
    """clarifier 的 interrupt 恢复链依赖继承(默认 None),禁用其他循环继承时不得一并禁用它。"""
    clarifier = Clarifier(object(), AgentConfig())
    assert clarifier.graph.checkpointer is None
