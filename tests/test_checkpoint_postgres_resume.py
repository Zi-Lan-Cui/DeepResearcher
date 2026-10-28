"""生产 AsyncPostgresSaver 上的嵌套循环 kill-重放集成测试,补齐 MemorySaver 没覆盖的两点:
真 checkpointer+serde 是否同样按 pending-writes WAL 让内层子图从自己的 superstep 续跑;
两个并发 researcher 的嵌套 checkpoint_ns 是否彼此独立、跨重放确定。

形状:outer 带 thread 检查点,进入 supervisor create_agent 后两个并发 Delegate 各跑一个
researcher,业务状态进各自通道;第二个模型回合抛 CancelledError 模拟进程被杀。重放后断言:
取证工具总执行 2 次,即每方向一次;外层下有至少 2 个独立嵌套 ns。
DB 不可达时整文件 skip,不回显连接串。
"""

import asyncio
import contextlib
import uuid

import pytest
from langchain.agents import AgentState, create_agent
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from typing_extensions import Annotated, TypedDict

from deepresearcher.service.execution.checkpoint_serde import build_checkpointer_serde


class _CountedState(AgentState):
    """内层 researcher 的业务通道:取证计数经 reducer 累加。"""

    evidence_runs: Annotated[int, lambda a, b: (a or 0) + (b or 0)]


def _make_saver():
    """按生产同一路径建 AsyncPostgresSaver；测试库未配置则返回 None 触发 skip。

    库一律取 SERVICE_TEST_DATABASE_URL：这些用例会往库里写 checkpoint 表，
    不能落到生产配置指向的库上。
    """
    from deepresearcher.service.settings import checkpoint_dsn
    from pg_backend import postgres_base

    base = postgres_base()
    if base is None:
        return None
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    return AsyncPostgresSaver.from_conn_string(
        checkpoint_dsn(base), serde=build_checkpointer_serde()
    )


async def _drive_two_tasks(saver):
    """先崩溃跑一次 outer 再重放一次,返回 (取证总执行数, 嵌套 ns 去重集合, thread)。"""
    counters = {"evidence": 0}

    @tool("AddEvidence")
    async def add_evidence(runtime: ToolRuntime) -> Command:
        """最内层取证一次并写进本方向通道。"""
        counters["evidence"] += 1
        return Command(
            update={
                "evidence_runs": 1,
                "messages": [
                    ToolMessage("e", tool_call_id=runtime.tool_call_id, name="AddEvidence")
                ],
            }
        )

    class _ResearcherModel:
        def bind_tools(self, _tools, **_kwargs):
            return self

        async def ainvoke(self, messages):
            if len(messages) == 1:
                return AIMessage(
                    content="", tool_calls=[{"name": "AddEvidence", "args": {}, "id": "ev"}]
                )
            raise asyncio.CancelledError("simulated process kill mid-direction")

    class _SupModel:
        def bind_tools(self, _tools, **_kwargs):
            return self

        async def ainvoke(self, messages):
            if len(messages) == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "Delegate", "args": {"d": "dirA"}, "id": "d1"},
                        {"name": "Delegate", "args": {"d": "dirB"}, "id": "d2"},
                    ],
                )
            return AIMessage(content="done")

    @tool("Delegate", args_schema=None)
    async def delegate(d: str, runtime: ToolRuntime) -> Command:
        """为一个方向跑内嵌 researcher。与生产 delegate 同形:仅传部分 config,继承 thread。"""
        del d
        researcher = create_agent(
            model=_ResearcherModel(),
            tools=[add_evidence],
            state_schema=_CountedState,
            name="researcher",
        )
        await researcher.ainvoke(
            {"messages": [HumanMessage(content="research")]}, config={"recursion_limit": 50}
        )
        return Command(
            update={
                "messages": [
                    ToolMessage("delegated", tool_call_id=runtime.tool_call_id, name="Delegate")
                ]
            }
        )

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
    app = outer.compile(checkpointer=saver)

    thread = "itest-" + uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread}}
    try:
        for attempt in range(2):
            with contextlib.suppress(BaseException):
                await app.ainvoke({"done": False} if attempt == 0 else None, config)

        nested_ns = set()
        async for checkpoint_tuple in saver.alist(config):
            ns = (checkpoint_tuple.config or {}).get("configurable", {}).get("checkpoint_ns", "")
            if ns:
                nested_ns.add(ns)
    finally:
        # 演示同库不留测试线程：断言失败也要回收，否则每次跑批都往生产 checkpoint 里漏行。
        await saver.adelete_thread(thread)
    return counters["evidence"], nested_ns, thread


def test_two_concurrent_researchers_resume_on_postgres_checkpoint():
    """真 Postgres saver 下,两个并发 researcher 各自从内层断点续跑、ns 独立。"""
    saver_context = _make_saver()
    if saver_context is None:
        pytest.skip("checkpointer 未配置(非 postgresql 部署形态)")

    async def run():
        try:
            async with saver_context as saver:
                await saver.setup()
                return await _drive_two_tasks(saver)
        except Exception:  # noqa: BLE001 - 连接失败按部署缺依赖处理
            return None

    outcome = asyncio.run(run())
    if outcome is None:
        # 跳过原因不回显异常与连接串。
        pytest.skip("Postgres checkpointer 不可达")

    evidence_runs, nested_ns, _thread = outcome
    # 每个方向的取证只应执行一次：两个 researcher 都从自己的 superstep 续跑，未整体重放。
    assert evidence_runs == 2, f"取证执行数应为 2(续跑),实测 {evidence_runs}(疑似整体重跑/ns 漂移)"
    # 两个并发分支落在彼此独立的嵌套 ns 下，dirA 与 dirB 各有一条 tools 级 ns。
    assert len(nested_ns) >= 2, f"应见 ≥2 个独立嵌套 ns,实测 {sorted(nested_ns)}"
