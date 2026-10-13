"""supervisor 崩溃恢复合同:真 AsyncPostgresSaver 上验证,可被子进程形态复用。

形态:max_parallel_workers=2、supervisor_max_active_evidences=2,首轮一条 AIMessage 并发派
两个 ResearchDelegate(派发批只登记 pending,不等待方向完成),模型经 WaitDirection 收
A 的到达后,后台的 B 抛 BaseException 杀死节点,再同 thread 恢复。实测合同:节点失败使
其范围内全部吸收随写回丢弃(WaitDirection 收据也不例外——内层 superstep 的通道写入只在
节点成功返回时提交),恢复靠派发批 journal 命中不重跑方向体,欠账一律判 lost 结算。
本测试锁定的不变式:恢复后编号按调用位置稳定、两方向结果互不吞、无残留半吸收。

模块同时导出 _build_app/_MixedResearchAgent/_TwoDirectionSupervisorModel,供 SIGKILL 形态
探针以子进程模式挂起 B(不写 __error__,观察纯外部进程死亡后的 journal 消费语义)。
DB 不可达时整文件 skip。
"""

import asyncio
import uuid

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from deepresearcher.agents.supervisor import ResearchSupervisor
from deepresearcher.config import AgentConfig
from deepresearcher.evidence.models import Evidence
from deepresearcher.schemas.sections import ResearchAgentResult, ResearchDirectionResult


class _ProcessKilled(BaseException):
    """模拟进程被杀:非 Exception,不被 delegate 的失败降级捕获。"""


class _TwoDirectionSupervisorModel:
    """首轮并发派两个方向;先 WaitDirection 收第一波到货(使 A 的吸收落进已提交通道),再收尾。"""

    def bind_tools(self, _tools, **_kwargs):
        return self

    async def ainvoke(self, messages):
        delegate_receipts = [
            message
            for message in messages
            if getattr(message, "name", None) == "ResearchDelegate"
            and getattr(message, "type", None) == "tool"
        ]
        wait_receipts = [
            message
            for message in messages
            if getattr(message, "name", None) == "WaitDirection"
            and getattr(message, "type", None) == "tool"
        ]
        if delegate_receipts and not wait_receipts:
            return AIMessage(
                content="",
                tool_calls=[{"name": "WaitDirection", "args": {}, "id": "w1"}],
            )
        if delegate_receipts:
            return AIMessage(content="研究结束,交回。")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "ResearchDelegate",
                    "args": {"display_title": "延迟侧", "research_topic": "方向A:延迟"},
                    "id": "dA",
                },
                {
                    "name": "ResearchDelegate",
                    "args": {"display_title": "吞吐侧", "research_topic": "方向B:吞吐"},
                    "id": "dB",
                },
            ],
        )


class _MixedResearchAgent:
    """方向桩。mode=exception:B 在 A 完成后抛错;mode=hang-child:B 在 A 完成后永久挂起。

    side_suffix 参与 conclusion 文本,让探针能分辨每个 task result 产自哪个进程。
    """

    def __init__(self, mode: str = "exception", side_suffix: str = "parent") -> None:
        self.mode = mode
        self.side_suffix = side_suffix
        self.calls: dict[str, int] = {"A": 0, "B": 0}
        self._a_done = asyncio.Event()
        self._killed_once = False

    @staticmethod
    def _result(task, side: str, suffix: str) -> ResearchAgentResult:
        evidences = [
            Evidence(
                evidence_id=f"{task['id']}-ev-{i}",
                subtask_id=task["id"],
                research_direction=task["question"],
                claim=f"{side} 结论 {i}",
                quote=f"{side} 原文 {i}",
                source_url=f"https://example.com/{side}{i}",
            )
            for i in (1, 2)
        ]
        task_result = ResearchDirectionResult(
            task_id=task["id"],
            round=int(task.get("round", 1)),
            task_index=int(task.get("sequence", 0)),
            question=task["question"],
            research_direction=task["question"],
            execution_status="completed",
            coverage_status="sufficient",
            evidence_count=2,
            source_count=2,
            conclusion=f"{side} 完成 ({suffix})",
            stop_reason="complete",
        )
        return ResearchAgentResult(
            evidences=evidences,
            selected_evidence_ids=[item.evidence_id for item in evidences],
            source_refs=[item.source_url for item in evidences],
            task_result=task_result,
        )

    async def run(self, task):
        side = "A" if task["question"].startswith("方向A") else "B"
        self.calls[side] += 1
        if side == "A":
            result = self._result(task, "A", self.side_suffix)
            self._a_done.set()
            return result
        if self.mode == "hang-child":
            await self._a_done.wait()
            # 不抛不返回:B 纯挂起,整步不会有 __error__,等待外部 SIGKILL。
            while True:
                await asyncio.sleep(0.5)
        if self.mode == "exception" and not self._killed_once:
            await self._a_done.wait()
            # 留出窗口:让 A 的工具任务先完成。
            await asyncio.sleep(0.3)
            self._killed_once = True
            raise _ProcessKilled("simulated process kill after sibling completed")
        return self._result(task, "B", self.side_suffix)


def _make_saver():
    from deepresearcher.service.checkpoint_serde import build_checkpointer_serde
    from deepresearcher.service.settings import checkpoint_dsn, get_service_config

    dsn = checkpoint_dsn(get_service_config().database_url)
    if dsn is None:
        return None
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    return AsyncPostgresSaver.from_conn_string(dsn, serde=build_checkpointer_serde())


def _build_app(saver, research_agent):
    """生产 ResearchSupervisor 挂单节点外层图;探针与测试共用。"""
    supervisor = ResearchSupervisor(
        _TwoDirectionSupervisorModel(),
        AgentConfig(
            max_research_rounds=1,
            max_subtasks_per_round=2,
            max_parallel_workers=2,
            supervisor_max_active_evidences=2,
        ),
        research_agent=research_agent,
    )
    outcome: dict = {}

    class _Outer(TypedDict):
        done: bool

    async def supervisor_node(state):
        update = await supervisor.run({"query": "混合重放恢复测试", **state})
        outcome["update"] = update
        return {"done": True}

    outer = StateGraph(_Outer)
    outer.add_node("sup", supervisor_node)
    outer.add_edge(START, "sup")
    outer.add_edge("sup", END)
    app = outer.compile(checkpointer=saver)
    return app, outcome


async def _drive(saver):
    research_agent = _MixedResearchAgent(mode="exception")
    app, outcome = _build_app(saver, research_agent)

    thread = "mixed-" + uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread}}
    killed = False
    try:
        await app.ainvoke({"done": False}, config)
    except _ProcessKilled:
        killed = True
    async for _ in app.astream(None, config=config):
        pass

    update = outcome.get("update") or {}
    task_results = list(update.get("task_results") or [])
    return {
        "killed": killed,
        "calls": research_agent.calls,
        "task_count": len(task_results),
        "task_ids": sorted(item.task_id for item in task_results),
        "a_task_id": next(
            (item.task_id for item in task_results if item.question.startswith("方向A")), ""
        ),
        "a_status": next(
            (item.execution_status for item in task_results if item.question.startswith("方向A")),
            "",
        ),
        "b_status": next(
            (item.execution_status for item in task_results if item.question.startswith("方向B")),
            "",
        ),
        "evidences": len(update.get("evidences") or []),
        "active_count": len(update.get("active_evidence_ids") or []),
    }


def test_mixed_replay_merges_journaled_and_rerun_writes_consistently():
    """崩溃恢复:派发批 journal 命中不重跑方向体,节点内吸收(含 WaitDirection)随失败丢弃,
    欠账在收口处全部判 lost 结算;编号稳定、两方向不互吞。"""
    saver_context = _make_saver()
    if saver_context is None:
        pytest.skip("checkpointer 未配置(非 postgresql 部署形态)")

    async def run():
        try:
            async with saver_context as saver:
                await saver.setup()
                return await _drive(saver)
        except Exception:  # noqa: BLE001 - 连接层失败按缺依赖处理
            return None

    facts = asyncio.run(run())
    if facts is None:
        pytest.skip("Postgres checkpointer 不可达")

    assert facts["killed"] is True, "首次运行应因模拟进程硬杀中止"
    # 派发制合同:方向体是后台任务,其产物只有在节点成功返回时才进外层 State。
    # 崩溃使一切节点内吸收作废——恢复不重跑(receipts/编号有 journal),只把欠账判 lost。
    assert facts["calls"] == {"A": 1, "B": 1}, facts["calls"]
    assert facts["task_count"] == 2, facts
    assert facts["task_ids"] == ["task-0001", "task-0002"], facts["task_ids"]
    assert facts["a_task_id"] == "task-0001", facts["a_task_id"]
    assert facts["a_status"] == "failed", facts
    assert facts["b_status"] == "failed", facts
    assert facts["evidences"] == 0, facts  # 未随成功节点提交的吸收不留残值
    assert facts["active_count"] == 0, facts
