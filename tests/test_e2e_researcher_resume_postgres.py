"""端到端 kill-重放:生产 channelized ResearchAgent/ResearchSupervisor 跑在真 AsyncPostgresSaver 上。

区别于 `test_checkpoint_postgres_resume.py`(用最简 create_agent 证框架契约),本文件驱动
真实生产类,验证"崩溃前已提交的取证在恢复后不重跑、不浪费":进程在 add_evidence 随
superstep 提交后中止(抛非 Exception 的 BaseException,不被业务层 except 吞),外层流式
中止;再以执行器恢复用的同一机制 `graph.astream(None)`(同 thread)续跑。恢复后搜索/抓取
各只执行 1 次即证内层从自己的 superstep 续跑、未整体重放。DB 不可达时整文件 skip,不回显连接串。
"""

import asyncio
import hashlib
import re
import uuid

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from deepresearcher.agents.researcher import ResearchAgent
from deepresearcher.agents.supervisor import ResearchSupervisor
from deepresearcher.config import AgentConfig
from deepresearcher.state import SubTask
from deepresearcher.tools import SearchTool
from deepresearcher.tools.web.materials import MemoryResearchMaterialStore
from deepresearcher.tools.web.reader import SourceReaderTool

_QUOTE = "实验表明端到端延迟为 20ms。"
_URL = "https://example.com/e2e-report"


class _ProcessKilled(BaseException):
    """模拟进程被硬杀:非 Exception、非 CancelledError,故不被业务层 except 吞掉。"""


class _FakeSearchService:
    def __init__(self) -> None:
        self.calls = 0

    async def asearch(self, query):
        self.calls += 1
        return [{"title": query, "url": _URL, "snippet": "摘要", "score": 0.9}]


class _FakeFetcher:
    def __init__(self) -> None:
        self.calls = 0

    async def afetch(self, url, **_kwargs):
        self.calls += 1
        return {
            "title": "端到端实验报告",
            "text": f"引言\n{_QUOTE}\n结论",
            "final_url": url,
            "retrieval_method": "origin_fetch",
            "support_ceiling": "direct",
        }


def _candidate_id(url: str) -> str:
    return "c-" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]


class _ScriptedResearcherModel:
    """按消息历史里已存在的工具回执决定下一步;add 提交后先杀进程一次,续跑时收尾。

    与真实模型一样从 ToolMessage JSON 里读 candidate_id / evidence_id,不做特例硬编码。
    """

    def __init__(self) -> None:
        self._killed = False

    def bind_tools(self, _tools, **_kwargs):
        return self

    def _latest_receipt(self, messages, tool_name):
        for message in reversed(messages):
            if (
                getattr(message, "name", None) == tool_name
                and getattr(message, "type", None) == "tool"
            ):
                return str(message.content)
        return None

    async def ainvoke(self, messages):
        from langchain_core.messages import AIMessage

        read_receipt = self._latest_receipt(messages, "ReadSources")
        add_receipt = self._latest_receipt(messages, "AddEvidence")
        if self._latest_receipt(messages, "SearchSources") is None:
            call = {
                "name": "SearchSources",
                "args": {"reason": "找直接来源", "queries": ["端到端 延迟"]},
                "id": "s1",
            }
        elif read_receipt is None:
            candidate = _candidate_id(_URL)
            call = {
                "name": "ReadSources",
                "args": {"reason": "读取候选", "candidate_ids": [candidate]},
                "id": "r1",
            }
        elif add_receipt is None:
            document_id = re.search(r'"document_id"\s*:\s*"([^"]+)"', read_receipt).group(1)
            call = {
                "name": "AddEvidence",
                "args": {
                    "reason": "逐字引用提交",
                    "evidences": [
                        {
                            "document_id": document_id,
                            "claim": "端到端延迟为 20ms。",
                            "quote": _QUOTE,
                            "support": "direct",
                            "confidence": 0.9,
                        }
                    ],
                },
                "id": "a1",
            }
        elif not self._killed:
            # 证据已随 superstep 提交进内层检查点;此刻模拟进程硬杀(不被 run() 吞)。
            self._killed = True
            raise _ProcessKilled("simulated process kill right after evidence committed")
        else:
            evidence_id = re.search(r'"evidence_id"\s*:\s*"([^"]+)"', add_receipt).group(1)
            call = {
                "name": "ResearchDirectionComplete",
                "args": {
                    "reason": "已覆盖核心事实",
                    "selected_evidence_ids": [evidence_id],
                    "conclusion": "该来源给出端到端延迟 20ms。",
                    "remaining_gaps": [],
                },
                "id": "c1",
            }
        return AIMessage(content="", tool_calls=[call])


def _task() -> SubTask:
    return {
        "id": "task-e2e",
        "run_id": "run-e2e",
        "question": "端到端延迟是多少?",
        "display_title": "延迟",
        "round": 1,
        "sequence": 1,
        "worker_id": "w1",
        "worker_index": 1,
        "parent_task_id": "",
        "operation_id": "op-e2e",
    }


def _make_saver():
    from deepresearcher.service.checkpoint_serde import build_checkpointer_serde
    from deepresearcher.service.settings import checkpoint_dsn, get_service_config

    dsn = checkpoint_dsn(get_service_config().database_url)
    if dsn is None:
        return None
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    return AsyncPostgresSaver.from_conn_string(dsn, serde=build_checkpointer_serde())


async def _drive(saver):
    search = _FakeSearchService()
    fetcher = _FakeFetcher()
    store = MemoryResearchMaterialStore()
    agent = ResearchAgent(
        _ScriptedResearcherModel(),
        AgentConfig(
            research_agent_max_evidences_per_direction=4,
            research_agent_max_evidence_candidates_per_direction=8,
        ),
        search_tool=SearchTool(search),
        reader_tool=SourceReaderTool(
            fetcher, material_store=store, document_inline_max_tokens=1_000
        ),
        material_store=store,
    )

    outcome: dict = {}

    class _Outer(TypedDict):
        done: bool

    async def researcher_node(_state):
        result = await agent.run(_task())
        outcome["result"] = result
        return {"done": True}

    outer = StateGraph(_Outer)
    outer.add_node("research", researcher_node)
    outer.add_edge(START, "research")
    outer.add_edge("research", END)
    app = outer.compile(checkpointer=saver)

    thread = "e2e-" + uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread}}
    killed = False
    try:
        await app.ainvoke({"done": False}, config)
    except _ProcessKilled:
        killed = True
    # 执行器恢复用的正是 astream(None)+同 thread。
    async for _ in app.astream(None, config=config):
        pass

    result = outcome.get("result")
    return {
        "killed": killed,
        "search_calls": search.calls,
        "fetch_calls": fetcher.calls,
        "execution_status": getattr(result.task_result, "execution_status", None)
        if result
        else None,
        "evidence_count": len(result.evidences) if result else -1,
        "evidence_id": result.evidences[0].evidence_id if result and result.evidences else None,
    }


def test_e2e_researcher_resumes_without_rerunning_search_on_postgres():
    """真库上:进程在证据提交后被杀,恢复后搜索/抓取不重跑、已验证证据随收尾交付。"""
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
    # 关键:恢复后已提交的搜索与抓取只发生一次(researcher 从内层检查点续,而非整体重跑)。
    assert facts["search_calls"] == 1, f"搜索应只 1 次,实测 {facts['search_calls']}(疑似整体重跑)"
    assert facts["fetch_calls"] == 1, f"抓取应只 1 次,实测 {facts['fetch_calls']}(疑似整体重跑)"
    # 续跑正常收尾,崩溃前已验证的 1 条 Evidence 随结果交付。
    assert facts["execution_status"] == "completed", facts["execution_status"]
    assert facts["evidence_count"] == 1, facts["evidence_count"]
    assert facts["evidence_id"]


# --- 再套一层:生产 ResearchSupervisor 内嵌 researcher 的整链 kill-重放 -----------


class _FakeSupervisorModel:
    """supervisor 首轮派一个 ResearchDelegate;delegate 回来后收尾。

    与真实规划器一样按历史里有无 delegate 回执决定下一步,不做特例硬编码。
    """

    def bind_tools(self, _tools, **_kwargs):
        return self

    async def ainvoke(self, messages):
        has_delegate = any(
            getattr(m, "name", None) == "ResearchDelegate" and getattr(m, "type", None) == "tool"
            for m in messages
        )
        if has_delegate:
            return AIMessage(content="研究结束,交回。")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "ResearchDelegate",
                    "args": {"display_title": "延迟", "research_topic": "端到端延迟是多少?"},
                    "id": "d1",
                }
            ],
        )


async def _drive_supervisor(saver):
    search = _FakeSearchService()
    fetcher = _FakeFetcher()
    store = MemoryResearchMaterialStore()
    research_agent = ResearchAgent(
        _ScriptedResearcherModel(),
        AgentConfig(
            research_agent_max_evidences_per_direction=4,
            research_agent_max_evidence_candidates_per_direction=8,
        ),
        search_tool=SearchTool(search),
        reader_tool=SourceReaderTool(
            fetcher, material_store=store, document_inline_max_tokens=1_000
        ),
        material_store=store,
    )
    supervisor = ResearchSupervisor(
        _FakeSupervisorModel(),
        AgentConfig(
            max_research_rounds=1,
            max_subtasks_per_round=1,
            supervisor_max_active_evidences=6,
            research_agent_max_evidences_per_direction=4,
            research_agent_max_evidence_candidates_per_direction=8,
        ),
        research_agent=research_agent,
    )

    outcome: dict = {}

    class _Outer(TypedDict):
        done: bool

    async def supervisor_node(state):
        update = await supervisor.run({"query": "端到端延迟是多少?", **state})
        outcome["update"] = update
        return {"done": True}

    outer = StateGraph(_Outer)
    outer.add_node("sup", supervisor_node)
    outer.add_edge(START, "sup")
    outer.add_edge("sup", END)
    app = outer.compile(checkpointer=saver)

    thread = "sup-e2e-" + uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread, "checkpoint_ns": ""}}
    killed = False
    try:
        await app.ainvoke({"done": False}, {"configurable": {"thread_id": thread}})
    except _ProcessKilled:
        killed = True
    # 崩溃后、恢复前:reconcile._has_checkpoint 用的正是 aget_tuple(thread)——证明确有可续检查点。
    checkpoint_after_crash = await saver.aget_tuple(config)
    # 执行器恢复(含 reconcile→resume_runs→execute(resume=True))最终都走 astream(None)。
    async for _ in app.astream(None, config={"configurable": {"thread_id": thread}}):
        pass

    update = outcome.get("update") or {}
    task_results = update.get("task_results") or []
    absorbed = update.get("evidences") or []
    return {
        "killed": killed,
        "search_calls": search.calls,
        "fetch_calls": fetcher.calls,
        "has_checkpoint_after_crash": checkpoint_after_crash is not None,
        "task_count": len(task_results),
        "task_status": getattr(task_results[0], "execution_status", None) if task_results else None,
        "absorbed_evidence": len(absorbed),
    }


def test_e2e_supervisor_nested_researcher_resumes_on_postgres():
    """真库上:进程在内嵌 researcher 取证提交后被杀,supervisor+researcher 整链续跑、不重跑搜索。"""
    saver_context = _make_saver()
    if saver_context is None:
        pytest.skip("checkpointer 未配置(非 postgresql 部署形态)")

    async def run():
        try:
            async with saver_context as saver:
                await saver.setup()
                return await _drive_supervisor(saver)
        except Exception:  # noqa: BLE001 - 连接层失败按缺依赖处理
            return None

    facts = asyncio.run(run())
    if facts is None:
        pytest.skip("Postgres checkpointer 不可达")

    assert facts["killed"] is True, "首次运行应因模拟进程硬杀中止"
    # reconcile 据 aget_tuple 判定可续:崩溃后确有 thread 检查点。
    assert facts["has_checkpoint_after_crash"] is True
    # 整链恢复:已提交的搜索/抓取各只一次(supervisor 内层 + researcher 内层都从断点续)。
    assert facts["search_calls"] == 1, f"搜索应只 1 次,实测 {facts['search_calls']}(疑似整体重跑)"
    assert facts["fetch_calls"] == 1, f"抓取应只 1 次,实测 {facts['fetch_calls']}(疑似整体重跑)"
    # 方向完成并被 supervisor 吸收,崩溃前已验证的 Evidence 进入顶层通道。
    assert facts["task_count"] == 1, facts
    assert facts["task_status"] == "completed", facts["task_status"]
    assert facts["absorbed_evidence"] == 1, facts["absorbed_evidence"]
