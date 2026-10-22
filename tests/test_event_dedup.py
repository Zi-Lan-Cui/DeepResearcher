"""事件去重的两侧契约：键的生成规则，和写入时同键只留一行。

重放产生的是新 event_id、新 seq 的同一条逻辑内容，只有 dedup_key 不变，
所以它是唯一可用于唯一约束的字段。
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy import func, select

from deepresearcher.observability.events.dedup import bind_position, build_dedup_key
from deepresearcher.observability.events.models import (
    make_audit_event,
    make_node_event,
    make_tool_event,
)
from deepresearcher.service.events.store import RunEventStore
from deepresearcher.service.persistence.database import init_db, make_engine, make_session_factory
from deepresearcher.service.persistence.models import Run, RunEvent, User

RUN_ID = "run-dedup"


@pytest_asyncio.fixture
async def store(tmp_path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'dedup.db'}")
    await init_db(engine)
    factory = make_session_factory(engine)
    async with factory() as session:
        session.add(User(id=1, email="d@test.dev", password_hash="h"))
        session.add(Run(id=RUN_ID, user_id=1, query="去重", status="running"))
        await session.commit()
    yield RunEventStore(factory)
    await engine.dispose()


def test_key_requires_position_or_anchor():
    """两样都拿不到就不给键：宁可不参与去重，也不能把不同事件判成一条。"""
    assert build_dedup_key("node_started", {}) is None
    assert build_dedup_key("node_started", None) is None


def test_key_falls_back_to_anchor_without_position():
    """没有位置时退到对象身份；重放时 payload 相同，键也相同。"""
    payload = {"task_id": "task-0004", "operation_id": "op-1"}
    assert (
        build_dedup_key("research_task_started", payload) == "research_task_started|task-0004/op-1"
    )


def test_key_is_stable_across_replay_of_same_position():
    """同一次节点执行重跑，键必须一字不差——这是整条去重链成立的前提。"""
    payload = {"task_id": "task-0004"}
    with bind_position("supervisor:abc|model:def"):
        first = build_dedup_key("supervisor_model_turn", payload)
    with bind_position("supervisor:abc|model:def"):
        replay = build_dedup_key("supervisor_model_turn", payload)
    assert first == replay == "supervisor:abc|model:def|supervisor_model_turn|task-0004"


def test_key_distinguishes_positions_and_tool_calls():
    """ns 区分回合，anchor 区分同一步内的不同工具调用。"""
    payload = {"task_id": "t1", "tool_call_id": "call_a"}
    with bind_position("ns|tools:one"):
        first = build_dedup_key("tool_started", payload)
    with bind_position("ns|tools:two"):
        assert build_dedup_key("tool_started", payload) != first
    with bind_position("ns|tools:one"):
        others = build_dedup_key("tool_started", {**payload, "tool_call_id": "call_b"})
    assert others != first


def test_key_is_hashed_when_too_long():
    """超长键改用摘要，避免写库时被列宽截断而把两条不同的键截成同一个。"""
    payload = {"task_id": "t" * 400}
    key = build_dedup_key("some_event", payload)
    assert key is not None and key.startswith("h:") and len(key) < 100


def test_audit_event_carries_key_only_when_identity_known():
    """事件工厂把键写进载荷；拿不到身份时字段缺席，而不是写一个 None。"""
    bare = make_audit_event("some_event", run_id=RUN_ID)
    assert "dedup_key" not in bare
    keyed = make_audit_event("some_event", run_id=RUN_ID, payload={"task_id": "task-0001"})
    assert keyed["dedup_key"] == "some_event|task-0001"


def test_node_event_uses_bound_namespace():
    with bind_position("root:s1"):
        event = make_node_event("supervisor", "started", run_id=RUN_ID)
    assert event.dedup_key == "root:s1|node_started|"


def test_tool_event_keeps_its_own_prefix():
    """工具事件名不传时应落到 tool_ 前缀；键也必须按 tool_completed 算，不能串成 node_。"""
    event = make_tool_event("search", "completed", payload={"task_id": "t1", "tool_call_id": "c1"})
    assert event.event_type == "tool_completed"
    assert event.dedup_key == "tool_completed|t1/c1"


def test_project_attaches_dedup_key():
    """键必须进到 SSE 帧里，否则前端没有可用于去重的字段。"""
    from deepresearcher.service.events.projector import project

    frame = project(
        {
            "event_type": "node_started",
            "node": "writer",
            "seq": 3,
            "dedup_key": "ns|node_started|",
        }
    )
    assert frame is not None
    assert frame.data["dedup_key"] == "ns|node_started|"


def test_project_omits_missing_dedup_key():
    """没有键的帧不带该字段，前端据此跳过这一层判断。"""
    from deepresearcher.service.events.projector import project

    frame = project({"event_type": "node_started", "node": "writer", "seq": 3})
    assert frame is not None
    assert "dedup_key" not in frame.data


async def _count(store: RunEventStore) -> int:
    async with store._session_factory() as session:
        return int((await session.scalar(select(func.count()).select_from(RunEvent))) or 0)


@pytest.mark.asyncio
async def test_append_drops_replayed_record(store):
    """同一条逻辑事件写两次，库里只留一行；第二次返回空、不发多余记录。"""
    record = {"event_type": "research_task_started", "dedup_key": "k1", "payload": {}}
    first = await store.append(RUN_ID, [record])
    second = await store.append(RUN_ID, [dict(record, event_id="evt-other")])
    assert len(first) == 1 and second == []
    assert await _count(store) == 1


@pytest.mark.asyncio
async def test_append_keeps_records_without_key(store):
    """没有键的记录一律写库——这类事件本来就不该被判定为重复。"""
    await store.append(RUN_ID, [{"event_type": "run_status"}])
    await store.append(RUN_ID, [{"event_type": "run_status"}])
    assert await _count(store) == 2


@pytest.mark.asyncio
async def test_append_dedupes_within_batch(store):
    """同一批里同键只留首条；seq 按剩余条数分配，不留空洞。"""
    records = [
        {"event_type": "a", "dedup_key": "same"},
        {"event_type": "a", "dedup_key": "same"},
        {"event_type": "b", "dedup_key": "other"},
    ]
    assigned = await store.append(RUN_ID, records)
    assert [item["dedup_key"] for item in assigned] == ["same", "other"]
    assert [item["seq"] for item in assigned] == [1, 2]


@pytest.mark.asyncio
async def test_replay_after_partial_write_does_not_shift_seq(store):
    """重放补写时只对新增的那几条分配 seq，序号与已写入部分连续。"""
    await store.append(RUN_ID, [{"event_type": "a", "dedup_key": "k1"}])
    assigned = await store.append(
        RUN_ID,
        [
            {"event_type": "a", "dedup_key": "k1"},
            {"event_type": "b", "dedup_key": "k2"},
        ],
    )
    assert [item["seq"] for item in assigned] == [2]


@pytest.mark.asyncio
async def test_guarded_node_binds_position_in_real_graph():
    """守卫节点必须真的从 LangGraph 拿到 config。

    节点签名若丢掉 config 参数，位置绑不上，事件键退回对象身份甚至变成 None，
    去重静默失效且不报错。
    """
    from typing import Any, TypedDict

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph

    from deepresearcher.graph import _guarded_node

    class Sink:
        def __init__(self) -> None:
            self.records: list[dict[str, Any]] = []

        def write(self, record: Any) -> None:
            if hasattr(record, "model_dump"):
                record = record.model_dump(exclude_none=True)
            self.records.append(dict(record))

    class State(TypedDict, total=False):
        v: int

    def node(state: State) -> dict:
        return {"v": 1}

    sink = Sink()
    graph = StateGraph(State)
    graph.add_node("only", _guarded_node("only", node, event_sink=sink, max_text_chars=80))
    graph.add_edge(START, "only")
    graph.add_edge("only", END)
    await graph.compile(checkpointer=MemorySaver()).ainvoke(
        {"v": 0}, config={"configurable": {"thread_id": "dedup-t"}}
    )

    types = [record.get("event_type") for record in sink.records]
    keys = [str(record.get("dedup_key")) for record in sink.records]
    assert types == ["node_started", "node_completed"]
    assert keys[0].startswith("only:") and keys[0].endswith("|node_started|")
    assert keys[1].startswith("only:") and keys[1].endswith("|node_completed|")
    # 同一个节点空间：两条键除事件类型外完全一致。
    assert keys[0].rsplit("|", 2)[0] == keys[1].rsplit("|", 2)[0]
