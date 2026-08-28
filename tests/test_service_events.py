import asyncio

import pytest
import pytest_asyncio

from deepsearch_agent.service.events import CLOSE_STREAM, CompositeSink, FanoutSink

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def sink():
    return FanoutSink(asyncio.get_running_loop())


async def _get_message(queue, timeout=1.0):
    return await asyncio.wait_for(queue.get(), timeout)


async def test_write_assigns_seq_and_delivers_to_subscribers(sink):
    sink.open("run-1")
    key_a, queue_a = sink.subscribe("run-1")
    _, queue_b = sink.subscribe("run-1")
    sink.write({"run_id": "run-1", "event_type": "node_started", "payload": {}})

    first = await _get_message(queue_a)
    assert first["seq"] == 1
    assert await _get_message(queue_b) is first  # 同一记录对象，同一 seq
    sink.unsubscribe("run-1", key_a)
    sink.write({"run_id": "run-1", "event_type": "node_completed", "payload": {}})
    assert queue_a.empty()  # 退订后不再投递
    assert queue_b.qsize() == 1
    assert (await _get_message(queue_b))["seq"] == 2


async def test_unrouted_and_unopened_records_are_dropped(sink):
    sink.write({"event_type": "x"})  # 无 run_id
    sink.write({"run_id": "never-opened", "event_type": "x"})
    assert sink.take_pending("never-opened") == []


async def test_take_pending_returns_all_in_order_and_drains(sink):
    sink.open("run-2")
    for i in range(3):
        sink.write({"run_id": "run-2", "event_type": f"e{i}", "payload": {}})
    pending = sink.take_pending("run-2")
    assert [item["seq"] for item in pending] == [1, 2, 3]
    assert [item["event_type"] for item in pending] == ["e0", "e1", "e2"]
    assert sink.take_pending("run-2") == []
    sink.write({"run_id": "run-2", "event_type": "e3", "payload": {}})
    assert [item["seq"] for item in sink.take_pending("run-2")] == [4]  # 排水后 seq 不回绕


async def test_overflow_drops_oldest_and_injects_truncation_marker():
    sink = FanoutSink(asyncio.get_running_loop(), queue_maxsize=2)
    sink.open("run-3")
    _, queue = sink.subscribe("run-3")
    for i in range(4):
        sink.write({"run_id": "run-3", "event_type": f"e{i}", "payload": {}})

    delivered = [await _get_message(queue), await _get_message(queue)]
    assert [item["event_type"] for item in delivered] == [
        "stream_truncated",
        "stream_truncated",
    ]
    assert [item["seq"] for item in delivered] == [1, 2]  # 标记沿用被丢者的 seq
    # 持久化侧不受溢出影响：4 条都在
    assert [item["seq"] for item in sink.take_pending("run-3")] == [1, 2, 3, 4]


async def test_close_sentinels_subscribers_and_late_writes_drop(sink):
    sink.open("run-4")
    _, queue = sink.subscribe("run-4")
    sink.close("run-4")
    assert await _get_message(queue) is CLOSE_STREAM
    sink.write({"run_id": "run-4", "event_type": "late", "payload": {}})
    assert sink.take_pending("run-4") == []  # 不报错、不排队、pending 内存在 close 时释放
    key, queue2 = sink.subscribe("run-4")
    assert key == -1
    assert await _get_message(queue2) is CLOSE_STREAM


async def test_write_from_worker_thread_reaches_queue(sink):
    sink.open("run-5")
    _, queue = sink.subscribe("run-5")
    await asyncio.to_thread(sink.write, {"run_id": "run-5", "event_type": "x", "payload": {}})
    assert (await _get_message(queue))["seq"] == 1


async def test_pydantic_model_records_are_dumped(sink):
    from pydantic import BaseModel

    class Rec(BaseModel):
        run_id: str
        event_type: str
        seq: int | None = None

    sink.open("run-6")
    sink.write(Rec(run_id="run-6", event_type="node_started"))
    pending = sink.take_pending("run-6")
    assert pending[0]["seq"] == 1
    assert "seq" in pending[0]


async def test_composite_sink_isolates_member_failures():
    class Boom:
        def write(self, _record):
            raise RuntimeError("sink down")

    class Collect:
        def __init__(self):
            self.records = []

        def write(self, record):
            self.records.append(record)

    collector = Collect()
    CompositeSink(Boom(), collector).write({"run_id": "r", "event_type": "x"})
    assert collector.records == [{"run_id": "r", "event_type": "x"}]
