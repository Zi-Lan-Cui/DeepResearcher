import asyncio

import pytest
import pytest_asyncio

from deepresearcher.service.events.stream import CLOSE_STREAM, CompositeSink, FanoutSink

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def sink():
    return FanoutSink(asyncio.get_running_loop())


async def _get_message(queue, timeout=1.0):
    return await asyncio.wait_for(queue.get(), timeout)


async def test_only_persisted_records_are_delivered_to_subscribers(sink):
    sink.open("run-1")
    key_a, queue_a = sink.subscribe("run-1")
    _, queue_b = sink.subscribe("run-1")
    sink.write({"run_id": "run-1", "event_type": "node_started", "payload": {}})
    assert queue_a.empty()
    first_batch = sink.take_pending("run-1")
    first_batch[0]["seq"] = 1
    sink.publish_persisted("run-1", first_batch)
    first = await _get_message(queue_a)
    assert first["seq"] == 1
    assert await _get_message(queue_b) is first  # 同一记录对象，同一 seq
    sink.unsubscribe("run-1", key_a)
    sink.write({"run_id": "run-1", "event_type": "node_completed", "payload": {}})
    second_batch = sink.take_pending("run-1")
    second_batch[0]["seq"] = 2
    sink.publish_persisted("run-1", second_batch)
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
    assert all("seq" not in item for item in pending)
    assert [item["event_type"] for item in pending] == ["e0", "e1", "e2"]
    assert sink.take_pending("run-2") == []
    sink.write({"run_id": "run-2", "event_type": "e3", "payload": {}})
    assert "seq" not in sink.take_pending("run-2")[0]


async def test_late_subscriber_is_woken_after_pending_batch_commits(sink):
    sink.open("run-7")
    sink.write({"run_id": "run-7", "event_type": "before", "payload": {}})  # 无订阅者时写入
    _, queue = sink.subscribe("run-7")
    sink.write({"run_id": "run-7", "event_type": "after", "payload": {}})
    pending = sink.take_pending("run-7")
    for seq, item in enumerate(pending, 1):
        item["seq"] = seq
    sink.publish_persisted("run-7", pending)
    received = [await _get_message(queue), await _get_message(queue)]
    assert [item["event_type"] for item in received] == ["before", "after"]
    assert [item["seq"] for item in received] == [1, 2]
    assert sink.take_pending("run-7") == []


async def test_overflow_drops_oldest_and_injects_truncation_marker():
    sink = FanoutSink(asyncio.get_running_loop(), queue_maxsize=2)
    sink.open("run-3")
    _, queue = sink.subscribe("run-3")
    for i in range(4):
        sink.write({"run_id": "run-3", "event_type": f"e{i}", "payload": {}})
    persisted = sink.take_pending("run-3")
    for seq, item in enumerate(persisted, 1):
        item["seq"] = seq
    sink.publish_persisted("run-3", persisted)

    delivered = [await _get_message(queue), await _get_message(queue)]
    assert [item["event_type"] for item in delivered] == [
        "stream_truncated",
        "stream_truncated",
    ]
    assert [item["seq"] for item in delivered] == [1, 2]  # 标记沿用被丢者的 seq
    # 持久化侧不受溢出影响：4 条都在
    assert [item["seq"] for item in persisted] == [1, 2, 3, 4]


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


async def test_open_set_bounded_evicts_oldest_without_termination_sentinel():
    """回收 ≠ 终结:最旧条目被逐出时不发 CLOSE_STREAM,在途订阅者只是降级为 DB tail。"""
    sink = FanoutSink(asyncio.get_running_loop(), open_max=2)
    sink.open("run-old")
    _, old_queue = sink.subscribe("run-old")
    sink.open("run-mid")
    assert sink.is_open("run-old") and sink.is_open("run-mid")

    sink.open("run-new")  # 超限:run-old 按插入序出局
    assert not sink.is_open("run-old")
    assert sink.is_open("run-mid") and sink.is_open("run-new")
    assert old_queue.empty()  # 无哨兵——订阅者不被告知 run 已死
    sink.write({"run_id": "run-old", "event_type": "late", "payload": {}})  # 静默丢弃不炸
    assert sink.take_pending("run-old") == []


async def test_write_from_worker_thread_reaches_queue(sink):
    sink.open("run-5")
    _, queue = sink.subscribe("run-5")
    await asyncio.to_thread(sink.write, {"run_id": "run-5", "event_type": "x", "payload": {}})
    pending = sink.take_pending("run-5")
    pending[0]["seq"] = 1
    sink.publish_persisted("run-5", pending)
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
    assert "seq" not in pending[0]


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


async def test_publisher_flush_requeues_batch_when_store_fails():
    """排水失败必须回插队首:done 帧整批丢失会让全部 SSE 靠兜底才收敛。"""

    class FlakyStore:
        def __init__(self):
            self.batches = []
            self.fail_first = True

        async def append(self, run_id, records):
            if self.fail_first:
                self.fail_first = False
                raise ConnectionError("db blip")
            self.batches.append(list(records))

        async def after(self, run_id, seq):
            return []

    sink = FanoutSink(asyncio.get_running_loop())
    sink.open("r1")
    store = FlakyStore()
    from deepresearcher.service.events.publisher import RunEventPublisher

    publisher = RunEventPublisher(
        session_factory=lambda: None,
        fanout=sink,
        event_store=store,
    )
    sink.write({"run_id": "r1", "event_type": "engine_0", "payload": {}})
    sink.write({"run_id": "r1", "event_type": "run_done", "payload": {"status": "completed"}})

    await publisher.flush("r1")
    assert store.batches == []  # 首批失败:一帧未持久化
    assert [r["event_type"] for r in sink._pending["r1"]] == [  # noqa: SLF001
        "engine_0",
        "run_done",
    ]  # 原样回队首且保序

    sink.write({"run_id": "r1", "event_type": "engine_1", "payload": {}})
    await publisher.flush("r1")
    assert [r["event_type"] for r in store.batches[0]] == [
        "engine_0",
        "run_done",
        "engine_1",
    ]  # 回插批次在新事件之前,seq 分配顺序不乱
