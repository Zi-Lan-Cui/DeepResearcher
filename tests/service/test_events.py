"""RunEventBuffer 接口契约与 LocalPreviewBus 协议实现的回归测试。

Buffer 只负责 open 登记、pending 缓冲、flush(append→notify) 的失败语义。
已提交事件不做进程内直投，写库后以 DB 副本为唯一来源；done 帧的去重键由
调用方带来，重复写入交给 run_events 的唯一约束，进程内不另记状态。
"""

import asyncio

import pytest
import pytest_asyncio

from deepresearcher.service.events.buffer import RunEventBuffer
from deepresearcher.service.events.sinks import CompositeSink
from deepresearcher.service.preview.local import LocalPreviewBus
from deepresearcher.service.preview.protocol import EphemeralEventBus

pytestmark = pytest.mark.asyncio


async def _get_message(queue, timeout=1.0):
    return await asyncio.wait_for(queue.get(), timeout)


class FakeStore:
    """模拟 RunEventStore:append 分配 seq;after 作为 SSE 尾读接口。"""

    def __init__(self, log: list[str]):
        self.batches: list[list[dict]] = []
        self.log = log
        self.fail_next = False
        self.fail_on_call: int | None = None
        self.calls = 0

    async def append(self, run_id, records):
        self.log.append("append")
        self.calls += 1
        if self.fail_next or self.calls == self.fail_on_call:
            self.fail_next = False
            raise ConnectionError("db blip")
        assigned = []
        start = sum(len(b) for b in self.batches) + 1
        for offset, record in enumerate(records):
            item = dict(record)
            item["seq"] = start + offset
            assigned.append(item)
        self.batches.append(assigned)
        return assigned

    async def after(self, run_id, seq):
        return []


class FakeSignal:
    def __init__(self, log: list[str]):
        self.log = log

    async def notify_event(self, run_id):
        self.log.append(f"notify:{run_id}")


@pytest.fixture
def log():
    return []


@pytest.fixture
def store(log):
    return FakeStore(log)


@pytest.fixture
def signal(log):
    return FakeSignal(log)


@pytest_asyncio.fixture
async def buffer(store, signal):
    return RunEventBuffer(store=store, signal_bus=signal)


@pytest_asyncio.fixture
async def preview_bus():
    bus = LocalPreviewBus()
    yield bus
    await bus.close()


async def test_write_drops_unrouted_without_open_seat(buffer):
    buffer.write({"event_type": "x"})  # 无 run_id
    buffer.write({"run_id": "never-opened", "event_type": "x"})
    assert buffer._pending.get("never-opened", []) == []  # noqa: SLF001


async def test_write_from_worker_thread_and_flush_persists(buffer, store, log):
    """任意线程 write → 循环线程 flush 落库并发 notify(锁纪律)。"""
    buffer.open("run-5")

    def write_from_thread():
        buffer.write({"run_id": "run-5", "event_type": "x", "payload": {}})

    await asyncio.to_thread(write_from_thread)
    await buffer.flush("run-5")
    assert store.batches[0][0]["seq"] == 1
    assert log == ["append", "notify:run-5"]


async def test_pydantic_model_records_are_dumped(buffer):
    from pydantic import BaseModel

    class Rec(BaseModel):
        run_id: str
        event_type: str
        seq: int | None = None

    buffer.open("run-6")
    buffer.write(Rec(run_id="run-6", event_type="node_started"))
    with buffer._lock:  # noqa: SLF001 - 只读取以断言缓冲语义
        pending = list(buffer._pending["run-6"])  # noqa: SLF001
    assert "seq" not in pending[0]


async def test_signal_bus_serializes_concurrent_notify():
    """同一连接上的并发 notify 串行发送;断言连接在途峰值始终为 1。"""
    from deepresearcher.service.coordination.signals import PostgresSignalBus

    class _ConnectionProbe:
        def __init__(self):
            self.in_flight = 0
            self.peak = 0
            self.sent = 0

        async def execute(self, _sql, _message):
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            await asyncio.sleep(0)
            self.in_flight -= 1
            self.sent += 1

    bus = PostgresSignalBus()
    probe = _ConnectionProbe()
    bus._send_connection = probe
    await asyncio.gather(*(bus.notify_event(f"run-{index}") for index in range(8)))
    assert probe.sent == 8
    assert probe.peak == 1, "同连接 notify 必须串行"


async def test_open_bounded_evicts_oldest(store, signal):
    """open 登记超额按插入序回收;被逐 run 的迟到 write/flush 静默 no-op。"""
    buffer = RunEventBuffer(store=store, signal_bus=signal, open_max=2)
    buffer.open("run-old")
    buffer.open("run-mid")
    assert buffer.is_open("run-old") and buffer.is_open("run-mid")

    buffer.open("run-new")  # 超限：run-old 按插入序出局
    assert not buffer.is_open("run-old")
    assert buffer.is_open("run-mid") and buffer.is_open("run-new")
    buffer.write({"run_id": "run-old", "event_type": "late", "payload": {}})  # 静默丢弃，不抛错
    await buffer.flush("run-old")
    assert store.batches == []


async def test_flush_orders_append_then_notify(buffer, store, log):
    buffer.open("run-7")
    buffer.write({"run_id": "run-7", "event_type": "before", "payload": {}})
    buffer.write({"run_id": "run-7", "event_type": "after", "payload": {}})
    await buffer.flush("run-7")

    assert log == ["append", "notify:run-7"]  # notify 每批一次，在 append 之后
    assert [r["event_type"] for r in store.batches[0]] == ["before", "after"]
    assert [r["seq"] for r in store.batches[0]] == [1, 2]
    with buffer._lock:  # noqa: SLF001
        assert buffer._pending.get("run-7", []) == []  # noqa: SLF001


async def test_flush_requeues_head_on_store_failure(buffer, store, log):
    """排水失败必须回插队首:done 帧整批丢失会让全部 SSE 只能靠合成帧结束。"""
    buffer.open("r1")
    buffer.write({"run_id": "r1", "event_type": "engine_0", "payload": {}})
    buffer.write({"run_id": "r1", "event_type": "run_done", "payload": {"status": "completed"}})
    store.fail_next = True

    await buffer.flush("r1")
    assert store.batches == []
    with buffer._lock:  # noqa: SLF001
        assert [r["event_type"] for r in buffer._pending["r1"]] == [  # noqa: SLF001
            "engine_0",
            "run_done",
        ]
    assert log == ["append"]  # 失败的 append 有记录，notify 不发

    buffer.write({"run_id": "r1", "event_type": "engine_1", "payload": {}})
    await buffer.flush("r1")
    assert [r["event_type"] for r in store.batches[0]] == [
        "engine_0",
        "run_done",
        "engine_1",
    ]  # 回插批次在新事件之前，seq 分配顺序不乱


async def test_close_drops_seat_and_late_writes_drop(buffer, store):
    buffer.open("run-4")
    buffer.close("run-4")
    assert not buffer.is_open("run-4")
    buffer.write({"run_id": "run-4", "event_type": "late", "payload": {}})
    await buffer.flush("run-4")
    assert buffer._pending.get("run-4") in (None, [])  # noqa: SLF001
    assert store.batches == []


async def test_write_status_then_flush_persists_frame(buffer, store):
    buffer.open("run-8")
    buffer.write_status("run-8", "queued")
    await buffer.flush("run-8")
    assert store.batches[0][0]["event_type"] == "run_status"


async def test_write_done_carries_state_and_dedup_key(buffer, store):
    buffer.open("run-9")
    buffer.write_done("run-9", status="completed", answer_mode="report", report_available=True)
    await buffer.flush("run-9")
    record = store.batches[0][0]
    assert record["event_type"] == "run_done"
    assert record["dedup_key"] == "terminal"
    assert record["payload"] == {
        "status": "completed",
        "answer_mode": "report",
        "report_available": True,
    }


async def test_flush_reports_whether_batch_reached_store(buffer, store, log):
    """返回值是周期任务判断退避的唯一依据：无待写算完成，写入失败才算未完成。"""
    buffer.open("run-10")
    assert await buffer.flush("run-10") is True  # 空批次不碰库
    assert log == []

    buffer.write({"run_id": "run-10", "event_type": "x", "payload": {}})
    assert await buffer.flush("run-10") is True
    assert log == ["append", "notify:run-10"]

    buffer.write({"run_id": "run-10", "event_type": "y", "payload": {}})
    store.fail_next = True
    assert await buffer.flush("run-10") is False


async def test_flush_drains_in_bounded_batches(buffer, store):
    """库恢复后不能一次写完整批：大事务自身更容易超时，失败又要整批回插，越积越难消化。"""
    from deepresearcher.service.events.buffer import _FLUSH_BATCH_MAX

    buffer.open("run-11")
    total = _FLUSH_BATCH_MAX * 2 + 3
    for index in range(total):
        buffer.write({"run_id": "run-11", "event_type": f"e{index}", "payload": {}})

    assert await buffer.flush("run-11") is True
    assert [len(batch) for batch in store.batches] == [
        _FLUSH_BATCH_MAX,
        _FLUSH_BATCH_MAX,
        3,
    ]  # 分片但仍在本轮排空：终态路径紧随 close，留尾巴等于丢帧
    with buffer._lock:  # noqa: SLF001
        assert buffer._pending.get("run-11") in (None, [])  # noqa: SLF001


async def test_flush_stops_at_failed_batch_and_keeps_remainder(buffer, store):
    """中途失败不回滚已写批次：已落库的帧有 seq，重写会撞唯一约束，只需保住失败批与后续。"""
    from deepresearcher.service.events.buffer import _FLUSH_BATCH_MAX

    buffer.open("run-13")
    total = _FLUSH_BATCH_MAX * 2 + 3
    for index in range(total):
        buffer.write({"run_id": "run-13", "event_type": f"e{index}", "payload": {}})
    store.fail_on_call = 2

    assert await buffer.flush("run-13") is False
    assert [len(batch) for batch in store.batches] == [_FLUSH_BATCH_MAX]
    with buffer._lock:  # noqa: SLF001
        assert len(buffer._pending["run-13"]) == total - _FLUSH_BATCH_MAX  # noqa: SLF001


async def test_flush_completes_when_notify_fails(store):
    """帧已落库，通知只是唤醒；订阅方还能靠 DB tail 追平，不该判整批失败去重试。"""

    class _BrokenSignal:
        async def notify_event(self, run_id):
            raise ConnectionError("notify down")

    buffer = RunEventBuffer(store=store, signal_bus=_BrokenSignal())
    buffer.open("run-12")
    buffer.write({"run_id": "run-12", "event_type": "x", "payload": {}})

    assert await buffer.flush("run-12") is True
    assert [record["event_type"] for record in store.batches[0]] == ["x"]
    with buffer._lock:  # noqa: SLF001
        assert buffer._pending.get("run-12") in (None, [])  # noqa: SLF001


async def test_local_bus_satisfies_protocol():
    bus = LocalPreviewBus()
    assert isinstance(bus, EphemeralEventBus)
    await bus.close()


async def test_publish_whitelist_and_no_seq(preview_bus):
    subscription = await preview_bus.subscribe("run-1")
    await preview_bus.publish(
        "run-1",
        {
            "run_id": "run-1",
            "event_type": "text_delta",
            "payload": {"channel": "supervisor", "text": "甲"},
        },
    )
    for invalid in (
        {
            "run_id": "other-run",
            "event_type": "text_delta",
            "payload": {"channel": "supervisor", "text": "串门"},
        },
        {"run_id": "run-1", "event_type": "run_status", "payload": {}},
        {
            "run_id": "run-1",
            "event_type": "text_delta",
            "payload": {"channel": "supervisor", "text": ""},
        },
    ):
        await preview_bus.publish("run-1", invalid)
    frame = await _get_message(subscription.queue)
    assert frame["payload"]["text"] == "甲" and "seq" not in frame
    assert subscription.queue.empty()  # 白名单外的帧没有排队


async def test_overflow_drops_oldest_without_marker():
    """预览帧允许丢弃:队满丢最旧,且不插入截断标记;已提交帧不走这条通道。"""
    bus = LocalPreviewBus(queue_maxsize=2)
    subscription = await bus.subscribe("run-3")
    for index in range(4):
        await bus.publish(
            "run-3",
            {
                "run_id": "run-3",
                "event_type": "text_delta",
                "payload": {"channel": "supervisor", "text": f"t{index}"},
            },
        )
    kept = [await _get_message(subscription.queue), await _get_message(subscription.queue)]
    assert [frame["payload"]["text"] for frame in kept] == ["t2", "t3"]
    assert subscription.queue.empty()


async def test_subscription_close_stops_delivery(preview_bus):
    first = await preview_bus.subscribe("run-4")
    second = await preview_bus.subscribe("run-4")
    await first.close()
    await first.close()  # 幂等
    await preview_bus.publish(
        "run-4",
        {
            "run_id": "run-4",
            "event_type": "text_delta",
            "payload": {"channel": "supervisor", "text": "x"},
        },
    )
    assert first.queue.empty()
    assert not second.queue.empty()


async def test_bus_close_releases_all_and_shuts_publish(preview_bus):
    subscription = await preview_bus.subscribe("run-5")
    await preview_bus.close()
    await preview_bus.publish(
        "run-5",
        {
            "run_id": "run-5",
            "event_type": "text_delta",
            "payload": {"channel": "supervisor", "text": "x"},
        },
    )
    assert subscription.queue.empty()


async def test_prune_event_jsonl_by_mtime(tmp_path):
    import os
    import time as _time

    from deepresearcher.service.runs.retention import prune_event_jsonl

    events = tmp_path / "events"
    events.mkdir()
    old_log = events / "run-old.jsonl"
    fresh_log = events / "run-new.jsonl"
    other = events / "keep.txt"
    for f in (old_log, fresh_log, other):
        f.write_text("x")
    stale = _time.time() - 40 * 86_400
    os.utime(old_log, (stale, stale))

    assert prune_event_jsonl(events, retention_days=14) == 1
    assert not old_log.exists() and fresh_log.exists() and other.exists()
    assert prune_event_jsonl(events, retention_days=0) == 0  # 0 表示关闭清理
    assert prune_event_jsonl(tmp_path / "missing", retention_days=14) == 0


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
