"""RunEventHub / LocalPreviewBus 闸口契约:收放铁律与降级语义的钉桩。"""

import asyncio

import pytest
import pytest_asyncio

from deepresearcher.service.events.hub import RunEventHub
from deepresearcher.service.events.preview import CLOSE_STREAM, LocalPreviewBus
from deepresearcher.service.events.sinks import CompositeSink

pytestmark = pytest.mark.asyncio


async def _get_message(queue, timeout=1.0):
    return await asyncio.wait_for(queue.get(), timeout)


class FakeStore:
    """模拟 RunEventStore:append 分配 seq 并原样返回;after 供 tail。"""

    def __init__(self):
        self.batches: list[list[dict]] = []
        self.calls: list[str] = []
        self.fail_next = False

    async def append(self, run_id, records):
        self.calls.append("append")
        if self.fail_next:
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
    def __init__(self):
        self.events: list[str] = []

    async def notify_event(self, run_id):
        self.events.append(run_id)


class _NoRunSession:
    async def get(self, _model, _pk):
        return None


class _SessionCtx:
    async def __aenter__(self):
        return _NoRunSession()

    async def __aexit__(self, *_exc):
        return False


@pytest.fixture
def store():
    return FakeStore()


@pytest.fixture
def signal():
    return FakeSignal()


@pytest_asyncio.fixture
async def preview():
    return LocalPreviewBus(asyncio.get_running_loop())


@pytest_asyncio.fixture
async def hub(preview, store, signal):
    return RunEventHub(
        session_factory=lambda: _SessionCtx(),
        store=store,
        preview=preview,
        signal_bus=signal,
    )


# ---------- Hub：收（write/席位） ----------


async def test_write_drops_unrouted_without_open_seat(hub):
    hub.write({"event_type": "x"})  # 无 run_id
    hub.write({"run_id": "never-opened", "event_type": "x"})
    assert hub._pending.get("never-opened", []) == []  # noqa: SLF001


async def test_write_from_worker_thread_and_flush_delivers(hub, preview):
    """任意线程 write → 循环线程 flush 后订阅者收帧(锁纪律 + 回环投递)。"""
    hub.open("run-5")
    _, queue = preview.subscribe("run-5")

    def write_from_thread():
        hub.write({"run_id": "run-5", "event_type": "x", "payload": {}})

    await asyncio.to_thread(write_from_thread)
    await hub.flush("run-5")
    assert (await _get_message(queue))["seq"] == 1


async def test_pydantic_model_records_are_dumped(hub):
    from pydantic import BaseModel

    class Rec(BaseModel):
        run_id: str
        event_type: str
        seq: int | None = None

    hub.open("run-6")
    hub.write(Rec(run_id="run-6", event_type="node_started"))
    with hub._lock:  # noqa: SLF001 - 只读取以断言缓冲语义
        pending = list(hub._pending["run-6"])  # noqa: SLF001
    assert "seq" not in pending[0]


async def test_open_bounded_evicts_oldest_without_termination_sentinel(preview, store, signal):
    """回收 ≠ 终结:最旧席位被逐出时不发 CLOSE_STREAM,只静默拆线降级 DB tail。"""
    hub = RunEventHub(
        session_factory=lambda: _SessionCtx(),
        store=store,
        preview=preview,
        signal_bus=signal,
        open_max=2,
    )
    hub.open("run-old")
    _, old_queue = preview.subscribe("run-old")
    hub.open("run-mid")
    assert hub.is_open("run-old") and hub.is_open("run-mid")

    hub.open("run-new")  # 超限:run-old 按插入序出局
    assert not hub.is_open("run-old")
    assert hub.is_open("run-mid") and hub.is_open("run-new")
    assert old_queue.empty()  # 无哨兵——订阅者不被告知 run 已死
    hub.write({"run_id": "run-old", "event_type": "late", "payload": {}})  # 静默丢弃不炸
    await hub.flush("run-old")
    assert store.batches == []


# ---------- Hub：放（flush 编排与失败语义） ----------


async def test_flush_ordering_append_then_deliver_then_notify(hub, preview, store, signal):
    hub.open("run-7")
    _, queue = preview.subscribe("run-7")
    hub.write({"run_id": "run-7", "event_type": "before", "payload": {}})
    hub.write({"run_id": "run-7", "event_type": "after", "payload": {}})
    await hub.flush("run-7")

    received = [await _get_message(queue), await _get_message(queue)]
    assert [item["event_type"] for item in received] == ["before", "after"]
    assert [item["seq"] for item in received] == [1, 2]
    assert signal.events == ["run-7"]  # 门铃每批一次,在 append 之后
    # pending 已排空
    with hub._lock:  # noqa: SLF001
        assert hub._pending.get("run-7", []) == []  # noqa: SLF001


async def test_flush_requeues_head_on_store_failure(hub, store, signal):
    """排水失败必须回插队首:done 帧整批丢失会让全部 SSE 靠兜底才收敛。"""
    hub.open("r1")
    hub.write({"run_id": "r1", "event_type": "engine_0", "payload": {}})
    hub.write({"run_id": "r1", "event_type": "run_done", "payload": {"status": "completed"}})
    store.fail_next = True

    await hub.flush("r1")
    assert store.batches == []
    with hub._lock:  # noqa: SLF001
        assert [r["event_type"] for r in hub._pending["r1"]] == [  # noqa: SLF001
            "engine_0",
            "run_done",
        ]
    assert signal.events == []  # 失败批不发门铃

    hub.write({"run_id": "r1", "event_type": "engine_1", "payload": {}})
    await hub.flush("r1")
    assert [r["event_type"] for r in store.batches[0]] == [
        "engine_0",
        "run_done",
        "engine_1",
    ]  # 回插批次在新事件之前,seq 分配顺序不乱


async def test_deliver_failure_after_commit_never_requeues(hub, store, signal):
    """铁律:commit 成功即覆水难收——投递抛错只告警,绝不回插造重复批次。"""

    class ExplodingPreview:
        def deliver(self, *_args):
            raise RuntimeError("local push down")

        def close(self, *_args):
            return None

        def drop(self, *_args):
            return None

        def publish_ephemeral(self, *_args):
            return None

    hub._preview = ExplodingPreview()  # noqa: SLF001 - 注入故障
    hub.open("r2")
    hub.write({"run_id": "r2", "event_type": "engine_0", "payload": {}})

    await hub.flush("r2")  # 不得抛出
    assert len(store.batches) == 1  # 已持久化
    with hub._lock:  # noqa: SLF001
        assert hub._pending.get("r2", []) == []  # noqa: SLF001 - 不回插
    assert signal.events == ["r2"]  # 门铃照发(它自带吞错)


async def test_close_sentinels_subscribers_and_late_writes_drop(hub, preview):
    hub.open("run-4")
    _, queue = preview.subscribe("run-4")
    hub.close("run-4")
    assert await _get_message(queue) is CLOSE_STREAM
    hub.write({"run_id": "run-4", "event_type": "late", "payload": {}})
    await hub.flush("run-4")
    assert hub._pending.get("run-4") in (None, [])  # noqa: SLF001


# ---------- Hub：服务合成帧 ----------


async def test_publish_status_then_flush_persists_frame(hub, store):
    hub.open("run-8")
    await hub.publish_status("run-8", "queued")
    await hub.flush("run-8")
    assert store.batches[0][0]["event_type"] == "run_status"


async def test_publish_done_is_idempotent_per_process(hub, store):
    hub.open("run-9")
    await hub.publish_done("run-9")
    await hub.publish_done("run-9")
    await hub.flush("run-9")
    assert [r["event_type"] for b in store.batches for r in b] == ["run_done"]
    assert store.batches[0][0]["payload"]["status"] == "failed"  # 无行时的安全兜底


# ---------- Preview：纯投递半区 ----------


async def test_deliver_reaches_every_subscriber_with_same_object(preview):
    _, queue_a = preview.subscribe("run-1")
    _, queue_b = preview.subscribe("run-1")
    record = {"run_id": "run-1", "event_type": "node_started", "payload": {}, "seq": 1}
    preview.deliver("run-1", [record])
    first = await _get_message(queue_a)
    assert first is record
    assert await _get_message(queue_b) is first


async def test_unsubscribe_stops_delivery(preview):
    key, queue_a = preview.subscribe("run-2")
    _, queue_b = preview.subscribe("run-2")
    preview.unsubscribe("run-2", key)
    preview.deliver("run-2", [{"event_type": "x", "seq": 2}])
    assert queue_a.empty()
    assert (await _get_message(queue_b))["seq"] == 2


async def test_overflow_drops_oldest_and_injects_truncation_marker():
    preview = LocalPreviewBus(asyncio.get_running_loop(), queue_maxsize=2)
    _, queue = preview.subscribe("run-3")
    batch = [
        {"run_id": "run-3", "event_type": f"e{i}", "payload": {}, "seq": i + 1} for i in range(4)
    ]
    preview.deliver("run-3", batch)

    delivered = [await _get_message(queue), await _get_message(queue)]
    assert [item["event_type"] for item in delivered] == ["stream_truncated"] * 2
    assert [item["seq"] for item in delivered] == [1, 2]  # 标记沿用被丢者的 seq
    assert [item["seq"] for item in batch] == [1, 2, 3, 4]  # 批次本身不受溢出影响


async def test_publish_ephemeral_reaches_subscriber(preview):
    _, queue = preview.subscribe("run-4")
    preview.publish_ephemeral("run-4", {"run_id": "run-4", "event_type": "text_delta"})
    frame = await _get_message(queue)
    assert frame["event_type"] == "text_delta" and "seq" not in frame


async def test_drop_is_silent_and_close_sentinels(preview):
    _, queue_dropped = preview.subscribe("run-d")
    preview.drop("run-d")
    assert queue_dropped.empty()
    preview.deliver("run-d", [{"event_type": "late"}])  # 拆线后投递 no-op
    assert queue_dropped.empty()

    _, queue_closed = preview.subscribe("run-c")
    preview.close("run-c")
    assert await _get_message(queue_closed) is CLOSE_STREAM


async def test_deliver_from_other_thread_hops_to_loop(preview):
    _, queue = preview.subscribe("run-t")
    await asyncio.to_thread(preview.deliver, "run-t", [{"event_type": "x", "seq": 9}])
    assert (await _get_message(queue))["seq"] == 9


# ---------- CompositeSink ----------


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
