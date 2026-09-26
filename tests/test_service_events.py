"""RunEventHub 接口契约与 LocalPreviewBus 协议实现的回归测试。

Hub 只负责:open 登记、pending 缓冲、flush(append→notify)的失败语义。
已提交事件不做进程内直投,落库后以 DB 副本为唯一来源。
"""

import asyncio

import pytest
import pytest_asyncio

from deepresearcher.service.events.hub import RunEventHub
from deepresearcher.service.events.sinks import CompositeSink
from deepresearcher.service.preview.local import LocalPreviewBus
from deepresearcher.service.preview.protocol import EphemeralEventBus

pytestmark = pytest.mark.asyncio


async def _get_message(queue, timeout=1.0):
    return await asyncio.wait_for(queue.get(), timeout)


class FakeStore:
    """模拟 RunEventStore:append 分配 seq;after 供 tail。"""

    def __init__(self, log: list[str]):
        self.batches: list[list[dict]] = []
        self.log = log
        self.fail_next = False

    async def append(self, run_id, records):
        self.log.append("append")
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
    def __init__(self, log: list[str]):
        self.log = log

    async def notify_event(self, run_id):
        self.log.append(f"notify:{run_id}")


class _NoRunSession:
    async def get(self, _model, _pk):
        return None


class _SessionCtx:
    async def __aenter__(self):
        return _NoRunSession()

    async def __aexit__(self, *_exc):
        return False


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
async def hub(store, signal):
    return RunEventHub(
        session_factory=lambda: _SessionCtx(),
        store=store,
        signal_bus=signal,
    )


@pytest_asyncio.fixture
async def preview_bus():
    bus = LocalPreviewBus()
    yield bus
    await bus.close()


# ---------- Hub：收（write/open 登记） ----------


async def test_write_drops_unrouted_without_open_seat(hub):
    hub.write({"event_type": "x"})  # 无 run_id
    hub.write({"run_id": "never-opened", "event_type": "x"})
    assert hub._pending.get("never-opened", []) == []  # noqa: SLF001


async def test_write_from_worker_thread_and_flush_persists(hub, store, log):
    """任意线程 write → 循环线程 flush 落库并发 notify(锁纪律)。"""
    hub.open("run-5")

    def write_from_thread():
        hub.write({"run_id": "run-5", "event_type": "x", "payload": {}})

    await asyncio.to_thread(write_from_thread)
    await hub.flush("run-5")
    assert store.batches[0][0]["seq"] == 1
    assert log == ["append", "notify:run-5"]


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


async def test_open_bounded_evicts_oldest(store, signal):
    """open 登记超额按插入序回收;被逐 run 的迟到 write/flush 静默 no-op。"""
    hub = RunEventHub(
        session_factory=lambda: _SessionCtx(),
        store=store,
        signal_bus=signal,
        open_max=2,
    )
    hub.open("run-old")
    hub.open("run-mid")
    assert hub.is_open("run-old") and hub.is_open("run-mid")

    hub.open("run-new")  # 超限:run-old 按插入序出局
    assert not hub.is_open("run-old")
    assert hub.is_open("run-mid") and hub.is_open("run-new")
    hub.write({"run_id": "run-old", "event_type": "late", "payload": {}})  # 静默丢弃不炸
    await hub.flush("run-old")
    assert store.batches == []


# ---------- Hub：放（flush 编排与失败语义） ----------


async def test_flush_orders_append_then_notify(hub, store, log):
    hub.open("run-7")
    hub.write({"run_id": "run-7", "event_type": "before", "payload": {}})
    hub.write({"run_id": "run-7", "event_type": "after", "payload": {}})
    await hub.flush("run-7")

    assert log == ["append", "notify:run-7"]  # notify 每批一次,在 append 之后
    assert [r["event_type"] for r in store.batches[0]] == ["before", "after"]
    assert [r["seq"] for r in store.batches[0]] == [1, 2]
    with hub._lock:  # noqa: SLF001
        assert hub._pending.get("run-7", []) == []  # noqa: SLF001


async def test_flush_requeues_head_on_store_failure(hub, store, log):
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
    assert log == ["append"]  # 失败的 append 有记录，notify 不发

    hub.write({"run_id": "r1", "event_type": "engine_1", "payload": {}})
    await hub.flush("r1")
    assert [r["event_type"] for r in store.batches[0]] == [
        "engine_0",
        "run_done",
        "engine_1",
    ]  # 回插批次在新事件之前,seq 分配顺序不乱


async def test_close_drops_seat_and_late_writes_drop(hub, store):
    hub.open("run-4")
    hub.close("run-4")
    assert not hub.is_open("run-4")
    hub.write({"run_id": "run-4", "event_type": "late", "payload": {}})
    await hub.flush("run-4")
    assert hub._pending.get("run-4") in (None, [])  # noqa: SLF001
    assert store.batches == []


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


# ---------- LocalPreviewBus：EphemeralEventBus 的同环实现 ----------


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
    """预览可丢:队满丢最旧,不注入截断标记——已提交帧根本不走这条通道。"""
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
