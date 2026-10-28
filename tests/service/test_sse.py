"""SSE 尾随链路的单元测试，不架起 DB、buffer 与预览总线。"""

import asyncio

import pytest

from deepresearcher.service.web.routes import events as events_module

pytestmark = pytest.mark.asyncio


class _Request:
    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self.headers = headers or {}


class _Subscription:
    def __init__(self) -> None:
        self.queue: asyncio.Queue = asyncio.Queue()
        self.closed = False

    async def close(self) -> None:
        self.closed = True


async def test_wait_next_frames_returns_empty_on_timeout(monkeypatch):
    """两条唤醒源都静默时按轮询间隔返回空，且不留下挂起的读取任务。"""
    monkeypatch.setattr(events_module, "SSE_DB_POLL_SECONDS", 0.05)
    notify_queue: asyncio.Queue = asyncio.Queue()

    before = len(asyncio.all_tasks())
    items = await events_module._wait_next_frames(notify_queue, None)

    assert items == []
    assert len(asyncio.all_tasks()) == before, "挂起的 queue.get 必须回收"


async def test_wait_next_frames_prefers_preview_payload():
    notify_queue: asyncio.Queue = asyncio.Queue()
    subscription = _Subscription()
    await subscription.queue.put({"text": "hi"})

    items = await events_module._wait_next_frames(notify_queue, subscription)

    assert items == [{"text": "hi"}]


async def test_wait_next_frames_returns_empty_for_bare_notify():
    """notify 只是唤醒信号，payload 是 None；数据由下一轮循环头的 tail 取。"""
    notify_queue: asyncio.Queue = asyncio.Queue()
    subscription = _Subscription()
    await notify_queue.put(None)

    items = await events_module._wait_next_frames(notify_queue, subscription)

    assert items == []


async def test_next_outputs_emits_heartbeat_when_due(monkeypatch):
    """到点才发心跳，且返回新的计时起点；计时基准是事件循环时钟，不是 0。"""
    monkeypatch.setattr(events_module, "SSE_DB_POLL_SECONDS", 0.01)
    monkeypatch.setattr(events_module, "SSE_HEARTBEAT_SECONDS", 0.0)
    notify_queue: asyncio.Queue = asyncio.Queue()
    now = asyncio.get_running_loop().time()

    texts, last_ping = await events_module._next_outputs(notify_queue, None, now)

    assert texts == [": ping\n\n"]
    assert last_ping >= now


async def test_next_outputs_skips_heartbeat_before_due(monkeypatch):
    monkeypatch.setattr(events_module, "SSE_DB_POLL_SECONDS", 0.01)
    monkeypatch.setattr(events_module, "SSE_HEARTBEAT_SECONDS", 600.0)
    notify_queue: asyncio.Queue = asyncio.Queue()
    now = asyncio.get_running_loop().time()

    texts, last_ping = await events_module._next_outputs(notify_queue, None, now)

    assert texts == []
    assert last_ping == now


async def test_next_outputs_drops_frames_outside_whitelist():
    """预览帧仍要过投影白名单：不在表里的通道返回 None，不产出。"""
    notify_queue: asyncio.Queue = asyncio.Queue()
    subscription = _Subscription()
    await subscription.queue.put({"event_type": "text_delta", "payload": {"channel": "unknown"}})

    texts, _ = await events_module._next_outputs(notify_queue, subscription, 0.0)

    assert texts == []


async def test_resume_seq_reads_last_event_id():
    assert events_module._resume_seq(_Request()) == 0
    assert events_module._resume_seq(_Request({"last-event-id": "12"})) == 12
    assert events_module._resume_seq(_Request({"last-event-id": "-3"})) == 0
    assert events_module._resume_seq(_Request({"last-event-id": "abc"})) == 0


async def test_subscribe_preview_returns_none_without_bus():
    class _State:
        ephemeral_bus = None

    assert await events_module._subscribe_preview(_State(), "run-1") is None


async def test_subscribe_preview_swallows_bus_failure():
    """预览总线挂掉只能丢 text_delta，持久流必须照常可读。"""

    class _BrokenBus:
        async def subscribe(self, run_id):
            raise ConnectionError("preview down")

    class _State:
        ephemeral_bus = _BrokenBus()

    assert await events_module._subscribe_preview(_State(), "run-1") is None


async def test_terminate_synthesizes_done_past_last_seq():
    """done 帧整批丢了才合成，且合成帧的 seq 必须大于已投最大 seq。

    同号帧会被前端当成回放丢弃，合成的终态反而送不到客户端。
    """

    class _Store:
        async def after(self, run_id, seq):
            return []

    class _Manager:
        event_store = _Store()

        async def terminal_state(self, run_id):
            return {"status": "completed", "answer_mode": "report", "report_available": True}

    class _State:
        manager = _Manager()

    frames, last_seq, settled = await events_module._terminate_if_settled(_State(), "run-1", 7)

    assert settled is True
    assert last_seq == 7
    assert len(frames) == 1
    assert "id: 8" in frames[0]
    assert "event: done" in frames[0]


async def test_terminate_keeps_running_when_no_terminal_row():
    class _Store:
        async def after(self, run_id, seq):
            return []

    class _Manager:
        event_store = _Store()

        async def terminal_state(self, run_id):
            return None

    class _State:
        manager = _Manager()

    frames, last_seq, settled = await events_module._terminate_if_settled(_State(), "run-1", 3)

    assert frames == []
    assert last_seq == 3
    assert settled is False
