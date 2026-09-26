"""API 与 Worker 进程之间由 PostgreSQL 承载的尽力而为信号。

本模块是全仓唯一允许绕开 sqlalchemy 引擎、直接 `asyncpg.connect` 的地方:
LISTEN 订阅连接的生死等于进程生死,进不了请求式 session 池——这是登记在案的
PG 第二入口,守卫 test_asyncpg_raw_connections_live_only_in_signals 钉住它。
"""

from __future__ import annotations

import asyncio
import inspect
import itertools
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Literal

SignalKind = Literal["event_committed", "run_cancel_requested", "run_available"]
SignalHandler = Callable[[str], Awaitable[None] | None]

_CHANNEL = "deepresearcher_service_signals"
logger = logging.getLogger("deepresearcher.service.signals")


class PostgresSignalBus:
    """在 PostgreSQL 表保持权威的同时提供低延迟提示。

    信号可能在断连或启动窗口丢失。消费方必须保留持久化的对账路径；
    本总线只消除常规轮询延迟。
    """

    def __init__(self) -> None:
        self._keys = itertools.count(1)
        self._handlers: dict[SignalKind, dict[int, SignalHandler]] = {
            "event_committed": {},
            "run_cancel_requested": {},
            "run_available": {},
        }
        self._listen_connection: Any = None
        self._send_connection: Any = None
        self._callback_tasks: set[asyncio.Task[None]] = set()

    async def start(self, database_url: str) -> None:
        if not database_url.startswith("postgresql"):
            return
        import asyncpg

        dsn = database_url.replace("postgresql+asyncpg://", "postgresql://", 1)
        self._listen_connection = await asyncpg.connect(dsn)
        self._send_connection = await asyncpg.connect(dsn)
        await self._listen_connection.add_listener(_CHANNEL, self._on_notification)

    async def close(self) -> None:
        for handlers in self._handlers.values():
            handlers.clear()
        if self._listen_connection is not None:
            await self._listen_connection.remove_listener(_CHANNEL, self._on_notification)
            await self._listen_connection.close()
            self._listen_connection = None
        if self._send_connection is not None:
            await self._send_connection.close()
            self._send_connection = None
        tasks = tuple(self._callback_tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def subscribe(self, kind: SignalKind, handler: SignalHandler) -> int:
        key = next(self._keys)
        self._handlers[kind][key] = handler
        return key

    def unsubscribe(self, kind: SignalKind, key: int) -> None:
        self._handlers[kind].pop(key, None)

    def subscribe_event(self, run_id: str) -> tuple[int, asyncio.Queue[None]]:
        queue: asyncio.Queue[None] = asyncio.Queue(maxsize=1)

        def wake(payload: str) -> None:
            if payload == run_id and queue.empty():
                queue.put_nowait(None)

        return self.subscribe("event_committed", wake), queue

    async def notify_event(self, run_id: str) -> None:
        await self._notify("event_committed", run_id)

    async def notify_cancel(self, run_id: str) -> None:
        await self._notify("run_cancel_requested", run_id)

    async def notify_work_available(self) -> None:
        await self._notify("run_available", "")

    async def _notify(self, kind: SignalKind, payload: str) -> None:
        await self._dispatch(kind, payload)
        if self._send_connection is None:
            return
        message = json.dumps({"kind": kind, "payload": payload}, separators=(",", ":"))
        try:
            await self._send_connection.execute(f"SELECT pg_notify('{_CHANNEL}', $1)", message)
        except Exception:  # noqa: BLE001 - durable polling/replay preserves correctness
            logger.warning("service_signal_send_failed kind=%s", kind, exc_info=True)

    def _on_notification(self, _connection: Any, _pid: int, _channel: str, raw: str) -> None:
        try:
            message = json.loads(raw)
            kind = message["kind"]
            payload = str(message.get("payload", ""))
            if kind not in self._handlers:
                return
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            logger.warning("service_signal_invalid payload=%r", raw)
            return
        task = asyncio.create_task(self._dispatch(kind, payload), name=f"service-signal-{kind}")
        self._callback_tasks.add(task)
        task.add_done_callback(self._callback_tasks.discard)

    async def _dispatch(self, kind: SignalKind, payload: str) -> None:
        for handler in tuple(self._handlers[kind].values()):
            try:
                result = handler(payload)
            except Exception:  # noqa: BLE001 - one subscriber must not break others
                logger.warning("service_signal_handler_failed kind=%s", kind, exc_info=True)
                continue
            if inspect.isawaitable(result):
                try:
                    await result
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 - signals are an availability fast path
                    logger.warning("service_signal_handler_failed kind=%s", kind, exc_info=True)
