"""Best-effort cross-process RunEvent wakeups backed by PostgreSQL NOTIFY."""

from __future__ import annotations

import asyncio
import itertools
import logging
from typing import Any

_CHANNEL = "deepsearch_run_events"
logger = logging.getLogger("deepsearch_agent.service.events.notifier")


class EventNotifier:
    """Wake DB-tail consumers; correctness never depends on notification delivery."""

    def __init__(self) -> None:
        self._keys = itertools.count(1)
        self._subs: dict[str, dict[int, asyncio.Queue[None]]] = {}
        self._listen_connection: Any = None
        self._send_connection: Any = None

    async def start(self, database_url: str) -> None:
        if not database_url.startswith("postgresql"):
            return
        import asyncpg

        dsn = database_url.replace("postgresql+asyncpg://", "postgresql://", 1)
        self._listen_connection = await asyncpg.connect(dsn)
        self._send_connection = await asyncpg.connect(dsn)
        await self._listen_connection.add_listener(_CHANNEL, self._on_notification)

    async def close(self) -> None:
        if self._listen_connection is not None:
            await self._listen_connection.remove_listener(_CHANNEL, self._on_notification)
            await self._listen_connection.close()
        if self._send_connection is not None:
            await self._send_connection.close()

    def subscribe(self, run_id: str) -> tuple[int, asyncio.Queue[None]]:
        key = next(self._keys)
        queue: asyncio.Queue[None] = asyncio.Queue(maxsize=1)
        self._subs.setdefault(run_id, {})[key] = queue
        return key, queue

    def unsubscribe(self, run_id: str, key: int) -> None:
        self._subs.get(run_id, {}).pop(key, None)

    async def notify(self, run_id: str) -> None:
        self._broadcast(run_id)
        if self._send_connection is not None:
            try:
                await self._send_connection.execute(f"SELECT pg_notify('{_CHANNEL}', $1)", run_id)
            except Exception:  # noqa: BLE001 - DB tail 保证正确性，通知只降延迟
                logger.warning("run_event_notify_failed run_id=%s", run_id, exc_info=True)

    def _on_notification(self, _connection: Any, _pid: int, _channel: str, payload: str) -> None:
        self._broadcast(payload)

    def _broadcast(self, run_id: str) -> None:
        for queue in tuple(self._subs.get(run_id, {}).values()):
            if queue.empty():
                queue.put_nowait(None)
