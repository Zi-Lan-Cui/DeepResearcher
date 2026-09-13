"""Lossy cross-process preview transport contracts.

Ephemeral events improve live presentation only. They have no sequence number,
are never persisted, and must never participate in run-state decisions.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class EphemeralSubscription:
    """One bounded preview subscription owned by an SSE request."""

    queue: asyncio.Queue[dict]
    _close: Callable[[], Awaitable[None]]
    _closed: bool = field(default=False, init=False)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._close()


class EphemeralEventBus(Protocol):
    """Best-effort preview bus; implementations must absorb transport failure."""

    async def publish(self, run_id: str, event: dict) -> None: ...

    async def subscribe(self, run_id: str) -> EphemeralSubscription: ...

    async def close(self) -> None: ...


def preview_event(value: object, *, run_id: str) -> dict | None:
    """Validate and rebuild the only event shape allowed on the lossy bus."""

    if not isinstance(value, dict) or value.get("run_id") != run_id:
        return None
    if value.get("event_type") != "text_delta":
        return None
    payload = value.get("payload")
    if not isinstance(payload, dict):
        return None
    channel = str(payload.get("channel") or "")
    text = str(payload.get("text") or "")[:200]
    if channel not in {"router", "clarify", "supervisor", "writer"} or not text:
        return None
    return {
        "run_id": run_id,
        "event_type": "text_delta",
        "payload": {"channel": channel, "text": text},
    }
