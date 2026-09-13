"""Redis Pub/Sub implementation for lossy cross-process token previews."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from deepresearcher.service.events.ephemeral import (
    EphemeralEventBus,
    EphemeralSubscription,
    preview_event,
)

logger = logging.getLogger("deepresearcher.service.events.redis_ephemeral")
REDIS_PREVIEW_IO_TIMEOUT_SECONDS = 1.0


class RedisEphemeralEventBus(EphemeralEventBus):
    """Publish previews without turning Redis into a source of truth."""

    def __init__(self, client: Any, *, channel_prefix: str, queue_size: int) -> None:
        self._client = client
        self._prefix = channel_prefix.strip(":") or "deepresearcher"
        self._queue_size = max(1, queue_size)
        self._publish_queue: asyncio.Queue[tuple[str, dict]] = asyncio.Queue(
            maxsize=self._queue_size
        )
        self._publisher: asyncio.Task[None] | None = None
        self._closed = False

    def _channel(self, run_id: str) -> str:
        return f"{self._prefix}:run:{run_id}:preview"

    async def publish(self, run_id: str, event: dict) -> None:
        safe = preview_event(event, run_id=run_id)
        if safe is None or self._closed:
            return
        if self._publish_queue.full():
            try:
                self._publish_queue.get_nowait()
                self._publish_queue.task_done()
            except asyncio.QueueEmpty:  # pragma: no cover - same-loop check/get
                pass
        self._publish_queue.put_nowait((run_id, safe))
        if self._publisher is None or self._publisher.done():
            self._publisher = asyncio.create_task(
                self._drain_publishes(),
                name="redis-preview-publisher",
            )

    async def _drain_publishes(self) -> None:
        while True:
            run_id, event = await self._publish_queue.get()
            try:
                await self._client.publish(
                    self._channel(run_id),
                    json.dumps(event, ensure_ascii=False, separators=(",", ":")),
                )
            except Exception:  # noqa: BLE001 - preview loss must not fail a run
                logger.warning("redis_preview_publish_failed run_id=%s", run_id, exc_info=True)
            finally:
                self._publish_queue.task_done()

    async def subscribe(self, run_id: str) -> EphemeralSubscription:
        queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=self._queue_size)
        pubsub = self._client.pubsub()
        await pubsub.subscribe(self._channel(run_id))
        reader = asyncio.create_task(
            self._read(pubsub, run_id, queue),
            name=f"redis-preview-{run_id}",
        )

        async def close() -> None:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
            try:
                await pubsub.unsubscribe(self._channel(run_id))
                await pubsub.aclose()
            except Exception:  # noqa: BLE001 - request teardown is best effort
                logger.debug("redis_preview_unsubscribe_failed", exc_info=True)

        return EphemeralSubscription(queue=queue, _close=close)

    async def _read(self, pubsub: Any, run_id: str, queue: asyncio.Queue[dict]) -> None:
        try:
            async for message in pubsub.listen():
                if message.get("type") != "message":
                    continue
                raw = message.get("data")
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                try:
                    decoded = json.loads(str(raw))
                except (TypeError, ValueError):
                    continue
                event = preview_event(decoded, run_id=run_id)
                if event is None:
                    continue
                if queue.full():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:  # pragma: no cover - same-loop check/get
                        pass
                queue.put_nowait(event)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - DB events remain the correctness path
            logger.warning("redis_preview_subscription_failed run_id=%s", run_id, exc_info=True)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._publisher is not None:
            self._publisher.cancel()
            await asyncio.gather(self._publisher, return_exceptions=True)
        try:
            await self._client.aclose()
        except Exception:  # noqa: BLE001 - shutdown remains best effort
            logger.debug("redis_preview_close_failed", exc_info=True)


async def create_redis_ephemeral_bus(
    redis_url: str,
    *,
    channel_prefix: str,
    queue_size: int,
) -> RedisEphemeralEventBus | None:
    """Connect when configured; degrade to durable-only streaming on failure."""

    try:
        from redis.asyncio import Redis

        client = Redis.from_url(
            redis_url,
            decode_responses=False,
            socket_connect_timeout=REDIS_PREVIEW_IO_TIMEOUT_SECONDS,
            socket_timeout=REDIS_PREVIEW_IO_TIMEOUT_SECONDS,
        )
        await client.ping()
    except Exception:  # noqa: BLE001 - Redis is explicitly non-authoritative
        logger.warning("redis_preview_unavailable; durable SSE remains active", exc_info=True)
        return None
    return RedisEphemeralEventBus(
        client,
        channel_prefix=channel_prefix,
        queue_size=queue_size,
    )
