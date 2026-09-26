import asyncio
import json
import os
from uuid import uuid4

import pytest

from deepresearcher.service.preview.protocol import preview_event
from deepresearcher.service.preview.redis import (
    RedisEphemeralEventBus,
    create_redis_ephemeral_bus,
)


class FakePubSub:
    def __init__(self):
        self.channel = None
        self.messages = asyncio.Queue()
        self.closed = False

    async def subscribe(self, channel):
        self.channel = channel

    async def unsubscribe(self, _channel):
        return None

    async def aclose(self):
        self.closed = True

    async def listen(self):
        while True:
            yield await self.messages.get()


class FakeRedis:
    def __init__(self):
        self.published = []
        self.pubsub_instance = FakePubSub()
        self.closed = False

    async def publish(self, channel, payload):
        self.published.append((channel, payload))

    def pubsub(self):
        return self.pubsub_instance

    async def aclose(self):
        self.closed = True


class BlockingRedis(FakeRedis):
    def __init__(self):
        super().__init__()
        self.release = asyncio.Event()

    async def publish(self, channel, payload):
        await self.release.wait()
        await super().publish(channel, payload)


def _event(run_id="run-1", text="preview"):
    return {
        "run_id": run_id,
        "event_type": "text_delta",
        "payload": {"channel": "supervisor", "text": text},
    }


@pytest.mark.asyncio
async def test_redis_preview_publish_is_safe_and_unsequenced():
    client = FakeRedis()
    bus = RedisEphemeralEventBus(client, channel_prefix="test", queue_size=2)

    await bus.publish("run-1", _event(text="x" * 300))
    async with asyncio.timeout(1):
        while not client.published:
            await asyncio.sleep(0)
    channel, raw = client.published[0]
    decoded = json.loads(raw)

    assert channel == "test:run:run-1:preview"
    assert decoded == _event(text="x" * 200)
    assert "seq" not in decoded
    await bus.close()


@pytest.mark.asyncio
async def test_redis_preview_subscription_filters_and_drops_oldest():
    client = FakeRedis()
    bus = RedisEphemeralEventBus(client, channel_prefix="test", queue_size=1)
    subscription = await bus.subscribe("run-1")

    await client.pubsub_instance.messages.put({"type": "message", "data": "not-json"})
    await client.pubsub_instance.messages.put(
        {"type": "message", "data": json.dumps(_event(run_id="another"))}
    )
    await client.pubsub_instance.messages.put(
        {"type": "message", "data": json.dumps(_event(text="first"))}
    )
    await client.pubsub_instance.messages.put(
        {"type": "message", "data": json.dumps(_event(text="latest"))}
    )
    async with asyncio.timeout(1):
        while subscription.queue.empty():
            await asyncio.sleep(0)
    assert await subscription.queue.get() == _event(text="latest")

    await subscription.close()
    assert client.pubsub_instance.closed


@pytest.mark.asyncio
async def test_redis_preview_publish_never_waits_for_network_and_drops_oldest():
    client = BlockingRedis()
    bus = RedisEphemeralEventBus(client, channel_prefix="test", queue_size=1)

    async with asyncio.timeout(0.1):
        await bus.publish("run-1", _event(text="first"))
        await asyncio.sleep(0)  # publisher takes the first item and blocks in Redis
        await bus.publish("run-1", _event(text="discarded"))
        await bus.publish("run-1", _event(text="latest"))

    client.release.set()
    async with asyncio.timeout(1):
        while len(client.published) < 2:
            await asyncio.sleep(0)
    assert [json.loads(raw)["payload"]["text"] for _, raw in client.published] == [
        "first",
        "latest",
    ]
    await bus.close()


def test_preview_event_rejects_business_events_and_unknown_channels():
    assert preview_event({"run_id": "run-1", "event_type": "run_done"}, run_id="run-1") is None
    assert (
        preview_event(
            {
                "run_id": "run-1",
                "event_type": "text_delta",
                "payload": {"channel": "secret", "text": "hidden"},
            },
            run_id="run-1",
        )
        is None
    )


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("SERVICE_TEST_REDIS_URL"),
    reason="set SERVICE_TEST_REDIS_URL to run the real Redis Pub/Sub smoke test",
)
async def test_real_redis_cross_client_preview_roundtrip():
    redis_url = os.environ["SERVICE_TEST_REDIS_URL"]
    run_id = f"test-{uuid4().hex}"
    publisher = await create_redis_ephemeral_bus(
        redis_url,
        channel_prefix="deepresearcher-tests",
        queue_size=4,
    )
    subscriber = await create_redis_ephemeral_bus(
        redis_url,
        channel_prefix="deepresearcher-tests",
        queue_size=4,
    )
    assert publisher is not None and subscriber is not None
    subscription = await subscriber.subscribe(run_id)
    try:
        await publisher.publish(run_id, _event(run_id=run_id, text="live"))
        async with asyncio.timeout(2):
            assert await subscription.queue.get() == _event(run_id=run_id, text="live")
    finally:
        await subscription.close()
        await publisher.close()
        await subscriber.close()
