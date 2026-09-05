import asyncio
import json

import pytest

from deepsearch_agent.service.events.ephemeral import preview_event
from deepsearch_agent.service.events.redis_ephemeral import RedisEphemeralEventBus


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
    channel, raw = client.published[0]
    decoded = json.loads(raw)

    assert channel == "test:run:run-1:preview"
    assert decoded == _event(text="x" * 200)
    assert "seq" not in decoded


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
