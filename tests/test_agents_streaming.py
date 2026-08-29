
import pytest

from deepsearch_agent.agents.streaming import ainvoke_agent_with_deltas

pytestmark = pytest.mark.asyncio


class _Chunk:
    def __init__(self, blocks):
        self.content_blocks = blocks


class FakeLoop:
    def __init__(self, final_state, message_chunks):
        self._final = final_state
        self._chunks = message_chunks
        self.ainvoke_calls = []

    async def ainvoke(self, state, **kwargs):
        self.ainvoke_calls.append((state, kwargs))
        return self._final

    async def astream(self, state, **kwargs):
        for index, chunk in enumerate(self._chunks):
            yield ("messages", (chunk, {"langgraph_node": "model"}))
            if index == 0:
                yield ("values", {"partial": True})  # 中间 values 不得成为返回值
        yield ("values", self._final)


async def test_helper_returns_final_state_and_relays_text_blocks():
    collected = []
    loop = FakeLoop(
        {"done": True},
        [
            _Chunk([{"type": "text", "text": "缺口在"}]),
            _Chunk([{"type": "tool_call_chunk", "args": '{"markdown": "...'}]),  # 不进预览
            _Chunk([{"type": "text", "text": "应用层"}]),
        ],
    )
    result = await ainvoke_agent_with_deltas(
        loop,
        {"messages": []},
        context=None,
        config=None,
        channel="supervisor",
        delta_sink=lambda channel, text: collected.append((channel, text)),
    )
    assert result == {"done": True}  # 最后一个 values == ainvoke 返回值
    assert collected == [("supervisor", "缺口在"), ("supervisor", "应用层")]


async def test_helper_passthrough_to_ainvoke_without_sink():
    loop = FakeLoop({"done": True}, [])
    result = await ainvoke_agent_with_deltas(
        loop, {"messages": ["x"]}, context=None, config=None, channel="writer", delta_sink=None
    )
    assert result == {"done": True}
    assert loop.ainvoke_calls, "delta_sink=None 必须走原 ainvoke 路径"


async def test_helper_survives_malformed_message_chunks():
    class BadChunk:  # 没有 content_blocks 属性
        pass

    collected = []
    loop = FakeLoop({"done": True}, [BadChunk(), _Chunk([{"type": "text", "text": "ok"}])])
    result = await ainvoke_agent_with_deltas(
        loop,
        {},
        context=None,
        config=None,
        channel="supervisor",
        delta_sink=lambda c, t: collected.append((c, t)),
    )
    assert result == {"done": True}
    assert collected == [("supervisor", "ok")]
