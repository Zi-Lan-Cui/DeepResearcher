import asyncio

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import PrivateAttr

uvloop = pytest.importorskip("uvloop")


class _WrapperSmokeModel(BaseChatModel):
    _calls: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "wrapper-smoke"

    def bind_tools(self, *_args, **_kwargs):
        return self

    async def _agenerate(self, messages, **_kwargs):
        self._calls += 1
        if any(isinstance(message, ToolMessage) for message in messages):
            response = AIMessage(content="done")
        else:
            response = AIMessage(
                content="",
                tool_calls=[{"name": "LoopSmokeEcho", "args": {"value": "ok"}, "id": "call-1"}],
            )
        return ChatResult(generations=[ChatGeneration(message=response)])

    def _generate(self, *_args, **_kwargs):
        raise NotImplementedError


@tool("LoopSmokeEcho")
async def _echo(value: str) -> str:
    """Return the supplied value."""
    return value


class _ModelWrapper(AgentMiddleware):
    async def awrap_model_call(self, request, handler):
        return await handler(request)


class _ToolWrapper(AgentMiddleware):
    async def awrap_tool_call(self, request, handler):
        return await handler(request)


def test_uvloop_completes_langchain_model_and_tool_wrappers():
    async def run_agent():
        assert isinstance(asyncio.get_running_loop(), uvloop.Loop)
        return await asyncio.wait_for(
            create_agent(
                _WrapperSmokeModel(),
                [_echo],
                middleware=[_ModelWrapper(), _ToolWrapper()],
            ).ainvoke({"messages": [{"role": "user", "content": "go"}]}),
            timeout=3,
        )

    result = asyncio.run(run_agent())

    assert result["messages"][-1].content == "done"
