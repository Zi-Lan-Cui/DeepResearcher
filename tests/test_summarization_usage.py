"""摘要调用的观测归属:callbacks 覆盖它,模型中间件预检不覆盖它。

结论若变化(升级 langchain 后传播行为改变),本测试即警报。
"""

import asyncio

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from deepresearcher.agents.middleware.factory import (
    ObservableSummarizationMiddleware,
    count_message_tokens,
)


class _SummarySmokeModel(BaseChatModel):
    _calls: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "summary-smoke"

    def bind_tools(self, *_args, **_kwargs):
        return self

    async def _agenerate(self, messages, **_kwargs):
        self._calls += 1
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="已摘要/完成"))])

    def _generate(self, *_args, **_kwargs):
        raise NotImplementedError


class _CountStarts(BaseCallbackHandler):
    def __init__(self):
        self.chat_model_starts = 0

    def on_chat_model_start(self, *_args, **_kwargs):
        self.chat_model_starts += 1


class _ProbeWrap(AgentMiddleware):
    """统计 awrap_model_call 包到的模型调用数(预算预检挂在这一层)。"""

    wrapped = 0

    async def awrap_model_call(self, request, handler):
        _ProbeWrap.wrapped += 1
        return await handler(request)


def test_summarization_call_is_recorded_but_bypasses_model_wrappers():
    async def run():
        model = _SummarySmokeModel()
        starts = _CountStarts()
        agent = create_agent(
            model,
            [],
            middleware=[
                ObservableSummarizationMiddleware(
                    agent_name="smoke",
                    emit=None,
                    trigger_tokens=100,
                    model=model,
                    trigger=("tokens", 100),
                    keep=("tokens", 50),
                    token_counter=count_message_tokens,
                ),
                _ProbeWrap(),
            ],
        )
        # 摘要需要"有历史可摘":单条消息会被 keep 窗口全部保留,不触发。
        history = []
        for i in range(6):
            history.append({"role": "user", "content": f"问题{i} " + "用于超过阈值的中文。" * 10})
            history.append({"role": "assistant", "content": f"回答{i} " + "用于超过阈值的中文。" * 10})
        result = await agent.ainvoke({"messages": history}, config={"callbacks": [starts]})
        return model, starts, result

    _ProbeWrap.wrapped = 0
    model, starts, result = asyncio.run(run())

    # 摘要确实发生:模型被调了不止一次(摘要 + 回合)。
    assert model._calls >= 2  # noqa: SLF001
    # 传播成立:astream 级 callbacks 覆盖了摘要调用,RunUsageCallback 能记到它。
    assert starts.chat_model_starts == model._calls  # noqa: SLF001
    # 预检不覆盖:摘要绕开 awrap_model_call(enforce_usage_budget 所在层),
    # 因此预算快耗尽时仍可能多花一次摘要调用——事后记账、下回合拦截。
    assert _ProbeWrap.wrapped < model._calls
    assert result["messages"][-1].content == "已摘要/完成"
