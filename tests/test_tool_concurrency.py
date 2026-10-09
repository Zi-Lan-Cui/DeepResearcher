"""串行批次守卫:违规工具批在模型输出后整批驳回、打回模型,不执行任何工具。"""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from deepresearcher.agents.middleware.serial_tools import SerialToolMiddleware

_SERIAL = {"ResearchComplete", "Revise"}
_middleware = SerialToolMiddleware(_SERIAL)


def _batch(names: list[str]) -> dict:
    """模拟一次模型输出的 state:末条 AIMessage 携带给定工具调用批。"""
    return {
        "messages": [
            AIMessage(
                id="ai-batch",
                content="",
                tool_calls=[
                    {"name": name, "args": {}, "id": f"call-{index}"}
                    for index, name in enumerate(names)
                ],
            )
        ]
    }


def test_lone_serial_batch_passes_through():
    assert asyncio.run(_middleware.aafter_model(_batch(["ResearchComplete"]), None)) is None


def test_parallel_only_batch_passes_through():
    assert (
        asyncio.run(
            _middleware.aafter_model(_batch(["AddEvidence", "ReadSources", "SearchSources"]), None)
        )
        is None
    )


def test_plain_text_output_passes_through():
    state = {"messages": [AIMessage(content="研究结束,交回。")]}
    assert asyncio.run(_middleware.aafter_model(state, None)) is None


def test_mixed_batch_rejected_wholesale():
    """[Delegate, Complete] 混批:整批驳回——删除发起消息、注入驳回说明、jump 回模型。"""
    origin = _batch(["ResearchDelegate", "ResearchComplete"])
    update = asyncio.run(_middleware.aafter_model(origin, None))
    assert update is not None
    assert update["jump_to"] == "model"
    injected = update["messages"]
    assert isinstance(injected[0], RemoveMessage)
    assert injected[0].id == origin["messages"][-1].id
    assert isinstance(injected[1], HumanMessage)
    assert "整批已驳回,没有任何工具执行" in injected[1].content


def test_all_serial_batch_rejected_wholesale():
    """[Revise, Complete] 两个串行同批:同样整批驳回,说明里点名全部违规工具。"""
    update = asyncio.run(_middleware.aafter_model(_batch(["Revise", "ResearchComplete"]), None))
    assert update is not None
    assert update["jump_to"] == "model"
    assert "ResearchComplete, Revise" in update["messages"][1].content
