"""在 agent 调用点就地接力 token 预览。

背景（实测，非猜测）：嵌套 Pregel——例如 create_agent 编译出的图——在宿主图
节点内被 `.ainvoke` 调用时，它的 token 流对外层
`astream(stream_mode="messages")` **完全不可见**（外层收到 0 个事件；把模型
直接放进节点则有 102 个）。langgraph 的 messages 模式只在"模型调用发生在被
stream 的那张图内部"时才成立。因此逐字预览必须在真正驱动 agent loop 的地方
接力：把 ainvoke 换成 astream(["values","messages"])，values 收集等价最终
状态，messages 里的 text 块转发给注入的 delta_sink。

delta_sink 为 None（CLI/测试默认）时原样走 ainvoke，零行为变化。
"""

from __future__ import annotations

from typing import Any, Callable

#: (channel, text) —— channel 由调用方 agent 决定（supervisor/writer/…）。
TextDeltaSink = Callable[[str, str], None]


async def ainvoke_agent_with_deltas(
    agent_loop: Any,
    state: dict,
    *,
    context: Any,
    config: dict | None,
    channel: str,
    delta_sink: TextDeltaSink | None,
) -> dict:
    """与 agent_loop.ainvoke(state, context=..., config=...) 语义等价。

    异常（含 GraphRecursionError）原样上抛，调用方的 except 分支不受影响；
    最后一个 values chunk 即 ainvoke 的返回值。
    """
    if delta_sink is None:
        return await agent_loop.ainvoke(state, context=context, config=config)
    final: dict = {}
    async for mode, chunk in agent_loop.astream(
        state, context=context, config=config, stream_mode=["values", "messages"]
    ):
        if mode == "values":
            if isinstance(chunk, dict):
                final = chunk
            continue
        if mode != "messages":
            continue
        try:
            message, _metadata = chunk
        except (TypeError, ValueError):
            continue
        # tool_call_chunk / 推理块 type 不是 text，天然不进预览。
        blocks = getattr(message, "content_blocks", None) or []
        text = "".join(
            str(block.get("text", ""))
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
        if text:
            delta_sink(channel, text)
    return final
