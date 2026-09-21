"""LangGraph 普通节点，按职责拆分并统一导出。"""

# ainvoke_structured 在此转交是节点测试的 monkeypatch 接缝
# (tests patch deepresearcher.nodes.ainvoke_structured)——非本包公共出口,不进 __all__。
from deepresearcher.llm import ainvoke_structured
from deepresearcher.nodes.quick_answer import quick_answer
from deepresearcher.nodes.render import render_final_report_node
from deepresearcher.nodes.reviewer import reviewer as _reviewer
from deepresearcher.nodes.router import router as _router


async def router(state, llm, *, agent_config):
    return await _router(
        state, llm, agent_config=agent_config, invoke_structured=ainvoke_structured
    )


async def reviewer(state, llm, *, agent_config):
    return await _reviewer(
        state, llm, agent_config=agent_config, invoke_structured=ainvoke_structured
    )


__all__ = [
    "quick_answer",
    "reviewer",
    "render_final_report_node",
    "router",
]
