"""LangGraph 普通节点，按职责拆分并统一导出。"""

from deepresearcher.llm import ainvoke_structured
from deepresearcher.nodes.quick_answer import quick_answer
from deepresearcher.nodes.render import render_final_report_node
from deepresearcher.nodes.reviewer import reviewer as _reviewer
from deepresearcher.nodes.router import router as _router


async def router(state, llm):
    return await _router(state, llm, invoke_structured=ainvoke_structured)


async def reviewer(state, llm):
    return await _reviewer(state, llm, invoke_structured=ainvoke_structured)


__all__ = [
    "ainvoke_structured",
    "quick_answer",
    "reviewer",
    "render_final_report_node",
    "router",
]
