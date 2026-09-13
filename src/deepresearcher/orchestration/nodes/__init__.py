"""LangGraph 普通节点，按职责拆分并统一导出。"""

from deepresearcher.llm import ainvoke_structured
from deepresearcher.orchestration.nodes.quick_answer import quick_answer
from deepresearcher.orchestration.nodes.reflection import reflection as _reflection
from deepresearcher.orchestration.nodes.render import render_final_report_node
from deepresearcher.orchestration.nodes.router import router as _router


async def router(state, llm):
    return await _router(state, llm, invoke_structured=ainvoke_structured)


async def reflection(state, llm):
    return await _reflection(state, llm, invoke_structured=ainvoke_structured)


__all__ = [
    "ainvoke_structured",
    "quick_answer",
    "reflection",
    "render_final_report_node",
    "router",
]
