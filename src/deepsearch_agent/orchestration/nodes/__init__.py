"""LangGraph 普通节点，按职责拆分并统一导出。"""

from deepsearch_agent.llm import ainvoke_structured
from deepsearch_agent.orchestration.nodes.clarify import clarify as _clarify
from deepsearch_agent.orchestration.nodes.quick_answer import quick_answer
from deepsearch_agent.orchestration.nodes.reflection import reflection as _reflection
from deepsearch_agent.orchestration.nodes.render import render_final_report_node
from deepsearch_agent.orchestration.nodes.router import router as _router


async def router(state, llm):
    return await _router(state, llm, invoke_structured=ainvoke_structured)


async def clarify(state, llm):
    return await _clarify(state, llm, invoke_structured=ainvoke_structured)


async def reflection(state, llm):
    return await _reflection(state, llm, invoke_structured=ainvoke_structured)


__all__ = [
    "ainvoke_structured",
    "clarify",
    "quick_answer",
    "reflection",
    "render_final_report_node",
    "router",
]
