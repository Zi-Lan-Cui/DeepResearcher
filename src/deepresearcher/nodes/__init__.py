"""LangGraph 普通节点，按职责拆分并统一导出。

与 agents/ 的切分判据:一次模型调用出结论的住这里;多轮工具循环的住 agents/。
"""

# 结构化调用的唯一测试接缝:包装层在调用期读本模块全局 ainvoke_structured
# 显式传给内层(内层无 def 期默认值,不提供第二个注入点)。测试只 patch
# deepresearcher.nodes.ainvoke_structured——非本包公共出口,不进 __all__。
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
