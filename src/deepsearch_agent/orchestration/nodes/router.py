"""请求路由节点。"""

from langchain_core.messages import HumanMessage, SystemMessage

from deepsearch_agent.config import get_settings, language_directive
from deepsearch_agent.llm import ainvoke_structured
from deepsearch_agent.schemas import RouteDecision, RunLifecycle


async def router(state, llm, *, invoke_structured=ainvoke_structured):
    query = state["query"].strip()
    try:
        result = await invoke_structured(
            llm,
            RouteDecision,
            [
                SystemMessage(
                    content=(
                        "你是研究请求路由器。明确、低风险、单一事实问题才走 quick_answer。"
                        "涉及多个对象、比较、影响力、推荐、历史、趋势、因果或需要来源核验的问题必须走 deep_research。"
                        "不确定时宁可选择 deep_research；不能把模型内部知识包装成研究结论。\n"
                        + language_directive(get_settings().agent.output_language)
                    )
                ),
                HumanMessage(content=query),
            ],
        )
        return {
            "route": "deep_research" if result.route == "clarify_needed" else result.route,
            "route_reason": result.reason,
            "run": RunLifecycle(phase="routing"),
        }
    except Exception:
        return {
            "route": "deep_research",
            "route_reason": "路由模型调用失败；为避免把未核验知识包装为答案，转入深度研究。",
            "run": RunLifecycle(phase="routing"),
        }
