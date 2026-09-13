"""请求路由节点。"""

import json

from langchain_core.messages import HumanMessage, SystemMessage

from deepresearcher.config import get_settings, language_directive
from deepresearcher.context.runtime import get_runtime_environment
from deepresearcher.llm import ainvoke_structured
from deepresearcher.prompts import load_prompt
from deepresearcher.schemas import RouteDecision, RunLifecycle


async def router(state, llm, *, invoke_structured=ainvoke_structured):
    query = state["query"].strip()
    try:
        result = await invoke_structured(
            llm,
            RouteDecision,
            [
                SystemMessage(
                    content=(
                        load_prompt("router")
                        + "\n"
                        + language_directive(get_settings().agent.output_language)
                    )
                ),
                HumanMessage(
                    content=(
                        "【运行时环境】\n"
                        + json.dumps(get_runtime_environment().payload(), ensure_ascii=False)
                        + f"\n【用户问题】\n{query}"
                    )
                ),
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
