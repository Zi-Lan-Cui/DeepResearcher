"""即时回答节点。"""

import json

from langchain_core.messages import HumanMessage, SystemMessage

from deepresearcher.context.runtime import get_runtime_environment
from deepresearcher.orchestration.nodes.common import content_text
from deepresearcher.prompts import load_prompt
from deepresearcher.schemas import ResearchProgress, RunLifecycle


async def quick_answer(state, llm):
    query = state["query"].strip()
    answer = content_text(
        await llm.ainvoke_text(
            [
                SystemMessage(content=load_prompt("quick_answer")),
                HumanMessage(
                    content=(
                        "【运行时环境】\n"
                        + json.dumps(get_runtime_environment().payload(), ensure_ascii=False)
                        + f"\n【用户问题】\n{query}"
                    )
                ),
            ]
        )
    ).strip()
    return {
        "clarified_query": query,
        "draft_answer": answer,
        "answer_mode": "quick_answer",
        "run": RunLifecycle(phase="writing"),
        "research": ResearchProgress(
            status="completed", generation_mode="full", is_sufficient=True
        ),
    }
