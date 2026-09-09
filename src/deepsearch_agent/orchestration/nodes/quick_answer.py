"""即时回答节点。"""

from langchain_core.messages import HumanMessage, SystemMessage

from deepsearch_agent.orchestration.nodes.common import content_text
from deepsearch_agent.prompts import load_prompt
from deepsearch_agent.schemas import ResearchProgress, RunLifecycle


async def quick_answer(state, llm):
    query = state["query"].strip()
    answer = content_text(
        await llm.ainvoke_text(
            [
                SystemMessage(content=load_prompt("quick_answer")),
                HumanMessage(content=query),
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
