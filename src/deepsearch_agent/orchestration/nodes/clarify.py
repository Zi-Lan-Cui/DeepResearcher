"""研究意图澄清节点。"""

from langchain_core.messages import HumanMessage, SystemMessage

from deepsearch_agent.config import get_settings, language_directive
from deepsearch_agent.llm import ainvoke_structured
from deepsearch_agent.schemas import ClarificationDecision, RunLifecycle


async def clarify(state, llm, *, invoke_structured=ainvoke_structured):
    """确认研究意图；绝不将开放问题压缩成定义或检索句。"""
    query = state["query"].strip()
    decision = await invoke_structured(
        llm,
        ClarificationDecision,
        [
            SystemMessage(
                content=(
                    "你是研究意图澄清器。用户原问题会原样保留并展示给用户；你不能改写、缩窄、"
                    "替换它，也不能把‘如何看待/意义/影响/评价’等开放分析题改成定义题或一句话任务。"
                    "你的工作只是提取研究意图和应覆盖的角度。"
                    "只有缺少一个会实质改变答案方向、且无法合理并列处理的必要选择时，"
                    "才设置 needs_user_input=true 并提出一个简短问题。"
                    "范围宽、带价值判断、需要多维分析，或术语可先给工作定义的题目，都应直接继续研究。"
                    "如果可以继续，needs_user_input=false；intent_summary 写用户真正想了解什么，"
                    "research_focus 列出不超过四个应覆盖的角度。\n"
                    + language_directive(get_settings().agent.output_language)
                )
            ),
            HumanMessage(content=query),
        ],
    )
    brief = (
        "；".join(
            [
                decision.intent_summary.strip(),
                *(item.strip() for item in decision.research_focus if item.strip()),
            ]
        ).strip("；")
        or query
    )
    if decision.needs_user_input:
        question = decision.clarification_question.strip() or "你希望优先从哪一个角度展开？"
        return {
            "clarified_query": query,
            "research_brief": brief,
            "clarification_question": question,
            "answer_mode": "clarification_needed",
            "run": RunLifecycle(phase="clarification"),
            "report": "\n".join(
                ["# 需要澄清", "", "## 原问题", query, "", "## 需要确认", question]
            ),
        }
    return {
        "clarified_query": query,
        "research_brief": brief,
        "run": RunLifecycle(phase="researching"),
    }
