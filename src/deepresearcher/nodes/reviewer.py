"""报告语义审阅节点。"""

from langchain_core.messages import HumanMessage, SystemMessage
from openai import LengthFinishReasonError

from deepresearcher.config import AgentConfig
from deepresearcher.observability.logging_config import get_logger
from deepresearcher.prompts import (
    get_runtime_environment,
    language_directive,
    load_prompt,
    render_data_section,
)
from deepresearcher.schemas import (
    Citation,
    ParagraphBinding,
    ReviewDecision,
    ReviewProgress,
)
from deepresearcher.schemas.sources import source_domain
from deepresearcher.state import section

_logger = get_logger("deepresearcher.nodes.reviewer")


def reviewer_evidence_card(citation: Citation) -> dict[str, object]:
    """Reviewer 视图：保留核验原文，并补充紧凑的支撑强度与来源背景。"""
    return {
        "id": citation.id,
        "fact": citation.claim,
        "quote": citation.quote,
        "support": citation.support,
        "source_title": citation.title,
        "source_domain": source_domain(citation.url),
        "source_type": citation.source_profile.source_type,
        "authority_tier": citation.source_profile.authority_tier,
        **({"published_at": citation.published_at} if citation.published_at else {}),
    }


async def reviewer(state, llm, *, agent_config: AgentConfig, invoke_structured):
    """审阅草稿并把结论交还 Supervisor，不自行调度 Writer 或研究员。"""
    review = section(state, "review", ReviewProgress)
    attempt = review.attempts + 1
    bindings = list(state.get("paragraph_bindings", []))
    citations = {item.id: item for item in list(state.get("citations", []))}
    if not all(isinstance(item, ParagraphBinding) for item in bindings):
        raise TypeError("审阅输入通道含非 ParagraphBinding 对象。")
    if not all(isinstance(item, Citation) for item in citations.values()):
        raise TypeError("审阅输入通道含非 Citation 对象。")
    if not bindings:
        raise ValueError("报告没有可审阅的论断绑定。")

    review_items = [
        {
            "paragraph": item.text,
            "kind": item.kind,
            "evidence": [
                reviewer_evidence_card(citations[source_id])
                for source_id in item.evidence_ids
                if source_id in citations
            ],
        }
        for item in bindings
    ]
    messages = [
        SystemMessage(
            content=(
                load_prompt("reviewer") + "\n" + language_directive(agent_config.output_language)
            )
        ),
        HumanMessage(
            content=(
                render_data_section("运行时环境", get_runtime_environment().payload())
                + "\n\n---\n\n"
                + render_data_section(
                    "审阅输入",
                    {
                        "研究问题": state.get("clarified_query", state.get("query", "")),
                        "待审阅段落": review_items,
                    },
                )
            )
        ),
    ]

    try:
        decision = await invoke_structured(llm, ReviewDecision, messages)
    except LengthFinishReasonError:
        # 输出超长是模型行为，重发同一请求大概率同样截断：写成审阅失败，
        # 由路由按「未经审阅」交付已写好的正文，不判成研究内容缺口。
        _logger.warning("reviewer_failed reason=output_length_limit")
        return {
            "review": ReviewProgress(
                status="failed",
                attempts=attempt,
                failure_reason="审阅输出超出长度上限，未返回结论",
            )
        }

    status = (
        "rejected" if any(issue.severity == "fatal" for issue in decision.issues) else "approved"
    )
    return {
        "review": ReviewProgress(
            status=status,
            attempts=attempt,
            feedback=decision.feedback,
            gaps=decision.gaps,
            issues=decision.issues,
        ),
    }
