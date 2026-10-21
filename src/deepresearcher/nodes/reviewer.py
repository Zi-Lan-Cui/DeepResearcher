"""报告语义审阅节点。"""

import asyncio

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from openai import ContentFilterFinishReasonError, LengthFinishReasonError
from pydantic import ValidationError

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

# 只重试同一请求重发后结果可能不同的失败：网关内容审查有随机抖动，content_filter 曾
# 实际发生，JSON 解析与 schema 校验失败同理。传输层超时与断连由 openai SDK 重试处理。
_RETRYABLE_REVIEW_ERRORS = (
    ContentFilterFinishReasonError,
    OutputParserException,
    ValidationError,
)


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
    # 审阅只看问题与段落引用本身；任务书等上游环节不随附，独立评审。
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

    decision = None
    runaway_attempts = 0
    for retry_index in range(agent_config.reviewer_retry_attempts + 1):
        try:
            decision = await invoke_structured(llm, ReviewDecision, messages)
            break
        except asyncio.CancelledError:
            raise
        except LengthFinishReasonError:
            # 输出超长是模型行为，重试可能同样超长；重试耗尽仍无有效输出则转弃权。
            runaway_attempts += 1
            _logger.warning(
                "reviewer_output_runaway attempt=%d/%d",
                retry_index + 1,
                agent_config.reviewer_retry_attempts + 1,
            )
        except _RETRYABLE_REVIEW_ERRORS as exc:
            if retry_index >= agent_config.reviewer_retry_attempts:
                raise
            _logger.warning(
                "reviewer_retry attempt=%d/%d error=%s",
                retry_index + 1,
                agent_config.reviewer_retry_attempts,
                type(exc).__name__,
            )
            await asyncio.sleep(agent_config.reviewer_retry_initial_seconds * (retry_index + 1))
    if decision is None:
        _logger.warning(
            "reviewer_abstained runaway_attempts=%d; 审阅未给出结论，报告按原样返回",
            runaway_attempts,
        )
        return {"review": ReviewProgress(status="approved", attempts=attempt)}

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
