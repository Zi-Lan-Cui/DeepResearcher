"""报告语义审阅节点。"""

import asyncio

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from openai import ContentFilterFinishReasonError
from pydantic import ValidationError

from deepsearch_agent.config import get_settings, language_directive
from deepsearch_agent.llm import ainvoke_structured
from deepsearch_agent.observability.logger import get_logger
from deepsearch_agent.prompts import load_prompt
from deepsearch_agent.schemas import Citation, ParagraphBinding, ReflectionDecision, ReviewProgress
from deepsearch_agent.state import section

_logger = get_logger("deepsearch_agent.orchestration.reflection")

# 只重试"同一请求重发可能改天换日"的失败：网关内容审查有随机抖动
# （content_filter 事故实锤），JSON 解析/schema 校验失败同理。
# 传输层超时/断连已由 with_transport_retry 处理，不在这里重复。
_RETRYABLE_REVIEW_ERRORS = (
    ContentFilterFinishReasonError,
    OutputParserException,
    ValidationError,
)


async def reflection(state, llm, *, invoke_structured=ainvoke_structured):
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
                {
                    "id": source_id,
                    "fact": citations[source_id].claim,
                    "quote": citations[source_id].quote,
                }
                for source_id in item.evidence_ids
                if source_id in citations
            ],
        }
        for item in bindings
    ]
    messages = [
        SystemMessage(
            content=(
                load_prompt("reflection")
                + "\n"
                + language_directive(get_settings().agent.output_language)
            )
        ),
        HumanMessage(
            content=(
                f"研究问题：{state.get('clarified_query', state.get('query', ''))}\n"
                f"Supervisor 报告任务书：{state.get('report_brief', {})}\n"
                f"待审阅段落：{review_items}"
            )
        ),
    ]

    retry_config = get_settings().agent
    decision = None
    for retry_index in range(retry_config.reflection_retry_attempts + 1):
        try:
            decision = await invoke_structured(llm, ReflectionDecision, messages)
            break
        except asyncio.CancelledError:
            raise
        except _RETRYABLE_REVIEW_ERRORS as exc:
            if retry_index >= retry_config.reflection_retry_attempts:
                raise
            _logger.warning(
                "reflection_retry attempt=%d/%d error=%s",
                retry_index + 1,
                retry_config.reflection_retry_attempts,
                type(exc).__name__,
            )
            await asyncio.sleep(retry_config.reflection_retry_initial_seconds * (retry_index + 1))
    assert decision is not None  # 循环不变量：break 或 raise

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
