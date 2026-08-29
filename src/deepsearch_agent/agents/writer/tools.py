"""Writer 的标准工具注册表。"""

import json

from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from deepsearch_agent.agents.writer.state import ValidatedDraft, WriterRuntimeContext
from deepsearch_agent.evidence.models import Evidence
from deepsearch_agent.reporting.validation import extract_cite_ids, validate_and_bind


class ReadEvidence(BaseModel):
    """读取目录中 Evidence 的完整内容，以便引用其可验证细节。"""

    evidence_ids: list[str] = Field(min_length=1, max_length=5)
    reason: str = Field(min_length=1)


class CompleteReport(BaseModel):
    """提交一篇带 Evidence 引用标记的 Markdown 草稿。"""

    selected_evidence_ids: list[str] = Field(min_length=1)
    markdown: str = Field(min_length=1)


def build_writer_tools():
    @tool("ReadEvidence", args_schema=ReadEvidence)
    def read_evidence(
        evidence_ids: list[str],
        reason: str,
        runtime: ToolRuntime[WriterRuntimeContext],
    ) -> str:
        """返回指定 Evidence 的完整内容，并将其加入本次 Writer 的已读工作集。"""
        del reason
        context = runtime.context
        remaining = max(0, len(context.evidence_by_id) - len(context.read_evidence_ids))
        ids = list(dict.fromkeys(evidence_ids))[: min(context.read_batch_size, remaining)]
        unknown = [item for item in ids if item not in context.evidence_by_id]
        readable: list[Evidence] = []
        for evidence_id in ids:
            evidence = context.evidence_by_id.get(evidence_id)
            if evidence is not None:
                context.read_evidence_ids.add(evidence_id)
                readable.append(evidence)
        context.emit(
            "writer_evidence_read",
            {
                "requested_ids": evidence_ids,
                "read_ids": ids,
                "unknown_ids": unknown,
            },
        )
        return json.dumps(
            {
                "evidence": [item.model_dump() for item in readable],
                "unknown_ids": unknown,
                "read_count": len(context.read_evidence_ids),
            },
            ensure_ascii=False,
        )

    @tool("CompleteReport", args_schema=CompleteReport)
    def complete_report(
        selected_evidence_ids: list[str],
        markdown: str,
        runtime: ToolRuntime[WriterRuntimeContext],
    ) -> str:
        """校验并提交报告草稿；校验失败时返回可供下一轮修正的错误。"""
        context = runtime.context
        context.last_markdown = markdown
        try:
            if len(selected_evidence_ids) > context.max_selected_evidence:
                # 报错必须自带修复指令：曾有无信息量的拒绝文案导致模型在最后一轮
                # 去重读证据、预算耗尽（53 条证据选超 24 上限事故）。
                raise ValueError(
                    f"已选 {len(selected_evidence_ids)} 条 Evidence，超过上限 "
                    f"{context.max_selected_evidence} 条。不要重新读取证据——只需把 "
                    "selected_evidence_ids 缩减为正文真正依赖的最核心若干条"
                    "（与正文 [[cite:...]] 标记一致），立即重新调用 CompleteReport。"
                )
            if len(markdown) > context.max_markdown_chars:
                raise ValueError(
                    f"Markdown 共 {len(markdown)} 字符，超过上限 {context.max_markdown_chars} 字符。"
                    "请压缩正文（保留全部 [[cite:...]] 标记与结论），立即重新调用 CompleteReport。"
                )
            cited_ids = extract_cite_ids(markdown)
            selected = list(
                dict.fromkeys(
                    item for item in selected_evidence_ids if item in context.read_evidence_ids
                )
            )
            selected.extend(
                item
                for item in cited_ids
                if item in context.read_evidence_ids and item not in selected
            )
            body, bindings, citations = validate_and_bind(
                markdown,
                {
                    item: context.evidence_by_id[item]
                    for item in context.read_evidence_ids
                    if item in context.evidence_by_id
                },
            )
            if not selected:
                raise ValueError("Writer 没有声明或实际引用任何已读取 Evidence。")
            context.validated_draft = ValidatedDraft(body, bindings, citations, selected)
        except ValueError as exc:
            context.last_error = str(exc)
            context.emit(
                "writer_citation_validation_failed",
                {"error": str(exc), "markdown": markdown},
            )
            return (
                f"提交校验失败：{exc}。"
                "只能引用已经由 ReadEvidence 返回的 evidence_id；修正 Markdown 后再次调用 CompleteReport。"
            )
        return "报告草稿已通过本地引用校验。请停止调用工具并结束回复。"

    return [read_evidence, complete_report]
