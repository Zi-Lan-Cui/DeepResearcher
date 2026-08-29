"""Writer 的标准工具注册表。"""

import json

from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from deepsearch_agent.agents.writer.state import ValidatedDraft, WriterRuntimeContext
from deepsearch_agent.reporting.validation import extract_cite_ids, validate_and_bind


class ReadEvidence(BaseModel):
    """读取目录中 Evidence 的完整内容，以便引用其可验证细节。"""

    evidence_ids: list[str] = Field(
        min_length=1,
        max_length=30,
        description="要读取的 evidence_id 列表；每轮实际返回条数受批量上限约束，"
        "未轮到的 id 会在 not_read_ids 中列出，引用前需再次调用补齐。",
    )
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
        """返回指定 Evidence 的完整内容，并将其加入本次 Writer 的已读工作集。

        每轮最多读取 read_batch_size 条；未轮到读取的 id 会在 not_read_ids 中
        显式列出——引用前必须先把它们读完。
        """
        del reason
        context = runtime.context
        requested = list(dict.fromkeys(evidence_ids))
        # 静默截断曾让模型误以为"请求即已读"，引用被截掉的证据后无限循环烧尽
        # 轮次（run-e10d1229 事故）。截断现在必须显式回传给模型。
        unknown = [item for item in requested if item not in context.evidence_by_id]
        already_read = [
            item for item in requested if item in context.evidence_by_id
            and item in context.read_evidence_ids
        ]
        fresh = [
            item for item in requested if item in context.evidence_by_id
            and item not in context.read_evidence_ids
        ]
        capacity = min(context.read_batch_size, len(context.evidence_by_id) - len(context.read_evidence_ids))
        ids = fresh[: max(0, capacity)]
        not_read = fresh[max(0, capacity):]
        readable = [context.evidence_by_id[item] for item in ids]
        context.read_evidence_ids.update(ids)
        payload: dict[str, object] = {
            "evidence": [item.model_dump() for item in readable],
            "read_count": len(context.read_evidence_ids),
            "unknown_ids": unknown,
        }
        hint_parts: list[str] = []
        if not_read:
            payload["not_read_ids"] = not_read
            hint_parts.append(
                f"本轮配额已满，{len(not_read)} 条尚未读取：{', '.join(not_read)}——"
                "引用它们之前必须先再次调用 ReadEvidence 补齐。"
            )
        if already_read:
            payload["already_read_ids"] = already_read
        if unknown:
            hint_parts.append(
                f"以下 id 不在证据目录中（可能是编造的 id），不得引用：{', '.join(unknown)}。"
            )
        if hint_parts:
            payload["hint"] = "；".join(hint_parts)
        context.emit(
            "writer_evidence_read",
            {
                "requested_ids": evidence_ids,
                "read_ids": ids,
                "unknown_ids": unknown,
                "truncated_ids": not_read,
            },
        )
        return json.dumps(payload, ensure_ascii=False)

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
