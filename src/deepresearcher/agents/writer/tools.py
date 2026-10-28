"""Writer 的标准工具注册表。"""

from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command

from deepresearcher.agents.writer.state import (
    ValidatedDraft,
    WriterContext,
    evidence_detail_card,
)
from deepresearcher.reporting.validation import extract_cite_ids, validate_and_bind
from deepresearcher.schemas import ReadEvidence, ReportComplete, format_tool_receipt
from deepresearcher.schemas.limits import READ_EVIDENCE_REQUEST_WINDOW_IDS


def _writer_command(
    runtime: ToolRuntime[WriterContext],
    content: str,
    *,
    tool_name: str,
    channel_update: dict[str, Any],
) -> Command:
    """把回执文本 content 与通道增量合并进同一个 Command。"""
    return Command(
        update={
            "messages": [ToolMessage(content, tool_call_id=runtime.tool_call_id, name=tool_name)],
            **channel_update,
        }
    )


def build_writer_tools(*, turn_budget: int, read_batch: int):
    """组装 Writer 工具；机制写在工具描述里，随运行配置动态生成。"""

    @tool(
        "ReadEvidence",
        args_schema=ReadEvidence,
        description=(
            "获取目录中 Evidence 的原文与引用所需来源信息，并加入已读集。"
            f"用法：先从目录选定要引用的条目，用一次调用批量读取全部"
            f"（一次可请求至多 {READ_EVIDENCE_REQUEST_WINDOW_IDS} 条，每轮交付 {read_batch} 条，"
            "未交付的会列在 not_read_ids 中，下一轮补齐；不要按个位数小口读取）；"
            f"工具轮次预算约 {turn_budget} 轮，读取应占 1-2 轮，其余留给写作与修正。"
            "只有本工具返回过的 evidence_id 才能引用。"
            "若返回含 not_read_ids/unknown_ids/hint，必须先按提示处理。"
        ),
    )
    async def read_evidence(
        evidence_ids: list[str],
        reason: str,
        runtime: ToolRuntime[WriterContext],
    ) -> Command:
        # 模型可见契约在上方 description=(args_schema 在场时 docstring 不进 prompt)。
        del reason
        context = runtime.context
        requested = list(dict.fromkeys(evidence_ids))
        # 截断必须显式回传给模型：静默截断会让模型误以为"请求即已读",
        # 引用被截掉的证据后反复循环直到耗尽轮次。
        unknown = [item for item in requested if item not in context.evidence_by_id]
        already_read = [
            item
            for item in requested
            if item in context.evidence_by_id and item in context.read_evidence_ids
        ]
        fresh = [
            item
            for item in requested
            if item in context.evidence_by_id and item not in context.read_evidence_ids
        ]
        capacity = min(
            context.read_batch_size, len(context.evidence_by_id) - len(context.read_evidence_ids)
        )
        ids = fresh[: max(0, capacity)]
        not_read = fresh[max(0, capacity) :]
        readable = [context.evidence_by_id[item] for item in ids]
        context.read_evidence_ids.update(ids)
        payload: dict[str, object] = {
            "evidence": [evidence_detail_card(item) for item in readable],
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
        return _writer_command(
            runtime,
            format_tool_receipt(payload),
            tool_name="ReadEvidence",
            channel_update={"read_evidence_ids": list(ids)},
        )

    @tool(
        "ReportComplete",
        args_schema=ReportComplete,
        description=(
            "唯一的结束信号：写完草稿后必须经本工具提交，提交前不要直接输出报告正文。"
            "selected_evidence_ids 数量不设上限，但必须全部来自 ReadEvidence 的返回、"
            "且只列正文真正依赖的证据；markdown 为含 [[cite:evidence_id]] 标记的完整正文。"
            "若返回校验失败，严格按错误消息修正后立即重新提交"
            "——修正通常无需读取新证据，不要重复调用 ReadEvidence。"
        ),
    )
    async def complete_report(
        title: str,
        selected_evidence_ids: list[str],
        markdown: str,
        runtime: ToolRuntime[WriterContext],
    ) -> Command:
        # 模型可见契约在上方 description=；校验失败文本即下一轮修正指令。
        context = runtime.context
        context.last_markdown = markdown
        try:
            title_text = title.strip().strip("#").strip()
            if not title_text or any(ch in title_text for ch in "#[]*"):
                raise ValueError(
                    "title 必须是纯文本标题（不含 Markdown 标记）；重新拟一个标题再次提交。"
                )
            if len(markdown) > context.max_markdown_chars:
                raise ValueError(
                    f"Markdown 共 {len(markdown)} 字符，超过上限 {context.max_markdown_chars} 字符。"
                    "请压缩正文（保留全部 [[cite:...]] 标记与结论），立即重新调用 ReportComplete。"
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
            context.validated_draft = ValidatedDraft(
                body, bindings, citations, selected, title=title_text
            )
        except ValueError as exc:
            context.last_error = str(exc)
            context.emit(
                "writer_citation_validation_failed",
                {"error": str(exc), "markdown": markdown},
            )
            return _writer_command(
                runtime,
                f"提交校验失败：{exc}。"
                "只能引用已经由 ReadEvidence 返回的 evidence_id；修正 Markdown 后再次调用 ReportComplete。",
                tool_name="ReportComplete",
                channel_update={
                    "last_error": context.last_error,
                    "last_markdown": markdown,
                },
            )
        return _writer_command(
            runtime,
            "报告草稿已通过本地引用校验。请停止调用工具并结束回复。",
            tool_name="ReportComplete",
            channel_update={
                "validated_draft": context.validated_draft,
                "last_error": "",
                "last_markdown": markdown,
            },
        )

    return [read_evidence, complete_report]
