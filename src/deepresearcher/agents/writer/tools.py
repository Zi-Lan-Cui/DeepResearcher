"""Writer 的标准工具注册表。"""

from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from deepresearcher.agents.writer.state import (
    ValidatedDraft,
    WriterLoopContext,
    evidence_detail_card,
)
from deepresearcher.reporting.validation import extract_cite_ids, validate_and_bind
from deepresearcher.schemas import format_tool_receipt

# 一次可请求的窗口（防失控的宽松值）；每轮实际交付量由 writer_read_batch_size
# 决定，差额走 not_read_ids 显式排队。引用总条数刻意不设上限：聚焦度是写作
# 质量问题，交给提示词引导、已读校验与审阅把关；硬上限只会催生为凑数而选。
REQUEST_WINDOW_IDS = 50


class ReadEvidence(BaseModel):
    """读取目录中 Evidence 的原文与紧凑来源信息，以便引用可验证细节。"""

    evidence_ids: list[str] = Field(
        min_length=1,
        max_length=REQUEST_WINDOW_IDS,
        description="要读取的 evidence_id 列表；引用前必须先读取。",
    )
    reason: str = Field(min_length=1)


class CompleteReport(BaseModel):
    """提交一篇带 Evidence 引用标记的 Markdown 草稿。"""

    selected_evidence_ids: list[str] = Field(min_length=1)
    markdown: str = Field(min_length=1)


def build_writer_tools(*, turn_budget: int, read_batch: int):
    """组装 Writer 工具；机制写在工具描述里，随运行配置动态生成。

    引用总条数不设上限；约束只剩三条：cite 必须已读（本工具集）、
    每轮交付 read_batch 条（差额 not_read_ids 排队）、正文有字符上限。
    聚焦度是写作质量问题，交给系统提示词的引导与审阅把关。
    """

    @tool(
        "ReadEvidence",
        args_schema=ReadEvidence,
        description=(
            "获取目录中 Evidence 的原文与引用所需来源信息，并加入已读集。"
            f"用法：先从目录选定要引用的条目，用一次调用批量读取全部"
            f"（一次可请求至多 {REQUEST_WINDOW_IDS} 条，每轮交付 {read_batch} 条，"
            "未交付的会列在 not_read_ids 中，下一轮补齐；不要按个位数小口读取）；"
            f"工具轮次预算约 {turn_budget} 轮，读取应占 1-2 轮，其余留给写作与修正。"
            "只有本工具返回过的 evidence_id 才能引用。"
            "若返回含 not_read_ids/unknown_ids/hint，必须先按提示处理。"
        ),
    )
    async def read_evidence(
        evidence_ids: list[str],
        reason: str,
        runtime: ToolRuntime[WriterLoopContext],
    ) -> str:
        # 模型可见契约在上方 description=(args_schema 在场时 docstring 不进 prompt)。
        del reason
        context = runtime.context
        requested = list(dict.fromkeys(evidence_ids))
        # 静默截断曾让模型误以为"请求即已读"，引用被截掉的证据后无限循环烧尽
        # 轮次（引用错位事故）。截断必须显式回传给模型。
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
        return format_tool_receipt(payload)

    @tool(
        "CompleteReport",
        args_schema=CompleteReport,
        description=(
            "唯一的结束信号：写完草稿后必须经本工具提交，提交前不要直接输出报告正文。"
            "selected_evidence_ids 数量不设上限，但必须全部来自 ReadEvidence 的返回、"
            "且只列正文真正依赖的证据；markdown 为含 [[cite:evidence_id]] 标记的完整正文。"
            "若返回校验失败，严格按错误消息修正后立即重新提交"
            "——修正通常无需读取新证据，不要重复调用 ReadEvidence。"
        ),
    )
    async def complete_report(
        selected_evidence_ids: list[str],
        markdown: str,
        runtime: ToolRuntime[WriterLoopContext],
    ) -> str:
        # 模型可见契约在上方 description=;校验失败文本即下一轮修正指令。
        context = runtime.context
        context.last_markdown = markdown
        try:
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
