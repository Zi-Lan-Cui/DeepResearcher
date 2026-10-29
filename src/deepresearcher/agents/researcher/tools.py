"""Researcher 的标准工具注册表"""

from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.types import Command

from deepresearcher.agents.researcher import services
from deepresearcher.agents.researcher.state import (
    ResearcherContext,
    _candidate_channel_entry,
    _document_channel_entry,
    working_set_snapshot,
)
from deepresearcher.agents.working_set import release_working_set, restore_working_set
from deepresearcher.schemas import (
    AddEvidence,
    DocumentReadTarget,
    EvidenceSubmission,
    GrepDocument,
    GrepWindowSegment,
    ListSearchResults,
    ReadDocument,
    ReadSources,
    ReadWorkingSet,
    ReleaseEvidence,
    RestoreEvidence,
    SearchSources,
    TaskComplete,
    format_tool_receipt,
)


def _channel_command(
    runtime: ToolRuntime[ResearcherContext],
    payload: dict[str, object],
    *,
    tool_name: str,
    channel_update: dict[str, Any],
) -> Command:
    """把回执(ToolMessage)与通道增量合并进同一个 Command。"""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    format_tool_receipt(payload),
                    tool_call_id=runtime.tool_call_id,
                    name=tool_name,
                )
            ],
            **channel_update,
        }
    )


def _tool_context(runtime: ToolRuntime[ResearcherContext]) -> dict[str, object]:
    """工具级事件上下文：任务级字段加本次调用的 tool_call_id。

    task_id 与 operation_id 在一次任务内恒定，缺 tool_call_id 时同一工具的多次调用
    同键，被事件存储判为重放而丢弃。tool_call_id 来自模型输出并随 checkpoint 持久化，
    重放同一回合时取值不变，不破坏去重语义。
    """
    context = runtime.context
    return {**context.event_context, "tool_call_id": runtime.tool_call_id}


def build_researcher_tools() -> list[BaseTool]:
    """创建一套绑定到 ResearcherContext 的任务级工具。"""

    @tool("SearchSources", args_schema=SearchSources)
    async def search_sources(
        reason: str,
        queries: list[str],
        runtime: ToolRuntime[ResearcherContext],
    ) -> Command:
        """发现当前研究任务的候选来源；不会自动读取网页。"""
        ctx = runtime.context
        candidate_keys_before = set(ctx.task_state.candidates)
        result = await services.search_sources(
            ctx.deps, ctx.task, ctx.task_state, _tool_context(runtime), queries, reason
        )
        new_candidates = {
            key: _candidate_channel_entry(value)
            for key, value in ctx.task_state.candidates.items()
            if key not in candidate_keys_before
        }
        return _channel_command(
            runtime,
            result,
            tool_name="SearchSources",
            channel_update={
                "queries": list(ctx.task_state.queries),
                "search_batches": dict(ctx.task_state.search_batches),
                "candidates": new_candidates,
                "failures": list(ctx.task_state.failures),
                "provider_exhausted": ctx.task_state.provider_exhausted,
            },
        )

    @tool("ReadSources", args_schema=ReadSources)
    async def read_sources(
        candidate_ids: list[str],
        reason: str,
        runtime: ToolRuntime[ResearcherContext],
    ) -> Command:
        """抓取模型选中的来源；短文内联，长文返回可按行读取的句柄。"""
        ctx = runtime.context
        document_keys_before = set(ctx.task_state.documents)
        result = await services.read_sources(
            ctx.deps, ctx.task, ctx.task_state, _tool_context(runtime), candidate_ids, reason
        )
        new_documents = {
            key: _document_channel_entry(value)
            for key, value in ctx.task_state.documents.items()
            if key not in document_keys_before
        }
        return _channel_command(
            runtime,
            result,
            tool_name="ReadSources",
            channel_update={
                "read_urls": list(ctx.task_state.read_urls),
                "source_refs": list(ctx.task_state.source_refs),
                "documents": new_documents,
                "selected_candidate_ids": sorted(ctx.task_state.selected_candidate_ids),
                "skipped": list(ctx.task_state.skipped),
                "failures": list(ctx.task_state.failures),
            },
        )

    @tool("ListSearchResults", args_schema=ListSearchResults)
    async def list_search_results(
        search_id: str,
        offset: int,
        limit: int,
        reason: str,
        runtime: ToolRuntime[ResearcherContext],
    ) -> str:
        """分页查看 SearchSources 已写入磁盘的完整候选目录。"""
        ctx = runtime.context
        result = await services.list_search_results(
            ctx.deps, ctx.task, ctx.task_state, search_id, offset, limit, reason
        )
        return format_tool_receipt(result)

    @tool("GrepDocument", args_schema=GrepDocument)
    async def grep_document(
        document_id: str,
        patterns: list[str],
        segments: list[GrepWindowSegment],
        context_lines: int,
        reason: str,
        runtime: ToolRuntime[ResearcherContext],
    ) -> str:
        """在当前任务已登记的长文中定位多个关键词；命中按文档行号归并，segments 一次取多段。"""
        ctx = runtime.context
        result = await services.grep_document(
            ctx.deps,
            ctx.task_state,
            document_id,
            list(patterns),
            [(item.offset, item.count) for item in segments],
            context_lines,
            reason,
        )
        return format_tool_receipt(result)

    @tool("ReadDocument", args_schema=ReadDocument)
    async def read_document(
        documents: list[DocumentReadTarget],
        reason: str,
        runtime: ToolRuntime[ResearcherContext],
    ) -> str:
        """按行号跨多份已登记文档批量读取窗口；读多份文档在 documents 里列全，程序逐文档展开。"""
        ctx = runtime.context
        result = await services.read_documents(
            ctx.deps,
            ctx.task_state,
            [
                (
                    item.document_id,
                    [(window.start_line, window.end_line) for window in item.ranges],
                )
                for item in documents
            ],
            reason,
        )
        return format_tool_receipt(result)

    @tool("AddEvidence", args_schema=AddEvidence)
    async def add_evidence(
        evidences: list[EvidenceSubmission],
        reason: str,
        runtime: ToolRuntime[ResearcherContext],
    ) -> Command:
        """批量提交 Evidence；按首尾片段在已登记原文中定位，程序逐字回填 quote 后写入。"""
        ctx = runtime.context
        task_state = ctx.task_state
        evidence_ids_before = {item.evidence_id for item in task_state.evidences}
        active_before = task_state.capture_active()
        result = await services.add_evidence(
            ctx.deps,
            ctx.task,
            task_state,
            _tool_context(runtime),
            [
                item.model_dump(mode="json") if isinstance(item, EvidenceSubmission) else dict(item)
                for item in evidences
            ],
            reason,
        )
        new_evidences = [
            item for item in task_state.evidences if item.evidence_id not in evidence_ids_before
        ]
        return _channel_command(
            runtime,
            result,
            tool_name="AddEvidence",
            channel_update={
                "evidences": new_evidences,
                "active_evidence_ids": task_state.active_diff_since(active_before),
            },
        )

    @tool("ReadWorkingSet", args_schema=ReadWorkingSet)
    async def read_working_set(
        reason: str,
        runtime: ToolRuntime[ResearcherContext],
    ) -> str:
        """查看当前任务工作集的轻量摘要。"""
        del reason
        return format_tool_receipt(working_set_snapshot(runtime.context.task_state))

    @tool("ReleaseEvidence", args_schema=ReleaseEvidence)
    async def release_evidence(
        evidence_ids: list[str],
        reason: str,
        runtime: ToolRuntime[ResearcherContext],
    ) -> Command:
        """从当前任务工作集释放 Evidence，但不删除全局档案。"""
        del reason
        task_state = runtime.context.task_state
        active_before = task_state.capture_active()
        result = release_working_set(
            task_state, evidence_ids, snapshot=lambda: working_set_snapshot(task_state)
        )
        return _channel_command(
            runtime,
            result,
            tool_name="ReleaseEvidence",
            channel_update={"active_evidence_ids": task_state.active_diff_since(active_before)},
        )

    @tool("RestoreEvidence", args_schema=RestoreEvidence)
    async def restore_evidence(
        evidence_ids: list[str],
        reason: str,
        runtime: ToolRuntime[ResearcherContext],
    ) -> Command:
        """从任务候选档案恢复 Evidence；不会超过活跃工作集上限。"""
        del reason
        task_state = runtime.context.task_state
        active_before = task_state.capture_active()
        result = restore_working_set(
            task_state, evidence_ids, snapshot=lambda: working_set_snapshot(task_state)
        )
        return _channel_command(
            runtime,
            result,
            tool_name="RestoreEvidence",
            channel_update={"active_evidence_ids": task_state.active_diff_since(active_before)},
        )

    # 不设 return_direct：被拒的提交也须留在循环内改正引用；接受后由 SubmittedExitMiddleware
    # 依 stop_reason 在下一跳退出，与 Supervisor、Writer 的出口方式一致。
    @tool("TaskComplete", args_schema=TaskComplete)
    async def complete_task(
        reason: str,
        selected_evidence_ids: list[str],
        conclusion: str,
        finding: str,
        runtime: ToolRuntime[ResearcherContext],
    ) -> Command:
        """提交当前任务的最终局部结果；被拒则按回执修正后重提。"""
        ctx = runtime.context
        task_state = ctx.task_state
        active_before = task_state.capture_active()
        result = services.complete_task(
            task_state,
            reason,
            selected_evidence_ids,
            conclusion,
            finding,
        )
        # 被拒时 stop_reason 与活跃集都不变，不落通道，只回一份纠错回执。
        if result.get("status") != "accepted":
            return _channel_command(runtime, result, tool_name="TaskComplete", channel_update={})
        return _channel_command(
            runtime,
            result,
            tool_name="TaskComplete",
            channel_update={
                "conclusion": task_state.conclusion,
                "finding": task_state.finding,
                "stop_reason": task_state.stop_reason,
                "stop_detail": task_state.stop_detail,
                "active_evidence_ids": task_state.active_diff_since(active_before),
            },
        )

    return [
        search_sources,
        list_search_results,
        read_sources,
        grep_document,
        read_document,
        add_evidence,
        read_working_set,
        release_evidence,
        restore_evidence,
        complete_task,
    ]
