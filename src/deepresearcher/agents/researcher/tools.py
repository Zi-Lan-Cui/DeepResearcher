"""ResearchAgent 的标准工具注册表。

写状态的工具在出口把回执写进 messages、业务增量写进 ResearcherAgentState 通道:并发工具
(AddEvidence/ReadSources)只交合并语义差量,串行工具(SearchSources/Release/Restore/
ResearchDirectionComplete)交定向的整字段快照。回执文本逐字节不变。
"""

from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.types import Command

from deepresearcher.agents.researcher import services
from deepresearcher.agents.researcher.state import (
    ResearcherLoopContext,
    _candidate_channel_entry,
    _document_channel_entry,
    working_set_snapshot,
)
from deepresearcher.agents.working_set import release_working_set, restore_working_set
from deepresearcher.schemas import (
    AddEvidence,
    DocumentLineRange,
    EvidenceSubmission,
    GrepDocument,
    ListSearchResults,
    ReadDocument,
    ReadSources,
    ReadWorkingSet,
    ReleaseEvidence,
    ResearchDirectionComplete,
    RestoreEvidence,
    SearchSources,
    format_tool_receipt,
)


def _channel_command(
    runtime: ToolRuntime[ResearcherLoopContext],
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


def build_researcher_tools() -> list[BaseTool]:
    """创建一套绑定到 ResearcherLoopContext 的方向级工具。"""

    @tool("SearchSources", args_schema=SearchSources)
    async def search_sources(
        reason: str,
        queries: list[str],
        runtime: ToolRuntime[ResearcherLoopContext],
    ) -> Command:
        """发现当前研究方向的候选来源；不会自动读取网页。"""
        ctx = runtime.context
        candidate_keys_before = set(ctx.loop_state.candidates)
        result = await services.search_sources(
            ctx.deps, ctx.task, ctx.loop_state, ctx.event_context, queries, reason
        )
        new_candidates = {
            key: _candidate_channel_entry(value)
            for key, value in ctx.loop_state.candidates.items()
            if key not in candidate_keys_before
        }
        return _channel_command(
            runtime,
            result,
            tool_name="SearchSources",
            channel_update={
                "queries": list(ctx.loop_state.queries),
                "search_batches": dict(ctx.loop_state.search_batches),
                "candidates": new_candidates,
                "failures": list(ctx.loop_state.failures),
                "provider_exhausted": ctx.loop_state.provider_exhausted,
            },
        )

    @tool("ReadSources", args_schema=ReadSources)
    async def read_sources(
        candidate_ids: list[str],
        reason: str,
        runtime: ToolRuntime[ResearcherLoopContext],
    ) -> Command:
        """抓取模型选中的来源；短文内联，长文返回可按行读取的句柄。"""
        ctx = runtime.context
        document_keys_before = set(ctx.loop_state.documents)
        result = await services.read_sources(
            ctx.deps, ctx.task, ctx.loop_state, ctx.event_context, candidate_ids, reason
        )
        new_documents = {
            key: _document_channel_entry(value)
            for key, value in ctx.loop_state.documents.items()
            if key not in document_keys_before
        }
        return _channel_command(
            runtime,
            result,
            tool_name="ReadSources",
            channel_update={
                "read_urls": list(ctx.loop_state.read_urls),
                "source_refs": list(ctx.loop_state.source_refs),
                "documents": new_documents,
                "selected_candidate_ids": sorted(ctx.loop_state.selected_candidate_ids),
                "skipped": list(ctx.loop_state.skipped),
                "failures": list(ctx.loop_state.failures),
            },
        )

    @tool("ListSearchResults", args_schema=ListSearchResults)
    async def list_search_results(
        search_id: str,
        offset: int,
        limit: int,
        reason: str,
        runtime: ToolRuntime[ResearcherLoopContext],
    ) -> str:
        """分页查看 SearchSources 已落盘的完整候选目录。"""
        ctx = runtime.context
        result = await services.list_search_results(
            ctx.deps, ctx.task, ctx.loop_state, search_id, offset, limit, reason
        )
        return format_tool_receipt(result)

    @tool("GrepDocument", args_schema=GrepDocument)
    async def grep_document(
        document_id: str,
        query: str,
        context_lines: int,
        offset: int,
        reason: str,
        runtime: ToolRuntime[ResearcherLoopContext],
    ) -> str:
        """在当前方向已登记的长文中定位单个关键词；多词请并发起多个调用，翻页用 offset。"""
        ctx = runtime.context
        result = await services.grep_document(
            ctx.deps, ctx.loop_state, document_id, query, context_lines, offset, reason
        )
        return format_tool_receipt(result)

    @tool("ReadDocument", args_schema=ReadDocument)
    async def read_document(
        document_id: str,
        ranges: list[DocumentLineRange],
        reason: str,
        runtime: ToolRuntime[ResearcherLoopContext],
    ) -> str:
        """按行号批量读取当前方向已经登记的文档窗口。"""
        ctx = runtime.context
        result = await services.read_document(
            ctx.deps,
            ctx.loop_state,
            document_id,
            [(item.start_line, item.end_line) for item in ranges],
            reason,
        )
        return format_tool_receipt(result)

    @tool("AddEvidence", args_schema=AddEvidence)
    async def add_evidence(
        evidences: list[EvidenceSubmission],
        reason: str,
        runtime: ToolRuntime[ResearcherLoopContext],
    ) -> Command:
        """批量提交 Evidence；按首尾片段锚在已登记原文中定位，程序逐字回填 quote 后入池。"""
        ctx = runtime.context
        loop_state = ctx.loop_state
        evidence_ids_before = {item.evidence_id for item in loop_state.evidences}
        active_before = loop_state.capture_active()
        result = await services.add_evidence(
            ctx.deps,
            ctx.task,
            loop_state,
            ctx.event_context,
            [
                item.model_dump(mode="json") if isinstance(item, EvidenceSubmission) else dict(item)
                for item in evidences
            ],
            reason,
        )
        new_evidences = [
            item for item in loop_state.evidences if item.evidence_id not in evidence_ids_before
        ]
        return _channel_command(
            runtime,
            result,
            tool_name="AddEvidence",
            channel_update={
                "evidences": new_evidences,
                "active_evidence_ids": loop_state.active_diff_since(active_before),
            },
        )

    @tool("ReadWorkingSet", args_schema=ReadWorkingSet)
    async def read_working_set(
        reason: str,
        runtime: ToolRuntime[ResearcherLoopContext],
    ) -> str:
        """查看当前方向工作集的轻量摘要。"""
        del reason
        return format_tool_receipt(working_set_snapshot(runtime.context.loop_state))

    @tool("ReleaseEvidence", args_schema=ReleaseEvidence)
    async def release_evidence(
        evidence_ids: list[str],
        reason: str,
        runtime: ToolRuntime[ResearcherLoopContext],
    ) -> Command:
        """从当前方向工作集释放 Evidence，但不删除全局档案。"""
        del reason
        loop_state = runtime.context.loop_state
        active_before = loop_state.capture_active()
        result = release_working_set(
            loop_state, evidence_ids, snapshot=lambda: working_set_snapshot(loop_state)
        )
        return _channel_command(
            runtime,
            result,
            tool_name="ReleaseEvidence",
            channel_update={"active_evidence_ids": loop_state.active_diff_since(active_before)},
        )

    @tool("RestoreEvidence", args_schema=RestoreEvidence)
    async def restore_evidence(
        evidence_ids: list[str],
        reason: str,
        runtime: ToolRuntime[ResearcherLoopContext],
    ) -> Command:
        """从方向候选档案恢复 Evidence；不会超过活跃工作集上限。"""
        del reason
        loop_state = runtime.context.loop_state
        active_before = loop_state.capture_active()
        result = restore_working_set(
            loop_state, evidence_ids, snapshot=lambda: working_set_snapshot(loop_state)
        )
        return _channel_command(
            runtime,
            result,
            tool_name="RestoreEvidence",
            channel_update={"active_evidence_ids": loop_state.active_diff_since(active_before)},
        )

    # 不设 return_direct:被拒的提交也须留在循环内改正引用;接受后由 SubmittedExitMiddleware
    # 依 stop_reason 在下一跳退出,与 Supervisor、Writer 的出口方式一致。
    @tool("ResearchDirectionComplete", args_schema=ResearchDirectionComplete)
    async def complete_direction(
        reason: str,
        selected_evidence_ids: list[str],
        conclusion: str,
        remaining_gaps: list[str],
        runtime: ToolRuntime[ResearcherLoopContext],
    ) -> Command:
        """提交当前方向的最终局部结果；被拒则按回执修正后重提。"""
        ctx = runtime.context
        loop_state = ctx.loop_state
        active_before = loop_state.capture_active()
        result = services.complete_direction(
            loop_state,
            reason,
            selected_evidence_ids,
            conclusion,
            remaining_gaps,
        )
        # 被拒时 stop_reason 与活跃集都不变,不落通道,只回一份纠错回执。
        if result.get("status") != "accepted":
            return _channel_command(
                runtime, result, tool_name="ResearchDirectionComplete", channel_update={}
            )
        return _channel_command(
            runtime,
            result,
            tool_name="ResearchDirectionComplete",
            channel_update={
                "remaining_gaps": list(loop_state.remaining_gaps),
                "conclusion": loop_state.conclusion,
                "stop_reason": loop_state.stop_reason,
                "stop_detail": loop_state.stop_detail,
                "active_evidence_ids": loop_state.active_diff_since(active_before),
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
        complete_direction,
    ]
