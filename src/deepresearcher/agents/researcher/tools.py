"""ResearchAgent 的标准工具注册表。"""

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, tool

from deepresearcher.agents.researcher import services
from deepresearcher.agents.researcher.state import (
    ResearcherLoopContext,
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


def build_researcher_tools() -> list[BaseTool]:
    """创建一套绑定到 ResearcherLoopContext 的方向级工具。"""

    @tool("SearchSources", args_schema=SearchSources)
    async def search_sources(
        reason: str,
        queries: list[str],
        runtime: ToolRuntime[ResearcherLoopContext],
    ) -> str:
        """发现当前研究方向的候选来源；不会自动读取网页。"""
        ctx = runtime.context
        result = await services.search_sources(
            ctx.deps, ctx.task, ctx.loop_state, ctx.event_context, queries, reason
        )
        return format_tool_receipt(result)

    @tool("ReadSources", args_schema=ReadSources)
    async def read_sources(
        candidate_ids: list[str],
        reason: str,
        runtime: ToolRuntime[ResearcherLoopContext],
    ) -> str:
        """抓取模型选中的来源；短文内联，长文返回可按行读取的句柄。"""
        ctx = runtime.context
        result = await services.read_sources(
            ctx.deps, ctx.task, ctx.loop_state, ctx.event_context, candidate_ids, reason
        )
        return format_tool_receipt(result)

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
    ) -> str:
        """批量提交 Evidence；只有能在已登记原文中复算的引用才会入池。"""
        ctx = runtime.context
        result = await services.add_evidence(
            ctx.deps,
            ctx.task,
            ctx.loop_state,
            ctx.event_context,
            ctx.commit_lock,
            [
                item.model_dump(mode="json") if isinstance(item, EvidenceSubmission) else dict(item)
                for item in evidences
            ],
            reason,
        )
        return format_tool_receipt(result)

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
    ) -> str:
        """从当前方向工作集释放 Evidence，但不删除全局档案。"""
        del reason
        loop_state = runtime.context.loop_state
        return format_tool_receipt(
            release_working_set(
                loop_state, evidence_ids, snapshot=lambda: working_set_snapshot(loop_state)
            )
        )

    @tool("RestoreEvidence", args_schema=RestoreEvidence)
    async def restore_evidence(
        evidence_ids: list[str],
        reason: str,
        runtime: ToolRuntime[ResearcherLoopContext],
    ) -> str:
        """从方向候选档案恢复 Evidence；不会超过活跃工作集上限。"""
        del reason
        loop_state = runtime.context.loop_state
        return format_tool_receipt(
            restore_working_set(
                loop_state, evidence_ids, snapshot=lambda: working_set_snapshot(loop_state)
            )
        )

    # 不设 return_direct:边级终结只看工具名、不看回执内容——被拒的提交也会
    # 当场终结循环,模型失去改正引用窗口。接受后 stop_reason 落定,由
    # SubmittedExitMiddleware 在下一跳静默出环(与 Supervisor/Writer 同构)。
    @tool("ResearchDirectionComplete", args_schema=ResearchDirectionComplete)
    async def complete_direction(
        reason: str,
        selected_evidence_ids: list[str],
        conclusion: str,
        remaining_gaps: list[str],
        runtime: ToolRuntime[ResearcherLoopContext],
    ) -> str:
        """提交当前方向的最终局部结果；被拒则按回执修正后重提。"""
        return format_tool_receipt(
            services.complete_direction(
                runtime.context.loop_state,
                reason,
                selected_evidence_ids,
                conclusion,
                remaining_gaps,
            )
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
