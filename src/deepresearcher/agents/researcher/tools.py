"""ResearchAgent 的标准工具注册表。"""

import json

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, tool

from deepresearcher.agents.researcher import services
from deepresearcher.agents.researcher.state import ResearcherLoopContext, ResearcherLoopState
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
)


def _tool_result(payload: object) -> str:
    """保留统一的工具回执前缀，便于模型区分工具结果和用户输入。"""
    return "【系统工具执行结果；不是用户补充】\n" + (
        payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    )


def _working_set_snapshot(loop_state: ResearcherLoopState) -> dict[str, object]:
    evidences = loop_state.active_evidences()
    return {
        "active_evidence": [
            {
                "evidence_id": item.evidence_id,
                "claim": item.claim,
                "support": item.support,
                "confidence": item.confidence,
            }
            for item in evidences
        ],
        "active_evidence_count": len(evidences),
        "active_evidence_limit": loop_state.active_evidence_limit,
        "reserve_evidence_count": len(loop_state.evidences) - len(evidences),
        "archive_evidence_count": len(loop_state.evidences),
        "archive_evidence_limit": loop_state.evidence_archive_limit,
    }


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
        return _tool_result(result)

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
        return _tool_result(result)

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
        return _tool_result(result)

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
        return _tool_result(result)

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
        return _tool_result(result)

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
        return _tool_result(result)

    @tool("ReadWorkingSet", args_schema=ReadWorkingSet)
    async def read_working_set(
        reason: str,
        runtime: ToolRuntime[ResearcherLoopContext],
    ) -> str:
        """查看当前方向工作集的轻量摘要。"""
        del reason
        return _tool_result(_working_set_snapshot(runtime.context.loop_state))

    @tool("ReleaseEvidence", args_schema=ReleaseEvidence)
    async def release_evidence(
        evidence_ids: list[str],
        reason: str,
        runtime: ToolRuntime[ResearcherLoopContext],
    ) -> str:
        """从当前方向工作集释放 Evidence，但不删除全局档案。"""
        del reason
        loop_state = runtime.context.loop_state
        requested = list(dict.fromkeys(evidence_ids))
        existing = set(loop_state.active_evidence_ids)
        released = loop_state.release_evidence(requested)
        return _tool_result(
            {
                "released_evidence_ids": released,
                "unknown_evidence_ids": [item for item in requested if item not in existing],
                **_working_set_snapshot(loop_state),
            }
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
        requested = list(dict.fromkeys(evidence_ids))
        archived = {item.evidence_id for item in loop_state.evidences}
        restored = loop_state.restore_evidence(requested)
        return _tool_result(
            {
                "restored_evidence_ids": restored,
                "not_restored_evidence_ids": [item for item in requested if item not in restored],
                "unknown_evidence_ids": [item for item in requested if item not in archived],
                **_working_set_snapshot(loop_state),
            }
        )

    @tool("ResearchDirectionComplete", args_schema=ResearchDirectionComplete, return_direct=True)
    async def complete_direction(
        reason: str,
        selected_evidence_ids: list[str],
        conclusion: str,
        remaining_gaps: list[str],
        runtime: ToolRuntime[ResearcherLoopContext],
    ) -> str:
        """提交当前方向的最终局部结果，并立即结束工具循环。"""
        loop_state = runtime.context.loop_state
        requested = list(dict.fromkeys(selected_evidence_ids))
        active = set(loop_state.active_evidence_ids)
        invalid = [item for item in requested if item not in active]
        if invalid:
            loop_state.failures.append("completion_unknown_evidence_ids: " + ", ".join(invalid))
            return _tool_result({"status": "rejected", "invalid_evidence_ids": invalid})
        if requested:
            loop_state.active_evidence_ids = set(requested)
        active_evidences = loop_state.active_evidences()
        loop_state.remaining_gaps = list(
            dict.fromkeys(gap.strip() for gap in remaining_gaps if gap.strip())
        )
        if active_evidences:
            loop_state.conclusion = conclusion.strip()
            loop_state.stop_reason = "complete"
        else:
            loop_state.conclusion = ""
            loop_state.stop_reason = "blocked_without_evidence"
            if not loop_state.remaining_gaps:
                loop_state.remaining_gaps = [reason]
        loop_state.stop_detail = reason
        return _tool_result(
            {
                "status": "accepted",
                "stop_reason": loop_state.stop_reason,
                "selected_evidence_ids": [item.evidence_id for item in active_evidences],
            }
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
