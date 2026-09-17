"""ResearchAgent 的标准工具注册表。"""

import json

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, tool

from deepresearcher.agents.researcher.state import DirectionRunState, ResearchRuntimeContext
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


def _working_set_snapshot(run_state: DirectionRunState) -> dict[str, object]:
    evidences = run_state.active_evidences()
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
        "active_evidence_limit": run_state.active_evidence_limit,
        "reserve_evidence_count": len(run_state.evidences) - len(evidences),
        "archive_evidence_count": len(run_state.evidences),
        "archive_evidence_limit": run_state.evidence_archive_limit,
    }


def build_researcher_tools() -> list[BaseTool]:
    """创建一套绑定到 ResearchRuntimeContext 的方向级工具。"""

    @tool("SearchSources", args_schema=SearchSources)
    async def search_sources(
        reason: str,
        queries: list[str],
        runtime: ToolRuntime[ResearchRuntimeContext],
    ) -> str:
        """发现当前研究方向的候选来源；不会自动读取网页。"""
        result = await runtime.context.search_sources(queries, reason)
        return _tool_result(result)

    @tool("ReadSources", args_schema=ReadSources)
    async def read_sources(
        candidate_ids: list[str],
        reason: str,
        runtime: ToolRuntime[ResearchRuntimeContext],
    ) -> str:
        """抓取模型选中的来源；短文内联，长文返回可按行读取的句柄。"""
        result = await runtime.context.read_sources(candidate_ids, reason)
        return _tool_result(result)

    @tool("ListSearchResults", args_schema=ListSearchResults)
    async def list_search_results(
        search_id: str,
        offset: int,
        limit: int,
        reason: str,
        runtime: ToolRuntime[ResearchRuntimeContext],
    ) -> str:
        """分页查看 SearchSources 已落盘的完整候选目录。"""
        result = await runtime.context.list_search_results(search_id, offset, limit, reason)
        return _tool_result(result)

    @tool("GrepDocument", args_schema=GrepDocument)
    async def grep_document(
        document_id: str,
        queries: list[str],
        context_lines: int,
        reason: str,
        runtime: ToolRuntime[ResearchRuntimeContext],
    ) -> str:
        """在当前方向已经登记的长文中批量定位普通文本。"""
        result = await runtime.context.grep_document(document_id, queries, context_lines, reason)
        return _tool_result(result)

    @tool("ReadDocument", args_schema=ReadDocument)
    async def read_document(
        document_id: str,
        ranges: list[DocumentLineRange],
        reason: str,
        runtime: ToolRuntime[ResearchRuntimeContext],
    ) -> str:
        """按行号批量读取当前方向已经登记的文档窗口。"""
        result = await runtime.context.read_document(
            document_id,
            [(item.start_line, item.end_line) for item in ranges],
            reason,
        )
        return _tool_result(result)

    @tool("AddEvidence", args_schema=AddEvidence)
    async def add_evidence(
        evidences: list[EvidenceSubmission],
        reason: str,
        runtime: ToolRuntime[ResearchRuntimeContext],
    ) -> str:
        """批量提交 Evidence；只有能在已登记原文中复算的引用才会入池。"""
        result = await runtime.context.add_evidence(
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
        runtime: ToolRuntime[ResearchRuntimeContext],
    ) -> str:
        """查看当前方向工作集的轻量摘要。"""
        del reason
        return _tool_result(_working_set_snapshot(runtime.context.run_state))

    @tool("ReleaseEvidence", args_schema=ReleaseEvidence)
    async def release_evidence(
        evidence_ids: list[str],
        reason: str,
        runtime: ToolRuntime[ResearchRuntimeContext],
    ) -> str:
        """从当前方向工作集释放 Evidence，但不删除全局档案。"""
        del reason
        run_state = runtime.context.run_state
        requested = list(dict.fromkeys(evidence_ids))
        existing = set(run_state.active_evidence_ids)
        released = run_state.release_evidence(requested)
        return _tool_result(
            {
                "released_evidence_ids": released,
                "unknown_evidence_ids": [item for item in requested if item not in existing],
                **_working_set_snapshot(run_state),
            }
        )

    @tool("RestoreEvidence", args_schema=RestoreEvidence)
    async def restore_evidence(
        evidence_ids: list[str],
        reason: str,
        runtime: ToolRuntime[ResearchRuntimeContext],
    ) -> str:
        """从方向候选档案恢复 Evidence；不会超过活跃工作集上限。"""
        del reason
        run_state = runtime.context.run_state
        requested = list(dict.fromkeys(evidence_ids))
        archived = {item.evidence_id for item in run_state.evidences}
        restored = run_state.restore_evidence(requested)
        return _tool_result(
            {
                "restored_evidence_ids": restored,
                "not_restored_evidence_ids": [item for item in requested if item not in restored],
                "unknown_evidence_ids": [item for item in requested if item not in archived],
                **_working_set_snapshot(run_state),
            }
        )

    @tool("ResearchDirectionComplete", args_schema=ResearchDirectionComplete, return_direct=True)
    async def complete_direction(
        reason: str,
        selected_evidence_ids: list[str],
        conclusion: str,
        remaining_gaps: list[str],
        runtime: ToolRuntime[ResearchRuntimeContext],
    ) -> str:
        """提交当前方向的最终局部结果，并立即结束工具循环。"""
        run_state = runtime.context.run_state
        requested = list(dict.fromkeys(selected_evidence_ids))
        active = set(run_state.active_evidence_ids)
        invalid = [item for item in requested if item not in active]
        if invalid:
            run_state.failures.append("completion_unknown_evidence_ids: " + ", ".join(invalid))
            return _tool_result({"status": "rejected", "invalid_evidence_ids": invalid})
        if requested:
            run_state.active_evidence_ids = set(requested)
        active_evidences = run_state.active_evidences()
        run_state.remaining_gaps = list(
            dict.fromkeys(gap.strip() for gap in remaining_gaps if gap.strip())
        )
        if active_evidences:
            run_state.conclusion = conclusion.strip()
            run_state.stop_reason = "complete"
        else:
            run_state.conclusion = ""
            run_state.stop_reason = "blocked_without_evidence"
            if not run_state.remaining_gaps:
                run_state.remaining_gaps = [reason]
        run_state.stop_detail = reason
        return _tool_result(
            {
                "status": "accepted",
                "stop_reason": run_state.stop_reason,
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
