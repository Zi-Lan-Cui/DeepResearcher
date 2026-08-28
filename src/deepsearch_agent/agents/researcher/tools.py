"""ResearchAgent 的标准工具注册表。"""

import json

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, tool

from deepsearch_agent.agents.researcher.state import DirectionRunState, ResearchRuntimeContext
from deepsearch_agent.schemas import (
    ForgetEvidence,
    ReadSources,
    ReadWorkingSet,
    ResearchDirectionComplete,
    SearchSources,
)


def _tool_result(payload: object) -> str:
    """保留统一的工具回执前缀，便于模型区分工具结果和用户输入。"""
    return "【系统工具执行结果；不是用户补充】\n" + (
        payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    )


def _working_set_snapshot(run_state: DirectionRunState) -> dict[str, object]:
    evidences = run_state.evidences
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
        """读取模型选中的候选来源并抽取 Evidence。"""
        result = await runtime.context.read_sources(candidate_ids, reason)
        return _tool_result(result)

    @tool("ReadWorkingSet", args_schema=ReadWorkingSet)
    def read_working_set(
        reason: str,
        runtime: ToolRuntime[ResearchRuntimeContext],
    ) -> str:
        """查看当前方向工作集的轻量摘要。"""
        del reason
        return _tool_result(_working_set_snapshot(runtime.context.run_state))

    @tool("ForgetEvidence", args_schema=ForgetEvidence)
    def forget_evidence(
        evidence_ids: list[str],
        reason: str,
        runtime: ToolRuntime[ResearchRuntimeContext],
    ) -> str:
        """从当前方向工作集释放 Evidence，但不删除全局档案。"""
        del reason
        run_state = runtime.context.run_state
        requested = list(dict.fromkeys(evidence_ids))
        existing = {item.evidence_id for item in run_state.evidences}
        forgotten = [item for item in requested if item in existing]
        run_state.evidences = [
            item for item in run_state.evidences if item.evidence_id not in forgotten
        ]
        return _tool_result(
            {
                "forgotten_evidence_ids": forgotten,
                "unknown_evidence_ids": [item for item in requested if item not in existing],
                **_working_set_snapshot(run_state),
            }
        )

    @tool("ResearchDirectionComplete", args_schema=ResearchDirectionComplete)
    def complete_direction(
        reason: str,
        answered_points: list[str],
        conclusion: str,
        remaining_gaps: list[str],
        runtime: ToolRuntime[ResearchRuntimeContext],
    ) -> str:
        """宣布当前方向结束；不代表整项研究完成。"""
        run_state = runtime.context.run_state
        run_state.remaining_gaps = list(
            dict.fromkeys(gap.strip() for gap in remaining_gaps if gap.strip())
        )
        if run_state.evidences:
            run_state.answered_points = list(dict.fromkeys(answered_points))
            run_state.conclusion = conclusion.strip()
            run_state.stop_reason = "complete"
            run_state.stop_detail = reason
        else:
            run_state.stop_reason = "blocked_without_evidence"
            run_state.stop_detail = reason
            if not run_state.remaining_gaps:
                run_state.remaining_gaps = [reason]
        return _tool_result(
            {
                "status": "accepted",
                "stop_reason": run_state.stop_reason,
                "evidence_count": len(run_state.evidences),
            }
        )

    return [search_sources, read_sources, read_working_set, forget_evidence, complete_direction]
