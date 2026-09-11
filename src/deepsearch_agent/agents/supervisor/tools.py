"""Supervisor 的标准工具注册表。"""

import json
from typing import Literal

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END
from langgraph.types import Command
from pydantic import ValidationError

from deepsearch_agent.agents.supervisor.state import SupervisorRuntimeContext, WorkingState
from deepsearch_agent.schemas import (
    ReadWorkingSet,
    ReleaseEvidence,
    ResearchAspect,
    ResearchComplete,
    ResearchDelegate,
    ResearchSynthesis,
    RestoreEvidence,
    ReviseResearchSynthesis,
    StopReason,
)


def _result(payload: object) -> str:
    return "【系统工具执行结果；不是用户补充】\n" + (
        payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    )


def _working_set_snapshot(working: WorkingState) -> dict[str, object]:
    """返回 Supervisor 当前活跃 Evidence 的轻量目录。"""
    active = working.active_evidences()
    active_ids = {item.evidence_id for item in active}
    reserve = [item for item in working.evidences if item.evidence_id not in active_ids]
    return {
        "working_set_revision": working.working_set_revision,
        "active_evidence": [
            {
                "evidence_id": item.evidence_id,
                "claim": item.claim,
                "support": item.support,
                "confidence": item.confidence,
            }
            for item in active
        ],
        "active_evidence_count": len(active),
        "active_evidence_limit": working.active_evidence_limit,
        "reserve_evidence": [
            {
                "evidence_id": item.evidence_id,
                "claim": item.claim,
                "support": item.support,
            }
            for item in reserve
        ],
        "reserve_evidence_count": len(reserve),
    }


def _synthesis_snapshot(synthesis: ResearchSynthesis | None) -> dict[str, object]:
    if synthesis is None:
        return {"synthesis_revision": 0, "research_synthesis": None}
    return {
        "synthesis_revision": synthesis.revision,
        "based_on_working_set_revision": synthesis.based_on_working_set_revision,
        "readiness": synthesis.readiness,
        "overall_summary": synthesis.overall_summary,
        "aspects": [
            {
                "aspect_id": item.aspect_id,
                "topic": item.topic,
                "status": item.status,
                "summary": item.summary,
                "evidence_ids": item.evidence_ids,
                "remaining_gap": item.remaining_gap,
            }
            for item in synthesis.aspects
        ],
        "selected_evidence_ids": synthesis.selected_evidence_ids,
        "open_gaps": synthesis.open_gaps,
        "conflicts": synthesis.conflicts,
        "next_actions": synthesis.next_actions,
    }


def build_supervisor_tools() -> list[BaseTool]:
    """创建绑定到 SupervisorRuntimeContext 的工具集合。"""

    @tool("ResearchDelegate", args_schema=ResearchDelegate)
    async def research_delegate(
        research_topic: str,
        runtime: ToolRuntime[SupervisorRuntimeContext],
    ) -> str:
        """派发一个具体、可验证且与历史互补的研究方向。"""
        return _result(await runtime.context.delegate_research(research_topic))

    # return_direct 才会让 Command(goto=END) 真正终止 Agent 循环；
    # 缺省时 langchain 仍会把消息送回模型，决策调用白白空转一整圈。
    @tool("ResearchComplete", args_schema=ResearchComplete, return_direct=True)
    async def research_complete(
        synthesis_revision: int,
        reason: str,
        runtime: ToolRuntime[SupervisorRuntimeContext],
    ) -> Command:
        """冻结最新且未过期的完整研究综合稿，并终止研究阶段。"""
        working = runtime.context.working
        synthesis = working.research_synthesis
        accepted = bool(
            synthesis is not None
            and synthesis.revision == synthesis_revision
            and working.synthesis_is_fresh(synthesis)
            and synthesis.readiness == "complete_candidate"
            and synthesis.selected_evidence_ids
        )
        if accepted:
            working.sufficient = True
            working.completed_synthesis = synthesis
            working.stop_reason = StopReason.SUFFICIENT
        else:
            working.coverage_gaps.append(
                "ResearchComplete 拒绝了过期、缺失或尚未达到 complete_candidate 的研究综合稿。"
            )
        return Command(
            goto=END,
            update={
                "messages": [
                    {
                        "role": "tool",
                        "content": _result(
                            {
                                "status": "accepted" if accepted else "rejected",
                                "reason": reason,
                                "requested_revision": synthesis_revision,
                                "current_working_set_revision": working.working_set_revision,
                                **_synthesis_snapshot(synthesis),
                            }
                        ),
                        "name": "ResearchComplete",
                        "tool_call_id": runtime.tool_call_id,
                    }
                ]
            },
        )

    @tool("ReviseResearchSynthesis", args_schema=ReviseResearchSynthesis)
    async def revise_research_synthesis(
        expected_revision: int,
        expected_working_set_revision: int,
        answer_goal: str,
        overall_summary: str,
        aspects: list[ResearchAspect],
        selected_evidence_ids: list[str],
        open_gaps: list[str],
        conflicts: list[str],
        next_actions: list[str],
        readiness: Literal["not_ready", "partial_ready", "complete_candidate"],
        decision_rationale: str,
        runtime: ToolRuntime[SupervisorRuntimeContext],
    ) -> str:
        """用最新方向结果修订当前唯一的研究综合稿；不会结束研究。

        仅当新的研究结果实质改变结论、Evidence 选择、缺口、冲突、下一步或
        可交付状态时使用。它不是工具日志；所有事实总结必须绑定当前活跃
        Evidence。调用 ResearchComplete 前必须先让本综合稿
        对齐最新 working_set_revision。
        """
        working = runtime.context.working
        current_revision = working.research_synthesis.revision if working.research_synthesis else 0
        if (
            expected_revision != current_revision
            or expected_working_set_revision != working.working_set_revision
        ):
            return _result(
                {
                    "status": "stale",
                    "expected_revision": current_revision,
                    "expected_working_set_revision": working.working_set_revision,
                }
            )
        active_ids = set(working.active_evidence_ids)
        referenced_ids = {evidence_id for aspect in aspects for evidence_id in aspect.evidence_ids}
        requested_ids = set(selected_evidence_ids) | referenced_ids
        unknown_ids = sorted(requested_ids - active_ids)
        if unknown_ids:
            return _result(
                {
                    "status": "rejected",
                    "reason": "研究综合稿只能引用当前活跃 Evidence。",
                    "invalid_evidence_ids": unknown_ids,
                    **_working_set_snapshot(working),
                }
            )
        try:
            synthesis = ResearchSynthesis(
                revision=current_revision + 1,
                based_on_working_set_revision=working.working_set_revision,
                answer_goal=answer_goal,
                overall_summary=overall_summary,
                aspects=aspects,
                selected_evidence_ids=selected_evidence_ids,
                open_gaps=open_gaps,
                conflicts=conflicts,
                next_actions=next_actions,
                readiness=readiness,
                decision_rationale=decision_rationale,
            )
        except ValidationError as exc:
            issues = [
                {
                    "field": ".".join(str(part) for part in error["loc"]),
                    "message": error["msg"],
                }
                for error in exc.errors(include_url=False, include_input=False)
            ]
            return _result(
                {
                    "status": "rejected",
                    "reason": "研究综合稿不满足提交契约，请按 issues 修正后重试。",
                    "issues": issues,
                    **_working_set_snapshot(working),
                }
            )
        working.research_synthesis = synthesis
        return _result({"status": "accepted", **_synthesis_snapshot(synthesis)})

    @tool("ReadWorkingSet", args_schema=ReadWorkingSet)
    async def read_working_set(
        reason: str,
        runtime: ToolRuntime[SupervisorRuntimeContext],
    ) -> str:
        """查看当前活跃 Evidence 的轻量摘要。"""
        del reason
        return _result(_working_set_snapshot(runtime.context.working))

    @tool("ReleaseEvidence", args_schema=ReleaseEvidence)
    async def release_evidence(
        evidence_ids: list[str],
        reason: str,
        runtime: ToolRuntime[SupervisorRuntimeContext],
    ) -> str:
        """释放当前工作集中的 Evidence，但不删除全局 Evidence。"""
        del reason
        working = runtime.context.working
        released = working.release_evidence(evidence_ids)
        return _result(
            {
                "released_evidence_ids": released,
                "unknown_evidence_ids": [item for item in evidence_ids if item not in released],
                **_working_set_snapshot(working),
            }
        )

    @tool("RestoreEvidence", args_schema=RestoreEvidence)
    async def restore_evidence(
        evidence_ids: list[str],
        reason: str,
        runtime: ToolRuntime[SupervisorRuntimeContext],
    ) -> str:
        """从全局 Evidence 档案恢复候选，不超过活跃工作集上限。"""
        del reason
        working = runtime.context.working
        restored = working.restore_evidence(evidence_ids)
        return _result(
            {
                "restored_evidence_ids": restored,
                "not_restored_evidence_ids": [
                    item for item in evidence_ids if item not in restored
                ],
                **_working_set_snapshot(working),
            }
        )

    return [
        research_delegate,
        revise_research_synthesis,
        research_complete,
        read_working_set,
        release_evidence,
        restore_evidence,
    ]
