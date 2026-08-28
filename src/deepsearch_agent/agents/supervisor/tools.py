"""Supervisor 的标准工具注册表。"""

import json

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END
from langgraph.types import Command

from deepsearch_agent.agents.supervisor.state import SupervisorRuntimeContext, WorkingState
from deepsearch_agent.schemas import (
    ForgetEvidence,
    ReadWorkingSet,
    ReportBrief,
    ResearchComplete,
    ResearchDelegate,
    ResearchReady,
    StopReason,
)


def _result(payload: object) -> str:
    return "【系统工具执行结果；不是用户补充】\n" + (
        payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    )


def _working_set_snapshot(working: WorkingState) -> dict[str, object]:
    """返回 Supervisor 当前活跃 Evidence 的轻量目录。"""
    active = working.active_evidences()
    return {
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
    def research_complete(
        report_brief: object,
        reason: str,
        runtime: ToolRuntime[SupervisorRuntimeContext],
    ) -> Command:
        """确认现有 Evidence 足以形成完整报告。"""
        brief = ReportBrief.model_validate(report_brief)
        working = runtime.context.working
        working.report_brief = brief
        if working.active_evidences():
            working.sufficient = True
            working.stop_reason = StopReason.SUFFICIENT
        else:
            working.stop_reason = StopReason.SUFFICIENT_WITHOUT_EVIDENCE
            working.coverage_gaps.append("Supervisor 判定材料充分，但当前没有可交付的 Evidence。")
        return Command(
            goto=END,
            update={
                "messages": [
                    {
                        "role": "tool",
                        "content": _result({"status": "accepted", "reason": reason}),
                        "name": "ResearchComplete",
                        "tool_call_id": runtime.tool_call_id,
                    }
                ]
            },
        )

    @tool("ResearchReady", args_schema=ResearchReady)
    def research_ready(
        report_brief: object,
        reason: str,
        runtime: ToolRuntime[SupervisorRuntimeContext],
    ) -> str:
        """记录可形成部分报告的状态，但不代表整项研究完成。"""
        brief = ReportBrief.model_validate(report_brief)
        working = runtime.context.working
        working.report_brief = brief
        if working.evidences:
            working.partial_ready = True
        else:
            working.coverage_gaps.append("Supervisor 判断可以形成部分报告，但当前没有可交付的 Evidence。")
        return _result({"status": "recorded", "reason": reason})

    @tool("ReadWorkingSet", args_schema=ReadWorkingSet)
    def read_working_set(
        reason: str,
        runtime: ToolRuntime[SupervisorRuntimeContext],
    ) -> str:
        """查看当前活跃 Evidence 的轻量摘要。"""
        del reason
        return _result(_working_set_snapshot(runtime.context.working))

    @tool("ForgetEvidence", args_schema=ForgetEvidence)
    def forget_evidence(
        evidence_ids: list[str],
        reason: str,
        runtime: ToolRuntime[SupervisorRuntimeContext],
    ) -> str:
        """释放当前工作集中的 Evidence，但不删除全局 Evidence。"""
        del reason
        working = runtime.context.working
        forgotten = working.release_evidence(evidence_ids)
        return _result(
            {
                "forgotten_evidence_ids": forgotten,
                "unknown_evidence_ids": [item for item in evidence_ids if item not in forgotten],
                **_working_set_snapshot(working),
            }
        )

    return [research_delegate, research_complete, research_ready, read_working_set, forget_evidence]
