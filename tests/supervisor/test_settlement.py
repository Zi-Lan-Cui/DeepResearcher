"""结算前的竣工验收：在途任务、无证据综合稿与已认证的执行故障。"""

from deepresearcher.agents.supervisor import services
from deepresearcher.agents.supervisor.state import (
    SupervisorRoundState,
)
from deepresearcher.config import AgentConfig
from deepresearcher.schemas import (
    PendingTask,
    ResearcherResult,
    StopReason,
    TaskResult,
    TaskStopReason,
)
from supervisor_kit import (
    FixedResearcher,
    delegate_context_for,
)


def test_freeze_rejected_while_tasks_running():
    supervisor, context = delegate_context_for(FixedResearcher())
    context.round_state.pending_tasks = {
        "task-0001": PendingTask(
            task_id="task-0001", payload={"id": "task-0001", "question": "在途"}, round=1
        ).model_dump()
    }
    receipt = services.freeze_synthesis(context.round_state, "想收口")
    assert receipt["status"] == "rejected"
    assert receipt["tasks_running"] == ["task-0001"]
    assert "在途" in receipt["rejection_cause"]


def _exhausted_direction_execution() -> ResearcherResult:
    """账户级熔断的已认证方向结果:execution_status=failed 仅表达研究未完成。"""
    return ResearcherResult(
        task_result=TaskResult(
            task_id="task-0001",
            round=1,
            task_index=1,
            question="检索方向",
            research_direction="检索方向",
            execution_status="failed",
            coverage_status="insufficient",
            evidence_count=0,
            source_count=0,
            failures=["tavily: quota_exhausted"],
            conclusion="搜索账户级不可用，未获取任何材料。",
            stop_reason=TaskStopReason.BLOCKED_WITHOUT_EVIDENCE,
            stop_detail="搜索熔断",
            provider_exhausted=True,
        ),
        evidences=[],
        selected_evidence_ids=[],
        source_refs=[],
    )


def _blank_round_state() -> SupervisorRoundState:
    return SupervisorRoundState(
        {"query": "研究问题", "clarified_query": "研究问题", "evidences": []},
        current_round=1,
        active_evidence_limit=30,
    )


def _blank_state() -> dict:
    return {"query": "研究问题", "clarified_query": "研究问题", "evidences": []}


def _revise_evidenceless(round_state: SupervisorRoundState) -> dict:
    return services.apply_workbench_changes(
        round_state,
        [
            {
                "op": "framing",
                "framing": {
                    "answer_goal": "回答研究问题",
                    "thesis": "搜索供应商账户级不可用,无任何证据。",
                    "decision_rationale": "执行故障收束。",
                },
            },
            {
                "op": "upsert_aspect",
                "aspect": {
                    "aspect_id": "core",
                    "topic": "核心结论",
                    "role": "主线",
                    "required": True,
                    "status": "uncovered",
                    "gaps": [{"text": "无法获取材料。", "disclose": True}],
                },
            },
        ],
    )


def test_freeze_synthesis_rejects_evidenceless_without_certification():
    """零证据综合稿默认被拒绝，拒绝回执必须带可行动原因。"""
    round_state = _blank_round_state()
    assert _revise_evidenceless(round_state)["status"] == "applied"

    receipt = services.freeze_synthesis(round_state, "请求无证据提交")

    assert receipt["status"] == "rejected"
    assert "Evidence" in str(receipt["rejection_cause"])
    assert round_state.completed_synthesis is None
    assert round_state.stop_reason is None


def test_absorb_certifies_provider_exhaustion_then_freeze_accepts_evidenceless():
    """链路:方向结果带熔断标志 → absorb 认证 → ResearchComplete 接受零证据稿并记专用终态。"""
    round_state = _blank_round_state()
    round_state.absorb(_exhausted_direction_execution())
    assert round_state.provider_exhausted is True

    assert _revise_evidenceless(round_state)["status"] == "applied"
    receipt = services.freeze_synthesis(round_state, "熔断认证后无证据收束")

    assert receipt["status"] == "accepted"
    assert round_state.completed_synthesis is not None
    assert round_state.stop_reason is StopReason.PROVIDER_EXHAUSTED


def test_compose_floor_uses_certified_execution_fault_not_round_budget():
    """默认值按实际原因记录：熔断认证后记 PROVIDER_EXHAUSTED，未认证仍记 ENDED_WITHOUT_COMPLETION。"""
    unproven = _blank_round_state()
    services.compose_final_update(_blank_state(), unproven, AgentConfig())
    assert unproven.stop_reason is StopReason.ENDED_WITHOUT_COMPLETION

    certified = _blank_round_state()
    certified.absorb(_exhausted_direction_execution())
    update = services.compose_final_update(_blank_state(), certified, AgentConfig())

    assert certified.stop_reason is StopReason.PROVIDER_EXHAUSTED
    assert update.run.terminal_reason == "supervisor_provider_exhausted"
    # 零证据进不了正文写作：直送终局渲染，诊断文案取自熔断事实而非轮次预算。
    assert update.supervisor_next == "render_final_report"
    assert "账户级不可用" in update.writer.feedback
