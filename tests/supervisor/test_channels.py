"""通道定义与重建：并发范围、reducer 语义、序列化往返。"""

import asyncio
from types import SimpleNamespace
from typing import Annotated, get_args, get_origin, get_type_hints

from deepresearcher.agents.middleware import StateRebuildMiddleware
from deepresearcher.agents.reducers import (
    merge_active_set,
    merge_registry_by_key,
    replace_channel,
    retain_if_absent,
)
from deepresearcher.agents.supervisor import services
from deepresearcher.agents.supervisor.agent import (
    SUPERVISOR_SERIAL_TOOLS,
)
from deepresearcher.agents.supervisor.state import (
    SupervisorContext,
    SupervisorDeps,
    SupervisorRoundState,
    SupervisorState,
)
from deepresearcher.agents.supervisor.tools import build_supervisor_tools
from deepresearcher.config import AgentConfig
from deepresearcher.schemas import (
    AspectIssue,
    PendingTask,
    ResearchAspect,
    ResearcherResult,
    ResearchSynthesis,
    StopReason,
    TaskResult,
)
from fakes import make_evidence
from supervisor_kit import (
    FixedResearcher,
)


def _reducer_of(hint):
    if get_origin(hint) is Annotated:
        metadata = get_args(hint)[1:]
        return next((arg for arg in metadata if callable(arg)), None)
    return None


def _replace_semantics_channels() -> set[str]:
    """last-write 与 retain-if-absent 语义的通道，整字段快照会互相覆盖，禁止并发写。"""
    hints = get_type_hints(SupervisorState, include_extras=True)
    replace_reducers = {replace_channel, retain_if_absent}
    return {name for name, hint in hints.items() if _reducer_of(hint) in replace_reducers}


def test_only_research_delegate_and_workbench_are_concurrent():
    """除 ResearchDelegate 与 UpdateWorkbench 外，每个工具都必须落在串行名单内。"""
    tool_names = {item.name for item in build_supervisor_tools()}
    assert tool_names - set(SUPERVISOR_SERIAL_TOOLS) == {"ResearchDelegate", "UpdateWorkbench"}


def test_concurrent_delegate_never_snapshots_replace_channels():
    """派发增量与到达吸收增量只能落在合并语义通道,绝不写 replace 语义通道。"""
    round_state = SupervisorRoundState(
        {"query": "q", "evidences": []}, current_round=1, active_evidence_limit=6
    )
    task_record = {
        "id": "task-0001",
        "run_id": "run-x",
        "question": "方向",
        "display_title": "",
        "round": 1,
        "sequence": 1,
        "worker_id": "w",
        "worker_index": 1,
        "parent_task_id": "",
        "operation_id": "o",
    }
    entry = PendingTask(task_id="task-0001", payload=dict(task_record), round=1).model_dump()
    round_state.pending_tasks = {"task-0001": entry}
    deps = SupervisorDeps(
        config=AgentConfig(),
        research_agent=FixedResearcher(),
        worker_limit=asyncio.Semaphore(1),
        emit=lambda *_args, **_kwargs: None,
    )
    execution = ResearcherResult.failed_for(task_record, 1, "boom")  # type: ignore[arg-type]
    _, delta = services.absorb_arrival(
        deps,
        round_state,
        task_record,
        execution,
        merged_via="drain",  # type: ignore[arg-type]
    )
    assert set(delta) & _replace_semantics_channels() == set()
    assert delta["pending_tasks"] == {"add": {}, "remove": ["task-0001"]}
    assert round_state.pending_tasks == {}  # 吸收即销账，登记不留残值


def test_round_state_round_trips_through_channels():
    """initial_channel_state → from_channel_values 与源对象逐字段等价，通道是唯一事实源。"""
    source = SupervisorRoundState(
        {"query": "问题", "clarified_query": "问题", "evidences": []},
        current_round=1,
        active_evidence_limit=6,
    )
    source.absorb(
        ResearcherResult(
            task_result=TaskResult(
                task_id="task-0001",
                round=1,
                task_index=1,
                question="方向",
                research_direction="方向",
                execution_status="completed",
                coverage_status="sufficient",
                evidence_count=1,
                source_count=1,
                stop_reason="complete",
            ),
            evidences=[make_evidence("e1", "事实")],
            selected_evidence_ids=["e1"],
            source_refs=["https://a"],
        )
    )
    source.set_stop_reason(StopReason.COMPLETED)

    channels = source.initial_channel_state()
    # 入口 input 在生产里经 langgraph 施加通道 reducer；测试如实模拟这一步。
    channels["active_evidence_ids"] = merge_active_set(None, channels["active_evidence_ids"])
    restored = SupervisorRoundState.from_channel_values(
        channels,
        research_query=source.research_query,
        current_round=source.current_round,
        active_evidence_limit=source.active_evidence_limit,
    )
    assert [item.evidence_id for item in restored.evidences] == ["e1"]
    assert restored.active_evidence_ids == source.active_evidence_ids
    assert restored.source_refs == source.source_refs
    assert [item.task_id for item in restored.task_results] == ["task-0001"]
    assert restored.working_set_revision == source.working_set_revision
    assert restored.stop_reason is source.stop_reason


def test_merge_registry_by_key_applies_add_and_remove_deltas():
    merged = merge_registry_by_key(None, {"add": {"task-0001": {"round": 1}}, "remove": []})
    merged = merge_registry_by_key(merged, {"add": {"task-0002": {"round": 1}}, "remove": []})
    merged = merge_registry_by_key(merged, {"add": {}, "remove": ["task-0001"]})
    assert sorted(merged) == ["task-0002"]


def test_pending_tasks_round_trip_and_inflight_snapshot():
    from deepresearcher.agents.supervisor.state import workbench_view as working_set_snapshot

    entry = PendingTask(
        task_id="task-0001",
        payload={"id": "task-0001", "question": "方向"},
        display_title="短题",
        round=2,
    ).model_dump()
    source = SupervisorRoundState({"query": "问题"}, current_round=2, active_evidence_limit=6)
    source.pending_tasks = {"task-0001": entry}
    # 在途登记不经入口种子：重放再施加 {} 会抹掉已提交登记，只由派发批 Command 写入。
    assert "pending_tasks" not in source.initial_channel_state()

    restored = SupervisorRoundState.from_channel_values(
        {"pending_tasks": {"task-0001": entry}},
        research_query="问题",
        current_round=2,
        active_evidence_limit=6,
    )
    assert restored.pending_tasks == source.pending_tasks
    snapshot = working_set_snapshot(restored)
    assert snapshot["inflight"] == [{"task_id": "task-0001", "display_title": "短题", "round": 2}]
    empty = SupervisorRoundState({"query": "问题"}, current_round=1, active_evidence_limit=6)
    assert working_set_snapshot(empty)["inflight"] == []  # 形状恒有键，不随有无在途漂移


def test_rebuild_middleware_discards_uncommitted_memory():
    """批重建只认通道：未经 Command 提交的内存改动在下一模型回合被替换。"""

    class _Runtime:
        def __init__(self, context):
            self.context = context

    committed = SupervisorRoundState(
        {"query": "q", "evidences": []}, current_round=1, active_evidence_limit=6
    )
    # abefore_model 触 round_state、两个标量与 deps.config 里的上限，scope 用占位对象即可。
    context = SupervisorContext(
        scope=SimpleNamespace(),
        deps=SimpleNamespace(
            config=SimpleNamespace(supervisor_max_active_evidences=6, supervisor_max_turns=34)
        ),
        round_state=committed,
        research_query="q",
        current_round=1,
    )
    # 先固化已提交通道(入口 input 在生产里经 langgraph 施加通道 reducer)，再对内存对象
    # 做未 Command 提交的改动：重建只认前者。
    committed_channels = committed.initial_channel_state()
    committed_channels["active_evidence_ids"] = merge_active_set(
        None, committed_channels["active_evidence_ids"]
    )
    context.round_state.provider_exhausted = True
    context.round_state.working_set_revision = 99

    asyncio.run(StateRebuildMiddleware().abefore_model(committed_channels, _Runtime(context)))
    assert context.round_state.provider_exhausted is False
    assert context.round_state.working_set_revision == 0


def test_report_brief_projection_filters_by_disclose():
    """任务书投影：剥管理字段、空结论以首条 gap 顶上、caveats 只收 disclose=true 并按板块归位。"""
    synthesis = ResearchSynthesis(
        revision=1,
        based_on_working_set_revision=0,
        answer_goal="目标",
        thesis="全文主张。",
        aspects=[
            ResearchAspect(
                aspect_id="a1",
                topic="章一",
                role="主线",
                required=True,
                status="covered",
                conclusion="章一结论。",
                key_points=["要点甲", "要点乙"],
                evidence_ids=["e1"],
                gaps=[
                    AspectIssue(text="原文锚点未补", disclose=False),
                    AspectIssue(text="关键期数据缺失", disclose=True),
                ],
                conflicts=[AspectIssue(text="两来源口径不相容", disclose=True)],
            ),
            ResearchAspect(
                aspect_id="a2",
                topic="章二",
                role="边界",
                required=True,
                status="uncovered",
                gaps=[AspectIssue(text="尚未研究", disclose=True)],
            ),
        ],
        decision_rationale="依据。",
    )

    brief = services.report_brief_from_synthesis(synthesis, max_caveats=6)

    assert [chapter.topic for chapter in brief.chapters] == ["章一", "章二"]
    first = brief.chapters[0]
    assert first.key_points == ["要点甲", "要点乙"]
    assert not hasattr(first, "aspect_id") and not hasattr(first, "status")
    assert brief.chapters[1].conclusion == "尚未研究"
    assert brief.caveats == [
        "缺口|板块「章一」:关键期数据缺失",
        "结论冲突|板块「章一」:两来源口径不相容",
        "缺口|板块「章二」:尚未研究",
    ]


def test_legacy_research_synthesis_shape_migrates():
    """旧 checkpoint 结构迁移：summary/remaining_gap 改名生效，全局 gaps/conflicts 并入首块。"""
    legacy = ResearchSynthesis.model_validate(
        {
            "revision": 2,
            "based_on_working_set_revision": 1,
            "answer_goal": "旧目标",
            "thesis": "旧主张",
            "aspects": [
                {
                    "aspect_id": "core",
                    "topic": "核心",
                    "role": "主线",
                    "status": "partial",
                    "summary": "旧摘要",
                    "evidence_ids": [],
                    "remaining_gap": "旧缺口",
                }
            ],
            "open_gaps": ["全局旧缺口"],
            "conflicts": ["旧冲突"],
            "decision_rationale": "旧依据",
            "next_actions": ["早已退役的键"],
        }
    )

    aspect = legacy.aspects[0]
    assert aspect.conclusion == "旧摘要"
    assert [issue.text for issue in aspect.gaps] == ["旧缺口", "全局旧缺口", "旧冲突"]
    assert all(issue.disclose for issue in aspect.gaps)
