"""等待链路：收据形态、超时处理，以及在途与残留任务的结算。"""

import asyncio
import dataclasses

from deepresearcher.agents.supervisor import Supervisor, services
from deepresearcher.agents.supervisor.state import (
    SupervisorContext,
    SupervisorRoundState,
)
from deepresearcher.config import AgentConfig
from deepresearcher.schemas import (
    PendingTask,
    ResearcherResult,
    TaskResult,
    WaitTask,
)
from supervisor_kit import (
    FixedResearcher,
    GatedResearcher,
    SupervisorLLM,
    delegate_context_for,
    pending_entry,
)


def _capturing_deps(supervisor, events: list):
    def capture(event_type, payload, **_kwargs):
        events.append((event_type, payload))

    return dataclasses.replace(supervisor._deps, emit=capture)


def test_wait_direction_blocks_then_returns_reports():
    """唯一消费口：无到货即挂起，到货后收据含完整方向报告并销账。"""

    async def scenario():
        events: list[tuple[str, dict]] = []
        agent = GatedResearcher()
        supervisor, context = delegate_context_for(agent)
        deps = _capturing_deps(supervisor, events)
        entry = pending_entry("task-0001", "等待方向", "等待")
        context.round_state.pending_tasks = {"task-0001": entry}
        context.task_handles["task-0001"] = asyncio.ensure_future(
            services._execute_research_task(deps, entry["payload"])  # type: ignore[arg-type]
        )

        waiter = asyncio.ensure_future(
            services.wait_for_tasks(deps, context.round_state, context.task_handles)
        )
        await asyncio.sleep(0.05)
        assert not waiter.done()  # 门未开：唯一合法的挂起点就是 wait

        agent.gate.set()
        reports, delta, _timed_out = await waiter
        assert [item["task_id"] for item in reports] == ["task-0001"]
        assert reports[0]["evidence_count"] == 1
        assert delta["pending_tasks"] == {"add": {}, "remove": ["task-0001"]}
        assert [item.task_id for item in context.round_state.task_results] == ["task-0001"]
        arrival = next(
            payload for event_type, payload in events if event_type == "direction_arrival_merged"
        )
        assert arrival["merged_via"] == "wait_tool"
        assert context.task_handles == {}

    asyncio.run(scenario())


def test_wait_direction_returns_immediately_when_arrivals_pending():
    """已有到货、另有方向仍在途时，立即返回已到货的波次，不等最后一路。"""

    async def scenario():
        agent = GatedResearcher()
        supervisor, context = delegate_context_for(agent)
        deps = _capturing_deps(supervisor, [])
        arrived_execution = ResearcherResult(
            task_result=TaskResult(
                task_id="task-0001",
                round=1,
                task_index=1,
                question="已到方向",
                research_direction="已到方向",
                execution_status="completed",
                coverage_status="sufficient",
                evidence_count=0,
                source_count=1,
                stop_reason="complete",
            ),
            evidences=[],
            selected_evidence_ids=[],
            source_refs=[],
        )
        arrived_task: asyncio.Future = asyncio.get_running_loop().create_future()
        arrived_task.set_result(arrived_execution)
        context.round_state.pending_tasks = {
            "task-0001": pending_entry("task-0001", "已到", "已到")
        }
        context.task_handles["task-0001"] = arrived_task

        slow_entry = pending_entry("task-0002", "仍在途", "在途")
        context.round_state.pending_tasks["task-0002"] = slow_entry
        slow_task = asyncio.ensure_future(
            services._execute_research_task(deps, slow_entry["payload"])  # type: ignore[arg-type]
        )
        context.task_handles["task-0002"] = slow_task
        await asyncio.sleep(0.05)  # slow 已进 gate 等待：在途而非欠账，不得被清算判 lost

        try:
            reports, _delta, _timed_out = await asyncio.wait_for(
                services.wait_for_tasks(deps, context.round_state, context.task_handles),
                timeout=1,
            )
        finally:
            agent.gate.set()  # 放行避免悬挂任务
            await asyncio.wait([slow_task])
        assert [item["task_id"] for item in reports] == ["task-0001"]
        assert "task-0001" not in context.task_handles
        assert "task-0002" in context.task_handles  # 在途的继续挂后台，不被裹挟

    asyncio.run(scenario())


def test_wait_direction_without_inflight_returns_hint():
    async def scenario():
        supervisor, context = delegate_context_for(FixedResearcher())
        reports, delta, _timed_out = await services.wait_for_tasks(
            context.deps, context.round_state, context.task_handles
        )
        assert reports == []
        assert delta == {}

    asyncio.run(scenario())


def test_wait_direction_receipts_distinguish_unpinned_from_empty_table():
    """钉不中与无在途必须是两种回执，只有后者报 no_inflight。"""
    with_pinned_mismatch = services.compose_wait_receipt([], {"task-0001"}, 3)
    assert with_pinned_mismatch["status"] == "unknown_task_ids"
    assert with_pinned_mismatch["inflight"] == ["task-0001"]
    assert "task_ids" in str(with_pinned_mismatch["instruction"])

    empty_table = services.compose_wait_receipt([], set(), 3)
    assert empty_table["status"] == "no_inflight"

    landed = services.compose_wait_receipt([{"task_id": "task-0001"}], {"task-0002"}, 4)
    assert landed["status"] == "completed"
    assert "instruction" not in landed


def test_wait_direction_args_strip_injected_control_characters():
    """入参里的控制字符在匹配前被剥除。"""
    parsed = WaitTask.model_validate({"task_ids": ["task\r-0001", "task-0002 ", ""]})
    assert parsed.task_ids == ["task-0001", "task-0002"]


def test_wait_direction_settles_lost_entries_before_waiting():
    """通道有登记、进程无任务体时，先按 lost 结算再进入等待。"""

    async def scenario():
        events: list[tuple[str, dict]] = []
        supervisor, context = delegate_context_for(FixedResearcher())
        deps = _capturing_deps(supervisor, events)
        context.round_state.pending_tasks = {
            "task-0001": pending_entry("task-0001", "丢失方向", "丢失")
        }
        reports, delta, _timed_out = await asyncio.wait_for(
            services.wait_for_tasks(deps, context.round_state, context.task_handles),
            timeout=1,
        )
        assert reports[0]["status"] == "failed"
        assert "丢失" in reports[0]["research_direction"] or reports[0]["failures"]
        assert delta["pending_tasks"]["remove"] == ["task-0001"]
        arrival = next(
            payload for event_type, payload in events if event_type == "direction_arrival_merged"
        )
        assert arrival["merged_via"] == "lost"
        assert context.round_state.task_results[0].execution_status == "failed"

    asyncio.run(scenario())


def test_wait_direction_timeout_reports_timed_out():
    """timeout 到期无一完成:回执返回 timed_out,不误报成 no_inflight 或 unknown_task_ids。"""

    async def scenario():
        supervisor, context = delegate_context_for(FixedResearcher())
        context.round_state.pending_tasks = {
            "task-0001": pending_entry("task-0001", "慢方向", "未完")
        }
        slow: asyncio.Future = asyncio.get_running_loop().create_future()
        context.task_handles["task-0001"] = slow
        try:
            reports, delta, timed_out = await services.wait_for_tasks(
                context.deps, context.round_state, context.task_handles, timeout=0.2
            )
            assert reports == [] and delta == {} and timed_out is True
            receipt = services.compose_wait_receipt(reports, {"task-0001"}, 3, timed_out=timed_out)
            assert receipt["status"] == "timed_out"
            assert receipt["inflight"] == ["task-0001"]
            # 不带到期标记的同类无报告回执仍按既有判定返回 unknown_task_ids，两种情形不混淆。
            assert (
                services.compose_wait_receipt([], {"task-0001"}, 3)["status"] == "unknown_task_ids"
            )
        finally:
            slow.cancel()

    asyncio.run(scenario())


def test_drain_absorbs_late_tasks_and_registers_nothing_left():
    """节点收尾：等齐在途并吸收，merged_via=drain 记入事件；登记清空。"""
    from deepresearcher.observability.scope import AgentScope

    async def scenario():
        events: list[tuple[str, dict]] = []

        def capture(event_type, payload, **_kwargs):
            events.append((event_type, payload))

        agent = FixedResearcher()
        config = AgentConfig(max_research_rounds=3, max_parallel_workers=2)
        supervisor = Supervisor(
            SupervisorLLM(delegate_topics=["x"], complete_args=None),
            config,
            research_agent=agent,
        )
        deps = dataclasses.replace(supervisor._deps, emit=capture)
        round_state = SupervisorRoundState(
            {"query": "问题", "evidences": []}, current_round=1, active_evidence_limit=6
        )
        context = SupervisorContext(
            scope=AgentScope(run_id="run-d", agent_name="Supervisor"),
            deps=deps,
            round_state=round_state,
        )
        task_record = {
            "id": "task-0007",
            "run_id": "run-d",
            "question": "迟到方向",
            "display_title": "迟到",
            "round": 1,
            "sequence": 7,
            "worker_id": "w7",
            "worker_index": 7,
            "parent_task_id": "",
            "operation_id": "o7",
        }
        round_state.pending_tasks = {
            "task-0007": PendingTask(
                task_id="task-0007", payload=dict(task_record), display_title="迟到", round=1
            ).model_dump()
        }
        context.task_handles["task-0007"] = asyncio.ensure_future(
            services._execute_research_task(deps, task_record)  # type: ignore[arg-type]
        )

        await supervisor._drain_task_handles(context, round_state)

        assert [item.task_id for item in round_state.task_results] == ["task-0007"]
        assert round_state.pending_tasks == {}
        assert context.task_handles == {}
        arrival = next(
            payload for event_type, payload in events if event_type == "direction_arrival_merged"
        )
        assert arrival["merged_via"] == "drain"

    asyncio.run(scenario())


def test_residual_tasks_cancelled_not_orphaned():
    async def scenario():
        agent = FixedResearcher()
        supervisor, context = delegate_context_for(agent)
        slow = asyncio.ensure_future(asyncio.sleep(30))
        context.task_handles["task-9999"] = slow
        await supervisor._stop_residual_tasks(context)
        assert slow.cancelled()
        assert context.task_handles == {}

    asyncio.run(scenario())
