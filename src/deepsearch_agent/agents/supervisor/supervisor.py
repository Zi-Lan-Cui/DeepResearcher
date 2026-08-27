"""动态研究 Supervisor：以工具调用循环驱动方向级研究。

模型每轮通过 ResearchDelegate 派发方向级 ResearchAgent(子 Agent 抽象为工具),
或通过 ResearchComplete 宣布现有 Evidence 足以成文;轮次预算、任务/URL 去重、
并发上限与异常降级由本地程序强制,不依赖模型自觉。
"""

import asyncio
import json
from collections.abc import Mapping
from typing import TypeVar, cast
from urllib.parse import parse_qsl, urldefrag, urlencode, urlsplit, urlunsplit

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, ValidationError

from deepsearch_agent.agents.researcher import ResearchAgent
from deepsearch_agent.agents.supervisor.state import (
    RunUrlReservations,
    TaskExecution,
    WorkingState,
)
from deepsearch_agent.config import AgentConfig
from deepsearch_agent.context import ContextPolicy
from deepsearch_agent.evidence.models import Evidence
from deepsearch_agent.llm import LLMConfigurationError, LLMInvoker
from deepsearch_agent.observability.events import JsonlSink, make_audit_event
from deepsearch_agent.observability.logger import get_logger
from deepsearch_agent.observability.tracing.context import current_context
from deepsearch_agent.schemas import (
    CoveredTopic,
    ForgetEvidence,
    ReadWorkingSet,
    ReportBrief,
    ResearchAgentResult,
    ResearchComplete,
    ResearchDelegate,
    ResearchDirectionResult,
    ResearchProgress,
    ResearchReady,
    ReviewProgress,
    RunLifecycle,
    SupervisorStateUpdate,
    WriterDirective,
    WriterProgress,
)
from deepsearch_agent.state import ResearchState, SubTask, section

_SUPERVISOR_SYSTEM_PROMPT = """【身份】你是深度研究系统的 Supervisor，管理可并发的方向级 ResearchAgent。
【职责】你决定何时派发互补方向、何时现有 Evidence 足以进入写作，以及 Reflection 拒绝后应改写或补研究。
你不直接撰写报告、不伪造 Evidence，也不把来源数量当作充分性。系统会持续提供方向结果和审阅回流；
请基于完整管理历史作决定。
【工具】你有五个工具：
- ResearchDelegate：派发一个方向级研究任务。每次可派发 1 到 N 个互补方向（不超过并行上限）；
 方向必须具体、可检索、可验证，不能重述原问题，也不能重复历史已做过的方向。
 每个任务描述必须明确：研究对象、范围、待回答的局部问题、与历史任务的边界、排除项和完成标准。
 补缺时只针对当前 Evidence 暴露出的一个或几个明确缺口，缩小范围；不要重新派发一个覆盖整段历史、
 整个流派或全部对象的宽泛任务。任务是否与历史方向重复由你根据研究语义判断，不要依赖程序替你判断。
  每次调用的结果会带着该方向带回的 Evidence 事实与结论注入历史。
- ResearchComplete：宣布现有 Evidence 已足以成文，必须同时给出 report_brief
  （成文目标、覆盖主题、必要结论和限定）。只有确实充分时才调用。
- ResearchReady：记录现有 Evidence 虽未完整覆盖、但已经能够形成一篇基本成立的部分报告，必须同时给出
  report_brief。它不是停止信号；只要还有研究轮次，仍应继续补齐并优先争取 ResearchComplete。
  只有轮次耗尽或没有新方向时，才把这份部分报告交给 Writer；Writer 必须诚实说明未覆盖主题、证据限制和剩余缺口。
 - ReadWorkingSet：查看当前活跃 Evidence 的轻量摘要和数量；不返回完整 quote。
 - ForgetEvidence：将重复、偏题或当前阶段不需要的 Evidence 从活跃工作集中释放；不删除全局 Evidence 档案。
ResearchComplete 与 ResearchReady 不能在同一轮同时使用；如果连一篇有证据支撑的基本报告都无法形成，继续派发 ResearchDelegate。
ResearchAgent 返回的 remaining_gaps 只是局部观察，不是全局结论。你必须综合原问题、所有方向结果和全部
Evidence 自己判断覆盖度；核心主题均覆盖时调用 ResearchComplete；核心主题尚未全部覆盖但已有清晰论证主线时调用 ResearchReady。"""

_ToolModelT = TypeVar("_ToolModelT", bound=BaseModel)

_TOOL_PARALLEL_ALLOWED = {
    ResearchDelegate.__name__: ResearchDelegate.allow_parallel,
    ResearchReady.__name__: ResearchReady.allow_parallel,
    ResearchComplete.__name__: ResearchComplete.allow_parallel,
    ReadWorkingSet.__name__: ReadWorkingSet.allow_parallel,
    ForgetEvidence.__name__: ForgetEvidence.allow_parallel,
}


class ResearchSupervisor:
    """维护研究工具循环、覆盖判断、任务派发与有界并发。"""

    def __init__(
        self,
        llm: LLMInvoker,
        config: AgentConfig,
        *,
        research_agent: ResearchAgent,
        event_sink: JsonlSink | None = None,
        context_policy: ContextPolicy | None = None,
    ):
        if llm is None:
            raise LLMConfigurationError("ResearchSupervisor 需要已装配的 LLMInvoker。")
        if research_agent is None:
            raise ValueError("ResearchSupervisor 需要 ResearchAgent。")
        self.llm = llm
        self.config = config
        self.research_agent = research_agent
        self.event_sink = event_sink
        self.context_policy = context_policy or ContextPolicy()
        self.logger = get_logger("deepsearch_agent.agents.supervisor")
        self._worker_limit = asyncio.Semaphore(config.max_parallel_workers)
        self._decision_runnable = llm.bind_tools(
            [
                ResearchDelegate,
                ResearchReady,
                ResearchComplete,
                ReadWorkingSet,
                ForgetEvidence,
            ],
            tool_choice="any",
        )

    @staticmethod
    def _build_supervisor_context(
        state: ResearchState,
    ) -> list[BaseMessage]:
        """从图 State 恢复 Supervisor 私有上下文；首次运行时写入初始委托。"""
        history = list(state.get("supervisor_messages", []))
        if history:
            return history
        return [
            SystemMessage(content=_SUPERVISOR_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    "【研究委托】\n"
                    + json.dumps(
                        {
                            "research_question": state.get(
                                "clarified_query", state.get("query", "")
                            ),
                            "research_brief": state.get("research_brief", ""),
                        },
                        ensure_ascii=False,
                    )
                )
            ),
        ]

    async def run(self, state: ResearchState) -> dict[str, object]:
        """注入审阅回流（如有），然后执行有界的研究工具调用循环。

        改写还是补研究不由独立决策判定，而由工具循环里的模型直接表达：
        ResearchComplete 进入改写，ResearchDelegate 继续补研究。
        """
        history = self._build_supervisor_context(state)
        # 首次运行时 _build_supervisor_context 会补入初始 System/Human 消息；
        # 快照必须取 State 入口长度，确保这些消息也能持久化到上下文历史。
        history_start = len(state.get("supervisor_messages", []))

        review = section(state, "review", ReviewProgress)
        if review.status == "rejected":
            if review.attempts > self.config.max_post_review_recovery_cycles:
                update = SupervisorStateUpdate(
                    run=RunLifecycle(phase="rendering", terminal_reason="review_recovery_exhausted"),
                    research=section(state, "research", ResearchProgress),
                    writer=section(state, "writer", WriterProgress),
                )
                return {**update.state_update(), "supervisor_messages": history[history_start:]}
            self._append_review_rejection(state, history)

        update = await self._coordinate_research_rounds(state, history)
        return {**update.state_update(), "supervisor_messages": history[history_start:]}

    def _append_review_rejection(self, state: ResearchState, history: list[BaseMessage]) -> None:
        """把审阅拒绝作为消息注入历史；如何响应留给工具循环里的模型。"""
        review = section(state, "review", ReviewProgress)
        history.append(
            HumanMessage(
                content=(
                    "【审阅回流】\n"
                    + json.dumps(
                        {
                            "review_feedback": review.feedback,
                            "fatal_gaps": list(review.gaps),
                            "decision_rules": {
                                "rewrite": "Evidence 已覆盖核心问题，问题仅是措辞、范围、组织或已知材料利用不足；"
                                "调用 ResearchComplete 并提供 report_brief，进入改写。",
                                "research": "核心结论缺少直接证据、来源矛盾，或必须补定义、比较对象或关键事实；"
                                "调用 ResearchDelegate 补充方向。",
                            },
                        },
                        ensure_ascii=False,
                    )
                )
            )
        )

    async def _coordinate_research_rounds(
        self,
        state: ResearchState,
        history: list[BaseMessage],
    ) -> SupervisorStateUpdate:
        """工具调用循环：每轮观察 → 模型决策 → 执行工具，直到终止或预算耗尽。"""
        url_reservations = RunUrlReservations(
            state.get("attempted_source_urls", []),
            normalize_url=self._normalize_source_url,
        )
        working = WorkingState(state, dedup_key=self._task_deduplication_key)

        research = section(state, "research", ResearchProgress)
        current_round = research.current_round
        remaining_rounds = max(0, self.config.max_research_rounds - current_round)
        if section(state, "review", ReviewProgress).status == "rejected":
            # 审阅恢复与研究轮次预算正交：即使研究轮次已耗尽，模型也至少获得
            # “决策 + 一次补研究后的再决策”机会；改写（ResearchComplete）不消耗研究预算。
            remaining_rounds = max(
                remaining_rounds,
                self.config.max_post_review_recovery_cycles + 1,
            )
        if remaining_rounds == 0:
            working.stop_reason = "global_round_budget_exhausted"
            self._emit_research_stopped(current_round, working)
            return self._final_update(state, working, url_reservations)

        for round_no in range(current_round + 1, current_round + 1 + remaining_rounds):
            working.current_round = round_no
            self._emit_audit_event(
                "research_round_started",
                {"round": round_no, "remaining_rounds": remaining_rounds - (round_no - current_round - 1)},
            )
            # 观察只携带历史中不可推导的增量；Evidence 与方向结果由工具消息承载，
            # 委托与审阅回流也在各自消息中，不在此重复投影。
            self._append_research_observation(
                history,
                {
                    "remaining_rounds": remaining_rounds - (round_no - current_round - 1),
                    "coverage_gaps": working.coverage_gaps,
                },
            )
            prepared_history = await self.context_policy.aprepare(history, agent="supervisor")
            response = await self._decision_runnable.ainvoke(prepared_history)

            calls = response.tool_calls or []
            if not calls:
                working.stop_reason = "no_tool_calls"
                self._emit_round_completed(round_no, 0, working, outcome="no_tool_calls")
                self._emit_research_stopped(round_no, working)
                break
            history.append(response)
            if not self._accept_tool_call_batch(calls, history):
                self._emit_round_completed(round_no, 0, working, outcome="rejected_tool_batch")
                continue

            self._emit_audit_event(
                "supervisor_tool_calls",
                {
                    "round": round_no,
                    "calls": [
                        {"name": item["name"], "args": str(item["args"])[:300]}
                        for item in calls
                    ],
                },
            )

            parsed_working_set = self._parse_tool_call(response, "ReadWorkingSet", ReadWorkingSet)
            if parsed_working_set is not None:
                _, call_id = parsed_working_set
                history.append(
                    ToolMessage(
                        content=json.dumps(self._working_set_snapshot(working), ensure_ascii=False),
                        name="ReadWorkingSet",
                        tool_call_id=call_id,
                    )
                )
                self._emit_round_completed(round_no, 0, working, outcome="read_working_set")
                continue

            parsed_forget = self._parse_tool_call(response, "ForgetEvidence", ForgetEvidence)
            if parsed_forget is not None:
                request, call_id = parsed_forget
                forgotten = working.release_evidence(request.evidence_ids)
                history.append(
                    ToolMessage(
                        content=json.dumps(
                            {
                                "forgotten_evidence_ids": forgotten,
                                "unknown_evidence_ids": [
                                    item for item in request.evidence_ids if item not in forgotten
                                ],
                                **self._working_set_snapshot(working),
                            },
                            ensure_ascii=False,
                        ),
                        name="ForgetEvidence",
                        tool_call_id=call_id,
                    )
                )
                self._emit_round_completed(round_no, 0, working, outcome="forget_evidence")
                continue

            parsed_completion = self._parse_tool_call(
                response, "ResearchComplete", ResearchComplete
            )
            if parsed_completion is not None:
                completion, call_id = parsed_completion
                history.append(
                    ToolMessage(
                        content="ResearchComplete 已接受，进入写作。",
                        name="ResearchComplete",
                        tool_call_id=call_id,
                    )
                )
                self._apply_completion(completion, working, round_no)
                self._emit_round_completed(round_no, 0, working, outcome="research_complete")
                break

            parsed_ready = self._parse_tool_call(response, "ResearchReady", ResearchReady)
            if parsed_ready is not None:
                ready, ready_call_id = parsed_ready
                self._apply_ready(ready, working, round_no)
                history.append(
                    ToolMessage(
                        content="ResearchReady 已记录；仍有研究预算时请继续补充研究",
                        name="ResearchReady",
                        tool_call_id=ready_call_id,
                    )
                )
                self._emit_round_completed(round_no, 0, working, outcome="research_ready")
                continue

            executed = await self._execute_tool_calls(
                response,
                round_no=round_no,
                working=working,
                history=history,
                url_reservations=url_reservations,
            )
            if not executed:
                working.stop_reason = "no_new_tasks"
                self._emit_round_completed(round_no, 0, working, outcome="no_new_tasks")
                self._emit_research_stopped(round_no, working)
                break
            self._emit_round_completed(round_no, executed, working, outcome="research_tasks")

        return self._final_update(state, working, url_reservations)

    def _append_research_observation(
        self,
        history: list[BaseMessage],
        payload: dict[str, object],
    ) -> None:
        """把轮次预算等历史不可推导的增量作为观察注入。"""
        history.append(
            HumanMessage(content=("【研究管理观察】\n" + json.dumps(payload, ensure_ascii=False)))
        )

    @staticmethod
    def _accept_tool_call_batch(
        calls: list[Mapping[str, object]],
        history: list[BaseMessage],
    ) -> bool:
        """执行并行策略校验；拒绝批次中的所有调用并逐个回填错误。"""
        if len(calls) <= 1 or all(
            call.get("id")
            and _TOOL_PARALLEL_ALLOWED.get(str(call.get("name", "")), False)
            for call in calls
        ):
            return True
        error = "本轮包含不允许并行的工具调用；本轮所有调用均未执行，请下一轮只调用一个工具。"
        for call in calls:
            history.append(
                ToolMessage(
                    content=error,
                    name=str(call.get("name", "unknown")),
                    tool_call_id=str(call.get("id", "")),
                )
            )
        return False

    @staticmethod
    def _parse_tool_call(
        response: AIMessage,
        name: str,
        model: type[_ToolModelT],
    ) -> tuple[_ToolModelT, str] | None:
        """解析指定工具调用；契约不成立时返回 None，不阻断循环。"""
        for tool_call in response.tool_calls or []:
            if tool_call["name"] == name:
                try:
                    call_id = tool_call.get("id")
                    if not call_id:
                        return None
                    return model.model_validate(tool_call.get("args") or {}), call_id
                except ValidationError:
                    return None
        return None

    @staticmethod
    def _materialize_delegate_tasks(
        response: AIMessage,
        *,
        round_no: int,
        next_index: int,
    ) -> list[tuple[SubTask, str]]:
        """从 tool_calls 提取方向任务；任务序号与 Supervisor 轮次分开保存。"""
        tasks: list[tuple[SubTask, str]] = []
        for index, item in enumerate(response.tool_calls or []):
            if item["name"] != "ResearchDelegate":
                continue
            args = item.get("args") or {}
            question = str(args.get("research_topic", "")).strip()
            if not question:
                continue
            task_index = next_index + index
            tasks.append((
                {
                    "id": f"task-{task_index:04d}",
                    "question": question,
                    "round": round_no,
                    "sequence": task_index,
                    "type": "search",
                    "status": "pending",
                    "assigned_agent": "research_agent",
                    "worker_id": f"research-agent-{task_index:04d}",
                    "worker_index": task_index,
                    "parent_task_id": "",
                    "operation_id": f"research-task-{task_index:04d}",
                }, str(item["id"])))
        return tasks

    async def _execute_tool_calls(
        self,
        response: AIMessage,
        *,
        round_no: int,
        working: WorkingState,
        history: list[BaseMessage],
        url_reservations: RunUrlReservations,
    ) -> int:
        """执行模型派发的 ResearchDelegate；返回实际执行的方向数（0 表示全部被去重）。"""
        task_entries = self._materialize_delegate_tasks(
            response,
            round_no=round_no,
            next_index=working.next_task_index,
        )
        tasks = [task for task, _ in task_entries]
        call_ids = {task["id"]: call_id for task, call_id in task_entries}
        materialized_call_ids = set(call_ids.values())
        for call in response.tool_calls or []:
            call_id = str(call.get("id", ""))
            if call_id not in materialized_call_ids:
                history.append(
                    ToolMessage(
                        content="该工具调用参数无效或无法转换为 ResearchDelegate，已跳过。",
                        name=str(call.get("name", "unknown")),
                        tool_call_id=call_id,
                    )
                )
        new_tasks = working.filter_new_tasks(
            tasks, max_tasks=self.config.max_subtasks_per_round
        )
        executed_ids = {task["id"] for task in new_tasks}
        for task, call_id in task_entries:
            if task["id"] not in executed_ids:
                history.append(
                    ToolMessage(
                        content="该研究方向与历史任务重复或超出本轮任务上限，已跳过。",
                        name="ResearchDelegate",
                        tool_call_id=call_id,
                    )
                )
        if not new_tasks:
            return 0
        batch = await asyncio.gather(
            *(
                self._execute_research_task(
                    task,
                    tool_call_id=call_ids[task["id"]],
                    url_reservations=url_reservations,
                )
                for task in new_tasks
            ),
            return_exceptions=True,
        )
        for task, outcome in zip(new_tasks, batch, strict=True):
            if isinstance(outcome, asyncio.CancelledError):
                raise outcome
            if isinstance(outcome, BaseException):
                execution = TaskExecution.failed_for(
                    task,
                    round_no,
                    str(outcome)[:500],
                    tool_call_id=call_ids[task["id"]],
                )
            else:
                execution = outcome
            history.append(execution.message)
            working.absorb(execution)
        return len(new_tasks)

    def _apply_completion(
        self,
        completion: ResearchComplete,
        working: WorkingState,
        round_no: int,
    ) -> None:
        """处理终止信号：有证据进入写作；零证据由程序护栏降级，不空手成文。"""
        working.report_brief = completion.report_brief
        if working.active_evidences():
            working.sufficient = True
            working.stop_reason = "supervisor_sufficient"
        else:
            working.stop_reason = "sufficient_without_evidence"
            working.coverage_gaps.append("Supervisor 判定材料充分，但当前没有可交付的 Evidence。")
        self._emit_research_stopped(round_no, working)

    def _apply_ready(
        self,
        ready: ResearchReady,
        working: WorkingState,
        round_no: int,
    ) -> None:
        """记录模型判断的部分就绪状态；没有 Evidence 时仍由本地护栏拒绝。

        该状态只是候选标记，不结束当前 Supervisor 研究循环。
        """
        working.report_brief = ready.report_brief
        if working.evidences:
            working.partial_ready = True
            self._emit_audit_event(
                "research_partial_ready_candidate",
                {"round": round_no, "reason": ready.reason},
            )
        else:
            working.coverage_gaps.append("Supervisor 判断可以形成部分报告，但当前没有可交付的 Evidence。")

    async def _execute_research_task(
        self,
        task: SubTask,
        *,
        tool_call_id: str,
        url_reservations: RunUrlReservations,
    ) -> TaskExecution:
        """执行单个方向研究；worker 异常降级为 failed 结果，不中断整轮。"""
        round_no = int(task.get("round", 1))
        task_context = {
            "task_id": task["id"],
            "worker_id": task.get("worker_id", task["id"]),
            "worker_index": int(task.get("worker_index", task.get("sequence", 0))),
            "parent_task_id": task.get("parent_task_id", ""),
            "operation_id": task.get("operation_id", task["id"]),
            "concurrency_limit": self.config.max_parallel_workers,
        }
        async with self._worker_limit:
            self._emit_audit_event(
                "research_task_started",
                {
                    **task_context,
                    "task_index": int(task.get("sequence", 0)),
                    "component": "research_agent",
                    "question": task["question"][: self.config.supervisor_preview_chars],
                    "type": task["type"],
                },
            )
            try:
                result = await self.research_agent.run(
                    task,
                    claim_url=url_reservations.reserve,
                    on_url_already_attempted=lambda url: self._emit_audit_event(
                        "source_duplicate_skipped",
                        {
                            "task_id": task["id"],
                            "url": self._normalize_source_url(url),
                            "dedup_scope": "research_run",
                        },
                    ),
                )
                # 研究员是子 Agent 边界：在写入审计事件或 State 前先验证结果契约；
                # 同 run 内已是模型时零开销，防御性恢复覆盖跨进程 checkpoint。
                agent_result = (
                    result
                    if isinstance(result, ResearchAgentResult)
                    else ResearchAgentResult.model_validate(result)
                )
                task_result = agent_result.task_result
                evidences = list(agent_result.evidences)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                execution = TaskExecution.failed_for(
                    task,
                    round_no,
                    str(exc)[: self.config.supervisor_preview_chars],
                    tool_call_id=tool_call_id,
                )
                self._emit_audit_event(
                    "research_task_failed",
                    {**task_context, **execution.task_result.model_dump()},
                    component="research_agent",
                )
                return execution
            self._emit_audit_event(
                "research_task_completed",
                {**task_context, **task_result.model_dump()},
                component="research_agent",
            )
            return TaskExecution(
                task_result=task_result,
                evidences=evidences,
                source_refs=list(agent_result.source_refs),
                message=TaskExecution._result_message(
                    task,
                    task_result,
                    evidences,
                    tool_call_id=tool_call_id,
                ),
            )

    def _emit_research_stopped(self, round_no: int, working: WorkingState) -> None:
        self._emit_audit_event(
            "research_stopped",
            {
                "round": round_no,
                "reason": working.stop_reason,
                "evidence_count": len(working.evidences),
            },
        )

    def _emit_round_completed(
        self,
        round_no: int,
        task_count: int,
        working: WorkingState,
        *,
        outcome: str,
    ) -> None:
        self._emit_audit_event(
            "research_round_completed",
            {
                "round": round_no,
                "task_count": task_count,
                "outcome": outcome,
                "completed_tasks": sum(
                    item.execution_status == "completed"
                    for item in working.task_results
                    if item.round == round_no
                ),
                "failed_tasks": sum(
                    item.execution_status == "failed"
                    for item in working.task_results
                    if item.round == round_no
                ),
                "evidence_added": sum(
                    item.evidence_count for item in working.task_results if item.round == round_no
                ),
                "total_evidence_count": len(working.evidences),
            },
        )

    def _final_update(
        self,
        state: ResearchState,
        working: WorkingState,
        url_reservations: RunUrlReservations,
    ) -> SupervisorStateUpdate:
        """把工作状态转为 State 增量与路由决策。"""
        if not working.sufficient and not working.stop_reason:
            working.stop_reason = "round_budget_exhausted"
        meets_material_floor = self._meets_partial_report_threshold(working)
        # ResearchReady 提供语义判断，但不能绕过本地最低材料安全线；预算耗尽时，
        # 达到安全线即可兜底进入 Writer，并由 Writer 明确披露未完成部分。
        can_generate_partial = meets_material_floor and (
            working.partial_ready or working.stop_reason in {
                "round_budget_exhausted",
                "global_round_budget_exhausted",
                "no_new_tasks",
            }
        )
        if working.partial_ready and not meets_material_floor:
            working.coverage_gaps.append(
                "Supervisor 判断可以形成部分报告，但当前材料未达到本地最低生成门槛。"
            )
        if can_generate_partial and working.report_brief is None:
            working.report_brief = self._build_partial_report_brief(working)
        can_write = working.sufficient or can_generate_partial
        writer_directive = self._build_writer_directive(state, working) if can_write else None
        deltas = working.deltas()
        research_status = "completed" if working.sufficient else "incomplete"
        generation_mode = (
            "full" if working.sufficient else "partial" if can_generate_partial else "not_ready"
        )
        can_continue_to_writer = can_write
        return SupervisorStateUpdate(
            evidences=cast(list[Evidence], deltas["evidences"]),
            source_refs=cast(list[str], deltas["source_refs"]),
            task_results=cast(list[ResearchDirectionResult], deltas["task_results"]),
            attempted_source_urls=url_reservations.newly_attempted,
            report_brief=working.report_brief,
            writer_directive=writer_directive,
            active_evidence_ids=sorted(working.active_evidence_ids),
            run=RunLifecycle(
                phase="writing" if can_continue_to_writer else "rendering",
                terminal_reason="" if can_continue_to_writer else working.stop_reason,
            ),
            research=ResearchProgress(
                status=research_status,
                current_round=working.current_round,
                coverage_gaps=working.coverage_gaps,
                generation_mode=generation_mode,
                is_sufficient=working.sufficient,
            ),
            writer=WriterProgress(
                status="not_started",
                feedback=(
                    ""
                    if working.sufficient
                    else self._describe_research_stop(working.stop_reason, working.coverage_gaps)
                ),
            ),
            supervisor_next="writer" if can_write else "render_final_report",
        )

    def _build_writer_directive(
        self,
        state: ResearchState,
        working: WorkingState,
    ) -> WriterDirective:
        """把 Supervisor 的判断整理成 Writer 唯一可见的写作指令。"""
        if working.report_brief is None:
            raise RuntimeError("WriterDirective 需要 ReportBrief。")
        review = section(state, "review", ReviewProgress)
        previous_draft = str(state.get("report_draft") or state.get("writer_draft") or "")
        revision_instructions = (
            [
                *([review.feedback] if review.feedback else []),
                *(f"修复缺口：{gap}" for gap in review.gaps),
            ]
            if review.status == "rejected"
            else []
        )
        return WriterDirective(
            query=str(state.get("clarified_query") or state.get("query") or ""),
            report_brief=working.report_brief,
            research_status="completed" if working.sufficient else "incomplete",
            generation_mode="full" if working.sufficient else "partial",
            evidence_ids=[item.evidence_id for item in working.active_evidences()],
            known_gaps=list(dict.fromkeys(working.coverage_gaps))[: self.config.report_max_caveats],
            revision_instructions=revision_instructions,
            previous_draft=previous_draft,
        )

    def _meets_partial_report_threshold(self, working: WorkingState) -> bool:
        """判断材料是否足以写一份明确标注缺口的部分报告。"""
        active_evidences = working.active_evidences()
        if len(active_evidences) < self.config.partial_report_min_evidences:
            return False
        source_count = len({item.source_url for item in active_evidences if item.source_url})
        return source_count >= self.config.partial_report_min_sources

    @staticmethod
    def _working_set_snapshot(working: WorkingState) -> dict[str, object]:
        """构造 Supervisor 工作集摘要，不把完整 quote 重复注入上下文。"""
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

    def _build_partial_report_brief(self, working: WorkingState) -> ReportBrief:
        """预算耗尽且没有 ResearchComplete 时，为 Writer 生成保守任务书。"""
        topics = [
            CoveredTopic(
                topic=result.research_direction or result.question,
                role="已获得证据的局部方向",
                reason="该方向已返回可用于成文的 Evidence。",
                required=False,
            )
            for result in working.task_results
            if result.evidence_count > 0
        ]
        if not topics:
            topics = [
                CoveredTopic(
                    topic=working.research_query or "已有研究材料",
                    role="部分证据",
                    reason="当前仅允许基于已获得材料进行有限回答。",
                    required=False,
                )
            ]
        return ReportBrief(
            answer_goal=(
                working.research_query
                or "基于已获得 Evidence 形成一份明确标注范围和缺口的部分研究报告。"
            ),
            covered_topics=topics[: self.config.report_max_topics],
            required_points=[],
            caveats=list(dict.fromkeys(working.coverage_gaps))[: self.config.report_max_caveats],
        )

    @staticmethod
    def _describe_research_stop(stop_reason: str, coverage_gaps: list[str]) -> str:
        """把研究无法继续的原因保留给最终不完整报告与事件诊断。"""
        detail = next(
            (gap for gap in reversed(coverage_gaps) if gap.strip()), "未形成可验证的完整覆盖。"
        )
        prefix = {
            "sufficient_without_evidence": "充分性决策与 Evidence 状态矛盾。",
            "no_new_tasks": "没有可去重的新研究任务。",
            "no_tool_calls": "Supervisor 模型既未派发研究任务，也未给出充分性决策。",
            "round_budget_exhausted": "研究轮次预算已耗尽。",
            "global_round_budget_exhausted": "研究轮次预算已耗尽，Supervisor 尚未确认材料足以成文。",
        }.get(stop_reason, "Supervisor 未确认现有材料足以形成完整研究报告。")
        return f"{prefix} {detail}"

    def _emit_audit_event(
        self,
        event_type: str,
        payload: dict[str, object],
        *,
        component: str = "supervisor",
    ) -> None:
        self.logger.info("%s payload=%s", event_type, payload)
        if self.event_sink is None:
            return
        context = current_context()
        self.event_sink.write(
            make_audit_event(
                event_type,
                trace_id=context.trace_id if context else None,
                span_id=context.span_id if context else None,
                run_id=context.run_id if context else None,
                session_id=context.session_id if context else None,
                node_id=context.node_id if context else "supervisor",
                payload=payload,
                component=component,
            )
        )

    @staticmethod
    def _normalize_source_url(url: str) -> str:
        url, _ = urldefrag(url.strip())
        parts = urlsplit(url)
        query = urlencode(
            [
                (key, value)
                for key, value in parse_qsl(parts.query, keep_blank_values=True)
                if not key.casefold().startswith("utm_")
            ],
            doseq=True,
        )
        return urlunsplit(
            (parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), query, "")
        )

    @staticmethod
    def _task_deduplication_key(question: str) -> str:
        return "".join(question.lower().split())
