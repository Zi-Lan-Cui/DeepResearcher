"""任务级自主研究 Agent：在全局预算内完成一个受派任务的局部研究"""

import asyncio
from typing import Any, Literal, cast

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.errors import GraphRecursionError

from deepresearcher.agents.middleware import (
    AGENT_RECURSION_LIMIT,
    MiddlewareProfile,
    StateRebuildMiddleware,
    SubmissionGuard,
    build_agent_middleware,
)
from deepresearcher.agents.researcher.state import (
    ResearcherContext,
    ResearcherDeps,
    ResearcherState,
    ResearcherTaskState,
)
from deepresearcher.agents.researcher.tools import build_researcher_tools
from deepresearcher.config import DEFAULT_CONTEXT_WINDOW_TOKENS, AgentConfig
from deepresearcher.observability.events import JsonlSink, emit_agent_event
from deepresearcher.observability.logging_config import get_logger
from deepresearcher.observability.scope import AgentScope
from deepresearcher.prompts import (
    get_runtime_environment,
    language_directive,
    load_prompt,
    render_data_section,
)
from deepresearcher.schemas import (
    ResearcherResult,
    Task,
    TaskResult,
    TaskStopReason,
)
from deepresearcher.tools import SearchTool, SourceReaderTool
from deepresearcher.tools.web.materials import ResearchMaterialStore

_RESEARCHER_SYSTEM_PROMPT = load_prompt("researcher")

# 串行工具必须单独一轮：与任何其他调用同批时，整批驳回、退回模型重规划。
RESEARCHER_SERIAL_TOOLS: frozenset[str] = frozenset(
    {
        "ReadWorkingSet",
        "ReleaseEvidence",
        "RestoreEvidence",
        "TaskComplete",
    }
)


class Researcher:
    """自主完成一个研究任务，不决定整项研究是否已经充分。"""

    def __init__(
        self,
        llm: BaseChatModel,
        config: AgentConfig,
        *,
        search_tool: SearchTool,
        reader_tool: SourceReaderTool,
        event_sink: JsonlSink | None = None,
        context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS,
        material_store: ResearchMaterialStore | None = None,
    ):
        self.llm = llm
        self.config = config
        self.search_tool = search_tool
        self.reader_tool = reader_tool
        self.material_store = material_store or getattr(reader_tool, "material_store", None)
        self.event_sink = event_sink
        self.logger = get_logger("deepresearcher.agents.researcher")
        # 构造期定稿的稳定零件；跨并发 run 共享只读，run 期不再新增任何依赖。
        self._deps = ResearcherDeps(
            config=self.config,
            search_tool=self.search_tool,
            reader_tool=self.reader_tool,
            material_store=self.material_store,
            emit=self._emit,
        )
        self._agent_loop = create_agent(
            model=self.llm,
            tools=build_researcher_tools(),
            system_prompt=_RESEARCHER_SYSTEM_PROMPT
            + "\n"
            + language_directive(config.output_language),
            context_schema=ResearcherContext,
            state_schema=ResearcherState,
            middleware=cast(
                Any,
                [
                    StateRebuildMiddleware(),
                    *build_agent_middleware(
                        MiddlewareProfile(
                            agent_name="Researcher",
                            event_slug="researcher",
                            model=self.llm,
                            # +1：步数预算走完后再给一次不带工具的收尾回合。
                            max_turns=self.config.research_agent_max_turns + 1,
                            context_window_tokens=context_window_tokens,
                            retry_tools=[["SearchSources"], ["ReadSources"]],
                            serial_tools=set(RESEARCHER_SERIAL_TOOLS),
                            submission_guard=SubmissionGuard(
                                missing_submission_message=(
                                    "你还没有调用 TaskComplete。"
                                    "普通文本不是有效收尾；请继续研究，或立即调用该工具提交。"
                                ),
                                submitted_probe=lambda ctx: (
                                    getattr(getattr(ctx, "task_state", None), "stop_reason", None)
                                    in {
                                        TaskStopReason.COMPLETE,
                                        TaskStopReason.BLOCKED_WITHOUT_EVIDENCE,
                                    }
                                ),
                                max_rejections=self.config.finalization_attempts,
                                warn_below_turns=4,
                                low_turn_budget_message=(
                                    "【剩余回合提醒】当前仅剩 {remaining_turns} 个模型回合。"
                                    "停止扩展范围、翻页或重复搜索。请立即将已读取且可逐字定位的"
                                    "原文批量提交给 AddEvidence；如被拒绝，只修正引用，不再搜索。"
                                    "随后调用 "
                                    "TaskComplete 诚实提交已覆盖内容和剩余缺口。"
                                ),
                            ),
                            # 与 submitted_probe 同一信号：stop_reason 收成终值后下一跳直接退出
                            # 循环；被拒回执不写 stop_reason，模型留在循环内改正。
                            exit_probe=lambda ctx, _state: (
                                getattr(getattr(ctx, "task_state", None), "stop_reason", None)
                                in {
                                    TaskStopReason.COMPLETE,
                                    TaskStopReason.BLOCKED_WITHOUT_EVIDENCE,
                                }
                            ),
                            emit=self._emit,
                        )
                    ),
                ],
            ),
            name="researcher",
            # 业务状态经通道持久化，崩溃重放从最后一个已提交 superstep 续跑。
        )

    async def run(self, task: Task) -> ResearcherResult:
        """运行任务级 Agent loop，返回任务级研究结论与轨迹。"""
        task_state = ResearcherTaskState(
            active_evidence_limit=self.config.research_agent_max_evidences_per_task,
            evidence_archive_limit=(self.config.research_agent_max_evidence_candidates_per_task),
        )
        scope = AgentScope.from_task(task, agent_name="Researcher")
        loop_context = ResearcherContext(
            deps=self._deps,
            scope=scope,
            task=task,
            task_state=task_state,
            event_context={
                **scope.event_fields(),
                "worker_id": task.get("worker_id", task["id"]),
                "worker_index": int(task.get("worker_index", task.get("sequence", 0))),
            },
        )
        messages = self._initial_messages(task)
        status: Literal["completed", "failed", "cancelled"] = "completed"
        try:
            result = await cast(Any, self._agent_loop).ainvoke(
                cast(Any, {"messages": messages, **task_state.initial_channel_state()}),
                context=loop_context,
                config={"recursion_limit": AGENT_RECURSION_LIMIT},
            )
        except GraphRecursionError:
            task_state = loop_context.task_state
            task_state.stop_reason = TaskStopReason.STEP_BUDGET_EXHAUSTED
            task_state.stop_detail = "任务级 Agent 回合预算已耗尽。"
        except asyncio.CancelledError:
            status = "cancelled"
            task_state = loop_context.task_state
            task_state.stop_reason = TaskStopReason.CANCELLED
            task_state.stop_detail = "任务级 Agent 被取消。"
            raise
        except Exception as exc:  # noqa: BLE001 - 任务级边界，异常转成 failed 状态继续收尾
            status = "failed"
            task_state = loop_context.task_state
            task_state.failures.append(f"direction_agent_failed: {exc}")
            task_state.stop_reason = TaskStopReason.TASK_AGENT_FAILED
            task_state.stop_detail = str(exc)
        else:
            # 正常收尾从通道现值重建 task_state，通道为唯一事实源。
            task_state = ResearcherTaskState.from_channel_values(
                result if isinstance(result, dict) else {},
                active_evidence_limit=self.config.research_agent_max_evidences_per_task,
                evidence_archive_limit=self.config.research_agent_max_evidence_candidates_per_task,
            )
        # 未正常收尾的路径统一套用最小保守结果，保住已验证证据并补降级标注。
        # 它只改写交付语义的 conclusion；执行真相由 status 与
        # stop_reason 如实记录，崩溃的任务就是 failed，不能标成 completed+fallback。
        if status != "cancelled" and task_state.stop_reason not in {
            TaskStopReason.COMPLETE,
            TaskStopReason.BLOCKED_WITHOUT_EVIDENCE,
        }:
            self._apply_minimum_result(task_state)
        return self._result(
            task,
            status=status,
            task_state=task_state,
        )

    @staticmethod
    def _apply_minimum_result(task_state: ResearcherTaskState) -> None:
        """不调用模型的最终降级路径：保留现有证据，明确标注自动结束与缺口。

        stop_reason 只在本无收尾原因时补写，回合耗尽对应 fallback 或 blocked;
        direction_agent_failed 是真实故障原因，保留，不能被交付层的名字覆写。
        """
        if not task_state.active_evidence_ids and task_state.evidences:
            task_state.active_evidence_ids.update(
                item.evidence_id
                for item in task_state.evidences[: task_state.active_evidence_limit]
            )
        evidences = task_state.active_evidences()
        fallback_note = "本任务未在回合限制内提交结构化总结，当前仅保留已验证材料。"
        if not evidences:
            task_state.conclusion = (
                f"{fallback_note}当前状态：没有取得可锚定的 Evidence，因此不给出结论；"
                "该方向仍是缺口，需要重新派发或调整检索范围。"
            )
            if task_state.stop_reason != TaskStopReason.TASK_AGENT_FAILED:
                task_state.stop_reason = TaskStopReason.BLOCKED_WITHOUT_EVIDENCE
        else:
            task_state.conclusion = (
                f"{fallback_note}当前状态：保留 {len(evidences)} 条可验证 Evidence，"
                "具体事实以 Evidence claim 为准，不应外推；本方向的综合结论尚未形成，"
                "仍是缺口，需要重新派发才能补齐。"
            )
            if task_state.stop_reason != TaskStopReason.TASK_AGENT_FAILED:
                task_state.stop_reason = TaskStopReason.FALLBACK_COMPLETE
        if task_state.stop_reason != TaskStopReason.TASK_AGENT_FAILED:
            task_state.stop_detail = "系统已基于现有 Evidence 生成最小保守结果。"

    def _emit(
        self,
        event_type: str,
        payload: dict[str, object],
        *,
        component: str = "researcher",
    ) -> None:
        """写入任务级 Agent 事件；事件只包含诊断元数据，不包含完整正文。"""
        emit_agent_event(
            self.event_sink,
            self.logger,
            event_type,
            payload,
            component=component,
            node_fallback="researcher",
        )

    def _initial_messages(self, task: Task) -> list[BaseMessage]:
        observation = {
            "research_direction": task["question"],
            "remaining_budget": {
                "queries": self.config.research_agent_max_queries,
                "active_evidence": self.config.research_agent_max_evidences_per_task,
                "evidence_archive": (self.config.research_agent_max_evidence_candidates_per_task),
                "turns": self.config.research_agent_max_turns,
            },
        }
        return [
            HumanMessage(
                content=(
                    render_data_section("运行时环境", get_runtime_environment().payload())
                    + "\n\n---\n\n"
                    + render_data_section("委派研究任务", {"question": task["question"]})
                )
            ),
            HumanMessage(content=render_data_section("系统研究观察", observation)),
        ]

    def _result(
        self,
        task: Task,
        *,
        status: Literal["completed", "failed", "cancelled"],
        task_state: ResearcherTaskState,
    ) -> ResearcherResult:
        active_evidences = task_state.active_evidences()
        active_sources = list(
            dict.fromkeys(item.source_url for item in active_evidences if item.source_url)
        )
        task_result = TaskResult(
            task_id=task["id"],
            round=int(task.get("round", 1)),
            task_index=int(task.get("sequence", 0)),
            question=task["question"],
            research_direction=task["question"],
            display_title=str(task.get("display_title") or ""),
            execution_status=status,
            coverage_status=(
                "sufficient"
                if task_state.stop_reason == TaskStopReason.COMPLETE and active_evidences
                else "partial"
                if active_evidences
                else "insufficient"
            ),
            evidence_count=len(active_evidences),
            source_count=len(active_sources),
            conclusion=task_state.conclusion,
            finding=task_state.finding,
            queries=task_state.queries,
            read_urls=task_state.read_urls,
            skip_reasons=sorted(set(task_state.skipped)),
            failures=task_state.failures[: self.config.research_failure_history_limit],
            stop_reason=task_state.stop_reason,
            stop_detail=task_state.stop_detail,
            provider_exhausted=task_state.provider_exhausted,
        )
        return ResearcherResult(
            evidences=task_state.evidences,
            selected_evidence_ids=[item.evidence_id for item in active_evidences],
            source_refs=list(dict.fromkeys(task_state.source_refs)),
            task_result=task_result,
        )
