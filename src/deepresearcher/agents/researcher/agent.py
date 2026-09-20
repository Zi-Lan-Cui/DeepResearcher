"""方向级自主研究 Agent：在全局预算内完成一个受派方向的局部研究闭环。

只负责装配:稳定零件收进 ResearcherDeps,每次 run 造一个纯名词的
ResearcherLoopContext;工具协议在 tools.py,业务实现（检索/读取/校验）在 services.py。
"""

import asyncio
from typing import Any, Literal, cast

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.errors import GraphRecursionError

from deepresearcher.agents.middleware import (
    AGENT_RECURSION_LIMIT,
    MiddlewareProfile,
    SubmissionGuard,
    build_agent_middleware,
)
from deepresearcher.agents.researcher.state import (
    ResearcherDeps,
    ResearcherLoopContext,
    ResearcherLoopState,
)
from deepresearcher.agents.researcher.tools import build_researcher_tools
from deepresearcher.config import AgentConfig, language_directive
from deepresearcher.llm import LLMConfigurationError
from deepresearcher.observability.events import JsonlSink, emit_agent_event
from deepresearcher.observability.execution import AgentExecutionScope
from deepresearcher.observability.logger import get_logger
from deepresearcher.prompts import get_runtime_environment, load_prompt, render_data_section
from deepresearcher.schemas import ResearchAgentResult, ResearchDirectionResult
from deepresearcher.state import SubTask
from deepresearcher.tools import SearchTool, SourceReaderTool
from deepresearcher.tools.web.materials import ResearchMaterialStore

_RESEARCHER_SYSTEM_PROMPT = load_prompt("researcher")


class ResearchAgent:
    """自主完成一个研究方向，不决定整项研究是否已经充分。"""

    def __init__(
        self,
        llm: BaseChatModel,
        config: AgentConfig,
        *,
        search_tool: SearchTool,
        reader_tool: SourceReaderTool,
        event_sink: JsonlSink | None = None,
        context_window_tokens: int = 32_768,
        material_store: ResearchMaterialStore | None = None,
    ):
        if llm is None:
            raise LLMConfigurationError("ResearchAgent 需要已装配的模型。")
        if search_tool is None or reader_tool is None:
            raise ValueError("ResearchAgent 需要 SearchTool 和 SourceReaderTool。")
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
            context_schema=ResearcherLoopContext,
            middleware=cast(
                Any,
                build_agent_middleware(
                    MiddlewareProfile(
                        agent_name="ResearchAgent",
                        model=self.llm,
                        max_turns=self.config.research_agent_max_turns + 1,
                        context_window_tokens=context_window_tokens,
                        retry_tools=[
                            (["SearchSources"], "SearchSources"),
                            (["ReadSources"], "ReadSources"),
                        ],
                        serial_tools={
                            "SearchSources",
                            "ReadWorkingSet",
                            "ReleaseEvidence",
                            "RestoreEvidence",
                            "ResearchDirectionComplete",
                        },
                        submission_guard=SubmissionGuard(
                            nudge_message=(
                                "你还没有调用 ResearchDirectionComplete。"
                                "普通文本不是有效收尾；请继续研究，或立即调用该工具提交。"
                            ),
                            submitted_probe=lambda ctx: (
                                getattr(getattr(ctx, "loop_state", None), "stop_reason", None)
                                in {"complete", "blocked_without_evidence"}
                            ),
                            max_nudges=self.config.finalization_attempts,
                            reminder_turns=4,
                            reminder_message=(
                                "【剩余回合提醒】当前仅剩 {remaining_turns} 个模型回合。"
                                "停止扩展范围、翻页或重复搜索。请立即将已读取且可逐字定位的"
                                "原文批量提交给 AddEvidence；如被拒绝，只修正引用，不再搜索。"
                                "随后调用 "
                                "ResearchDirectionComplete 诚实提交已覆盖内容和剩余缺口。"
                            ),
                        ),
                        emit=self._emit,
                    )
                ),
            ),
            name="researcher",
        )

    async def run(self, task: SubTask) -> ResearchAgentResult:
        """运行方向级 Agent loop，返回方向级研究结论与轨迹。"""
        loop_state = ResearcherLoopState(
            active_evidence_limit=self.config.research_agent_max_evidences_per_direction,
            evidence_archive_limit=(
                self.config.research_agent_max_evidence_candidates_per_direction
            ),
        )
        scope = AgentExecutionScope.from_task(task, agent_name="ResearchAgent")
        loop_context = ResearcherLoopContext(
            deps=self._deps,
            scope=scope,
            task=task,
            loop_state=loop_state,
            event_context={
                **scope.event_fields(),
                "worker_id": task.get("worker_id", task["id"]),
                "worker_index": int(task.get("worker_index", task.get("sequence", 0))),
            },
            commit_lock=asyncio.Lock(),
        )
        messages = self._initial_messages(task)
        status: Literal["completed", "failed", "cancelled"] = "completed"
        try:
            await self._agent_loop.ainvoke(
                cast(Any, {"messages": messages}),
                context=loop_context,
                config={"recursion_limit": AGENT_RECURSION_LIMIT},
            )
        except GraphRecursionError:
            loop_state.stop_reason = "step_budget_exhausted"
            loop_state.stop_detail = "方向级 Agent 回合预算已耗尽。"
        except asyncio.CancelledError:
            status = "cancelled"
            loop_state.stop_reason = "cancelled"
            loop_state.stop_detail = "方向级 Agent 被取消。"
            raise
        except Exception as exc:
            status = "failed"
            loop_state.failures.append(f"direction_agent_failed: {exc}")
            loop_state.stop_reason = "direction_agent_failed"
            loop_state.stop_detail = str(exc)
        if status != "cancelled" and loop_state.stop_reason not in {
            "complete",
            "blocked_without_evidence",
        }:
            status = "completed"
            self._apply_minimum_result(loop_state)
        return self._result(
            task,
            status=status,
            loop_state=loop_state,
        )

    @staticmethod
    def _apply_minimum_result(loop_state: ResearcherLoopState) -> None:
        """不调用模型的最终保险：保留现有证据，明确标注自动收束与缺口。"""
        if not loop_state.active_evidence_ids and loop_state.evidences:
            loop_state.active_evidence_ids.update(
                item.evidence_id
                for item in loop_state.evidences[: loop_state.active_evidence_limit]
            )
        evidences = loop_state.active_evidences()
        fallback_gap = "方向研究未在回合限制内提交结构化总结，当前仅保留已验证材料。"
        if not evidences:
            loop_state.conclusion = ""
            loop_state.remaining_gaps = list(
                dict.fromkeys([*loop_state.remaining_gaps, fallback_gap, "未获得可用 Evidence。"])
            )
            loop_state.stop_reason = "blocked_without_evidence"
        else:
            loop_state.conclusion = (
                "本方向未完成模型综合；仅交付已选中的可验证 Evidence，"
                "具体事实以 Evidence claim 为准，不应外推。"
            )
            loop_state.remaining_gaps = list(
                dict.fromkeys([*loop_state.remaining_gaps, fallback_gap])
            )
            loop_state.stop_reason = "fallback_complete"
        loop_state.stop_detail = "系统已基于现有 Evidence 生成最小保守结果。"

    def _emit(
        self,
        event_type: str,
        payload: dict[str, object],
        *,
        component: str = "research_agent",
    ) -> None:
        """写入方向级 Agent 事件；事件只包含诊断元数据，不包含完整正文。"""
        emit_agent_event(
            self.event_sink,
            self.logger,
            event_type,
            payload,
            component=component,
            node_fallback="research_agent",
        )

    def _initial_messages(self, task: SubTask) -> list[BaseMessage]:
        observation = {
            "research_direction": task["question"],
            "remaining_budget": {
                "queries": self.config.research_agent_max_queries,
                "active_evidence": self.config.research_agent_max_evidences_per_direction,
                "evidence_archive": (
                    self.config.research_agent_max_evidence_candidates_per_direction
                ),
                "turns": self.config.research_agent_max_turns,
            },
        }
        return [
            HumanMessage(
                content=(
                    render_data_section("运行时环境", get_runtime_environment().payload())
                    + "\n\n---\n\n"
                    + render_data_section("委派研究方向", {"question": task["question"]})
                )
            ),
            HumanMessage(content=render_data_section("系统研究观察（不是用户补充）", observation)),
        ]

    def _result(
        self,
        task: SubTask,
        *,
        status: Literal["completed", "failed", "cancelled"],
        loop_state: ResearcherLoopState,
    ) -> ResearchAgentResult:
        active_evidences = loop_state.active_evidences()
        active_sources = list(
            dict.fromkeys(item.source_url for item in active_evidences if item.source_url)
        )
        task_result = ResearchDirectionResult(
            task_id=task["id"],
            round=int(task.get("round", 1)),
            task_index=int(task.get("sequence", 0)),
            question=task["question"],
            research_direction=task["question"],
            execution_status=status,
            coverage_status=(
                "sufficient"
                if loop_state.stop_reason == "complete" and active_evidences
                else "partial"
                if active_evidences
                else "insufficient"
            ),
            evidence_count=len(active_evidences),
            source_count=len(active_sources),
            conclusion=loop_state.conclusion,
            remaining_gaps=loop_state.remaining_gaps,
            queries=loop_state.queries,
            read_urls=loop_state.read_urls,
            skip_reasons=sorted(set(loop_state.skipped)),
            failures=loop_state.failures[: self.config.research_failure_history_limit],
            stop_reason=loop_state.stop_reason,
            stop_detail=loop_state.stop_detail,
            provider_exhausted=loop_state.provider_exhausted,
        )
        return ResearchAgentResult(
            evidences=loop_state.evidences,
            selected_evidence_ids=[item.evidence_id for item in active_evidences],
            source_refs=list(dict.fromkeys(loop_state.source_refs)),
            task_result=task_result,
        )
