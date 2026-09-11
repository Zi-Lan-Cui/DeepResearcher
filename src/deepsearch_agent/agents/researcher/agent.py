"""方向级自主研究 Agent：在全局预算内完成一个受派方向的局部研究闭环。"""

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from typing import Any, Literal, cast

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.errors import GraphRecursionError

from deepsearch_agent.agents.middleware import (
    AGENT_RECURSION_LIMIT,
    MiddlewareProfile,
    SubmissionGuard,
    build_agent_middleware,
)
from deepsearch_agent.agents.researcher.state import (
    DirectionRunState,
    ResearchRuntimeContext,
)
from deepsearch_agent.agents.researcher.tools import build_researcher_tools
from deepsearch_agent.config import AgentConfig, language_directive
from deepsearch_agent.context.execution import AgentExecutionScope
from deepsearch_agent.context.runtime import get_runtime_environment
from deepsearch_agent.evidence.models import Evidence
from deepsearch_agent.llm import LLMConfigurationError, LLMInvoker
from deepsearch_agent.observability.events import JsonlSink, emit_agent_event
from deepsearch_agent.observability.logger import get_logger
from deepsearch_agent.prompts import load_prompt
from deepsearch_agent.schemas import ResearchAgentResult, ResearchDirectionResult
from deepsearch_agent.state import SubTask
from deepsearch_agent.tools import SearchTool, SourceReaderTool
from deepsearch_agent.tools.search.models import (
    SearchCandidate,
    SearchResult,
    SearchToolResult,
    classify_source,
    describe_source,
)
from deepsearch_agent.tools.sources.models import SourceReaderToolResult

_RESEARCHER_SYSTEM_PROMPT = load_prompt("researcher")


class ResearchAgent:
    """自主完成一个研究方向，不决定整项研究是否已经充分。"""

    def __init__(
        self,
        llm: LLMInvoker,
        config: AgentConfig,
        *,
        search_tool: SearchTool,
        reader_tool: SourceReaderTool,
        event_sink: JsonlSink | None = None,
        context_window_tokens: int = 32_768,
    ):
        if llm is None:
            raise LLMConfigurationError("ResearchAgent 需要已装配的 LLMInvoker。")
        if search_tool is None or reader_tool is None:
            raise ValueError("ResearchAgent 需要 SearchTool 和 SourceReaderTool。")
        self.llm = llm
        self.config = config
        self.search_tool = search_tool
        self.reader_tool = reader_tool
        self.event_sink = event_sink
        self.logger = get_logger("deepsearch_agent.agents.researcher")
        self._agent_loop = create_agent(
            model=cast(Any, self.llm),
            tools=build_researcher_tools(),
            system_prompt=_RESEARCHER_SYSTEM_PROMPT
            + "\n"
            + language_directive(config.output_language),
            context_schema=ResearchRuntimeContext,
            middleware=cast(
                Any,
                build_agent_middleware(
                    MiddlewareProfile(
                        agent_name="ResearchAgent",
                        model=getattr(self.llm, "chat_model", None),
                        max_turns=self.config.research_agent_max_turns + 1,
                        context_window_tokens=context_window_tokens,
                        retry_tools=[
                            (["SearchSources"], "SearchSources"),
                            (["ReadSources"], "ReadSources"),
                        ],
                        serial_tools={
                            "SearchSources",
                            "ReadSources",
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
                            submitted_probe=lambda ctx: getattr(
                                getattr(ctx, "run_state", None), "stop_reason", None
                            )
                            in {"complete", "blocked_without_evidence"},
                            max_nudges=self.config.finalization_attempts,
                        ),
                        emit=self._emit,
                    )
                ),
            ),
            name="researcher",
        )

    async def run(
        self,
        task: SubTask,
        *,
        claim_url: Callable[[str], Awaitable[bool]],
        on_url_already_attempted: Callable[[str], None] | None = None,
    ) -> ResearchAgentResult:
        """运行方向级 Agent loop，返回方向级研究结论与轨迹。"""
        run_state = DirectionRunState(
            active_evidence_limit=self.config.research_agent_max_evidences_per_direction,
            evidence_archive_limit=(
                self.config.research_agent_max_evidence_candidates_per_direction
            ),
        )
        runtime = self._runtime_context(
            task,
            run_state,
            claim_url=claim_url,
            on_url_already_attempted=on_url_already_attempted,
        )
        messages = self._initial_messages(task)
        status: Literal["completed", "failed", "cancelled"] = "completed"
        try:
            await self._agent_loop.ainvoke(
                cast(Any, {"messages": messages}),
                context=runtime,
                config={"recursion_limit": AGENT_RECURSION_LIMIT},
            )
        except GraphRecursionError:
            run_state.stop_reason = "step_budget_exhausted"
            run_state.stop_detail = "方向级 Agent 回合预算已耗尽。"
        except asyncio.CancelledError:
            status = "cancelled"
            run_state.stop_reason = "cancelled"
            run_state.stop_detail = "方向级 Agent 被取消。"
            raise
        except Exception as exc:
            status = "failed"
            run_state.failures.append(f"direction_agent_failed: {exc}")
            run_state.stop_reason = "direction_agent_failed"
            run_state.stop_detail = str(exc)
        if status != "cancelled" and run_state.stop_reason not in {
            "complete",
            "blocked_without_evidence",
        }:
            status = "completed"
            self._apply_minimum_result(run_state)
        return self._result(
            task,
            status=status,
            run_state=run_state,
        )

    @staticmethod
    def _apply_minimum_result(run_state: DirectionRunState) -> None:
        """不调用模型的最终保险：保留现有证据，明确标注自动收束与缺口。"""
        if not run_state.active_evidence_ids and run_state.evidences:
            run_state.active_evidence_ids.update(
                item.evidence_id for item in run_state.evidences[: run_state.active_evidence_limit]
            )
        evidences = run_state.active_evidences()
        fallback_gap = "方向研究未在回合限制内提交结构化总结，当前仅保留已验证材料。"
        if not evidences:
            run_state.answered_points = []
            run_state.conclusion = ""
            run_state.remaining_gaps = list(
                dict.fromkeys([*run_state.remaining_gaps, fallback_gap, "未获得可用 Evidence。"])
            )
            run_state.stop_reason = "blocked_without_evidence"
        else:
            claims = list(dict.fromkeys(item.claim.strip() for item in evidences if item.claim.strip()))
            run_state.answered_points = claims[:4]
            run_state.conclusion = "当前已验证材料支持上述有限结论；不应外推至未覆盖范围。"
            run_state.remaining_gaps = list(
                dict.fromkeys([*run_state.remaining_gaps, fallback_gap])
            )[:4]
            run_state.stop_reason = "fallback_complete"
        run_state.stop_detail = "系统已基于现有 Evidence 生成最小保守结果。"

    def _runtime_context(
        self,
        task: SubTask,
        run_state: DirectionRunState,
        *,
        claim_url: Callable[[str], Awaitable[bool]],
        on_url_already_attempted: Callable[[str], None] | None,
    ) -> ResearchRuntimeContext:
        scope = AgentExecutionScope.from_task(task, agent_name="ResearchAgent")
        event_context = {
            **scope.event_fields(),
            "worker_id": task.get("worker_id", task["id"]),
            "worker_index": int(task.get("worker_index", task.get("sequence", 0))),
        }

        async def search_sources(queries: list[str], reason: str) -> dict[str, object]:
            return await self._search_sources(
                task, queries, reason, run_state=run_state, event_context=event_context
            )

        async def read_sources(candidate_ids: list[str], reason: str) -> dict[str, object]:
            return await self._read_sources(
                task,
                candidate_ids,
                reason,
                run_state=run_state,
                event_context=event_context,
                claim_url=claim_url,
                on_url_already_attempted=on_url_already_attempted,
            )

        return ResearchRuntimeContext(
            task=task,
            scope=scope,
            run_state=run_state,
            search_sources=search_sources,
            read_sources=read_sources,
            on_url_already_attempted=on_url_already_attempted,
            event_context=event_context,
        )

    async def _read_sources(
        self,
        task: SubTask,
        candidate_ids: list[str],
        reason: str,
        *,
        run_state: DirectionRunState,
        event_context: dict[str, object],
        claim_url: Callable[[str], Awaitable[bool]],
        on_url_already_attempted: Callable[[str], None] | None,
    ) -> dict[str, object]:
        """读取模型选中的候选来源，并返回紧凑的工具结果。"""
        del reason
        selected_ids = list(dict.fromkeys(candidate_ids))
        selected = [
            run_state.candidates[item] for item in selected_ids if item in run_state.candidates
        ]
        unknown_ids = [item for item in selected_ids if item not in run_state.candidates]
        if unknown_ids:
            run_state.failures.append(f"unknown_candidate_ids: {', '.join(unknown_ids)}")
        candidates: list[SearchCandidate] = []
        for candidate in selected:
            if candidate.candidate_id in run_state.selected_candidate_ids:
                continue
            if not await claim_url(candidate.url):
                if on_url_already_attempted:
                    on_url_already_attempted(candidate.url)
                run_state.skipped.append("url_already_attempted")
                continue
            run_state.selected_candidate_ids.add(candidate.candidate_id)
            run_state.read_urls.append(candidate.url)
            candidates.append(candidate)

        read_results = await self._read_candidates(task, candidates)
        accepted_evidence: list[Evidence] = []
        for candidate, read_result in zip(candidates, read_results, strict=True):
            url = candidate.url
            if isinstance(read_result, asyncio.CancelledError):
                raise read_result
            if isinstance(read_result, Exception):
                run_state.failures.append(f"{url}: {read_result}")
                self._emit(
                    "source_read_failed",
                    {
                        **event_context,
                        "research_direction": task["question"],
                        "url": url,
                        "error": str(read_result)[:500],
                    },
                )
                continue
            result = SourceReaderToolResult.model_validate(read_result)
            if result.status == "completed":
                remaining = run_state.evidence_archive_limit - len(run_state.evidences)
                accepted = list(result.evidences)[: max(0, remaining)]
                accepted = run_state.add_evidences(accepted)
                accepted_evidence.extend(accepted)
                if accepted and result.source_url:
                    run_state.source_refs.append(result.source_url)
            elif result.status == "skipped":
                reason_code = result.reason_code or "unknown"
                run_state.skipped.append(reason_code)
                self._emit(
                    "source_read_skipped",
                    {
                        **event_context,
                        "research_direction": task["question"],
                        "url": url,
                        "reason_code": reason_code,
                    },
                )
            else:
                error = result.error or "read_failed"
                run_state.failures.append(f"{url}: {error}")
                self._emit(
                    "source_read_failed",
                    {
                        **event_context,
                        "research_direction": task["question"],
                        "url": url,
                        "error": error,
                    },
                )
        return {
            "candidate_ids": selected_ids,
            "read_candidate_count": len(candidates),
            "unknown_candidate_ids": unknown_ids,
            "evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "claim": item.claim,
                    "quote": item.quote[: self.config.research_observation_quote_chars],
                    "source": item.source_url,
                    "source_profile": item.source_profile.model_dump(),
                    "support": item.support,
                }
                for item in accepted_evidence
            ],
            "archive_evidence_count": len(run_state.evidences),
            "active_evidence_count": len(run_state.active_evidence_ids),
            "active_evidence_limit": run_state.active_evidence_limit,
            "archive_evidence_limit": run_state.evidence_archive_limit,
            "skip_reasons": sorted(set(run_state.skipped)),
            "recent_failures": run_state.failures[-4:],
        }

    async def _search_sources(
        self,
        task: SubTask,
        proposed_queries: list[str],
        reason: str,
        *,
        run_state: DirectionRunState,
        event_context: dict[str, object],
    ) -> dict[str, object]:
        """搜索并返回候选目录；此方法不读取任何来源。"""
        del reason
        remaining = self.config.research_agent_max_queries - len(run_state.queries)
        new_queries = self._new_queries(proposed_queries, run_state.queries)[: max(0, remaining)]
        if not new_queries:
            error = "没有新的可执行检索式；请基于已有候选读取来源或调用 Complete。"
            run_state.failures.append(f"no_novel_queries: {error}")
            return {
                "status": "skipped",
                "reason": "no_novel_queries",
                "proposed_queries": proposed_queries,
            }
        run_state.queries.extend(new_queries)
        result = SearchToolResult.model_validate(
            await self.search_tool.arun_queries(task, queries=new_queries)
        )
        run_state.failures.extend(
            f"search query={item.query}: {item.error}" for item in result.failures
        )
        self._emit(
            "direction_search_completed",
            {
                **event_context,
                "research_direction": task["question"],
                "queries": new_queries,
                "status": result.status,
                "candidate_count": len(result.results),
                "failure_count": len(result.failures),
            },
        )
        if result.status != "completed":
            error = result.error or "search_failed"
            run_state.failures.append(f"search: {error}")
            return {"status": "failed", "queries": new_queries, "error": error}

        candidates: list[dict[str, object]] = []
        for item in result.results:
            url = str(item.get("url", "")).strip()
            if not url:
                continue
            candidate_id = "c-" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
            candidate = SearchCandidate(
                candidate_id=candidate_id,
                title=str(item.get("title", "")),
                url=url,
                snippet=str(item.get("snippet", "")),
                score=float(item.get("score", 0.0)),
                content_provider=str(item.get("content_provider", "")),
                published_at=str(item.get("published_at", "")),
                source_tier=classify_source(url),
                source_profile=describe_source(url),
            )
            run_state.candidates[candidate_id] = candidate
            candidates.append(candidate.model_dump())
        return {"status": "completed", "queries": new_queries, "candidates": candidates}

    def _emit(self, event_type: str, payload: dict[str, object]) -> None:
        """写入方向级 Agent 事件；事件只包含诊断元数据，不包含完整正文。"""
        emit_agent_event(
            self.event_sink,
            self.logger,
            event_type,
            payload,
            component="research_agent",
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
                    "【运行时环境】\n"
                    + json.dumps(get_runtime_environment().payload(), ensure_ascii=False)
                    + f"\n【委派研究方向】\n{task['question']}"
                )
            ),
            HumanMessage(
                content="【系统研究观察；不是用户补充】\n"
                + json.dumps(observation, ensure_ascii=False)
            ),
        ]

    async def _read_candidates(
        self, task: SubTask, candidates: list[SearchCandidate]
    ) -> list[object]:
        semaphore = asyncio.Semaphore(self.config.research_agent_read_concurrency)

        async def read_one(candidate: SearchCandidate) -> object:
            async with semaphore:
                url = candidate.url
                try:
                    result = cast(SearchResult, candidate.model_dump())
                    return await asyncio.wait_for(
                        self.reader_tool.arun(task, result),
                        timeout=self.config.source_total_timeout,
                    )
                except asyncio.TimeoutError:
                    return TimeoutError(
                        "来源读取超时 "
                        f"（超过来源总时限 {self.config.source_total_timeout:.1f}s）：{url}"
                    )

        return list(
            await asyncio.gather(
                *(read_one(candidate) for candidate in candidates), return_exceptions=True
            )
        )

    def _new_queries(self, proposed: list[str], seen: list[str]) -> list[str]:
        known = {item.casefold().strip() for item in seen}
        return list(
            dict.fromkeys(
                query.strip()[: self.config.research_query_chars]
                for query in proposed
                if query.strip() and query.casefold().strip() not in known
            )
        )

    def _result(
        self,
        task: SubTask,
        *,
        status: Literal["completed", "failed", "cancelled"],
        run_state: DirectionRunState,
    ) -> ResearchAgentResult:
        active_evidences = run_state.active_evidences()
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
                if run_state.stop_reason == "complete" and active_evidences
                else "partial"
                if active_evidences
                else "insufficient"
            ),
            evidence_count=len(active_evidences),
            source_count=len(active_sources),
            answered_points=run_state.answered_points,
            conclusion=run_state.conclusion,
            remaining_gaps=run_state.remaining_gaps,
            queries=run_state.queries,
            read_urls=run_state.read_urls,
            skip_reasons=sorted(set(run_state.skipped)),
            failures=run_state.failures[: self.config.research_failure_history_limit],
            stop_reason=run_state.stop_reason,
            stop_detail=run_state.stop_detail,
        )
        return ResearchAgentResult(
            evidences=run_state.evidences,
            selected_evidence_ids=[item.evidence_id for item in active_evidences],
            source_refs=list(dict.fromkeys(run_state.source_refs)),
            task_result=task_result,
        )
