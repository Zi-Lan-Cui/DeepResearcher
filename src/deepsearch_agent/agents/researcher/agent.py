"""方向级自主研究 Agent：在全局预算内完成一个受派方向的局部研究闭环。"""

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from deepsearch_agent.config import AgentConfig
from deepsearch_agent.context import ContextPolicy
from deepsearch_agent.evidence.models import Evidence
from deepsearch_agent.llm import LLMConfigurationError, LLMInvoker
from deepsearch_agent.observability.events import JsonlSink, make_audit_event
from deepsearch_agent.observability.logger import get_logger
from deepsearch_agent.observability.tracing.context import current_context
from deepsearch_agent.schemas import (
    ForgetEvidence,
    ReadSources,
    ReadWorkingSet,
    ResearchAgentResult,
    ResearchDirectionComplete,
    ResearchDirectionDecision,
    ResearchDirectionResult,
    SearchSources,
)
from deepsearch_agent.state import SubTask
from deepsearch_agent.tools import SearchTool, SourceReaderTool
from deepsearch_agent.tools.research_models import (
    SearchCandidate,
    SearchToolResult,
    SourceReaderToolResult,
)
from deepsearch_agent.tools.search import SearchResult

_RESEARCHER_SYSTEM_PROMPT = """【身份】你是深度研究系统中的方向级 ResearchAgent。Supervisor 已把一个具体研究方向委派给你。
【职责】你可自主选择下一轮检索式、判断该方向是否已被证据回答，或在没有可信路径时止损。
你不决定整项研究是否完成，不写最终报告，也不把原任务原样交给搜索引擎。
【行动】调用 SearchSources 请求一到两条短、可直接搜索的检索式；观察候选目录后，调用 ReadSources
选择真正需要读取的候选来源。可调用 ReadWorkingSet 查看当前已保留 Evidence 的摘要；材料过多或偏题时，
调用 ForgetEvidence 释放当前工作集中的 Evidence，再继续读取。最后调用 ResearchDirectionComplete 宣布本方向结束。
检索式必须针对当前缺口，不能重复历史查询，也不能扩展到 Supervisor 未委派的对象。
SearchSources 只发现来源，不会自动读取；只有 ReadSources 选择的来源才会抓取和抽取 Evidence。
ReadWorkingSet 只返回当前工作集摘要，不返回完整 quote；ForgetEvidence 只释放当前方向的工作集，不删除全局档案。
Complete 只表示你已完成本方向的有界执行，不能表示整项研究完成。
【标准】Evidence 的 claim/quote 才是事实基础；搜索标题、失败 URL 和常识不能充当证据。
你可因来源被拦截而换术语、语言、资料类型或缩小到可验证子问题，但不得虚构来源。
【完成标准】只有当当前方向已经获得足以支撑局部问题的 Evidence，或预算/来源条件已经没有合理的下一步时，
才调用 ResearchDirectionComplete。answered_points 和 conclusion 只能总结当前 Evidence；remaining_gaps
只是给 Supervisor 的局部线索，不是整项研究的全局判断。"""


@dataclass
class _DirectionRunState:
    """方向研究循环中由决策工具直接更新的状态。"""

    evidences: list[Evidence] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    read_urls: list[str] = field(default_factory=list)
    candidates: dict[str, SearchCandidate] = field(default_factory=dict)
    selected_candidate_ids: set[str] = field(default_factory=set)
    skipped: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    answered_points: list[str] = field(default_factory=list)
    remaining_gaps: list[str] = field(default_factory=list)
    conclusion: str = ""
    stop_reason: str = "step_budget_exhausted"
    stop_detail: str = "方向级探索步数预算已耗尽。"


@dataclass(frozen=True)
class ToolExecutionContext:
    """一次工具执行的上下文；各执行器共享但不修改它。"""

    task: SubTask
    decision: ResearchDirectionDecision
    tool_call_id: str
    messages: list[BaseMessage]
    run_state: _DirectionRunState
    event_context: dict[str, object]
    claim_url: Callable[[str], Awaitable[bool]]
    on_url_already_attempted: Callable[[str], None] | None


ToolExecutor = Callable[[ToolExecutionContext], Awaitable[bool]]


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
        context_policy: ContextPolicy | None = None,
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
        self.context_policy = context_policy or ContextPolicy()
        self.logger = get_logger("deepsearch_agent.agents.researcher")
        self._decision_runnable = llm.bind_tools(
            [SearchSources, ReadSources, ReadWorkingSet, ForgetEvidence, ResearchDirectionComplete],
            tool_choice="any",
            parallel_tool_calls=False,
        )
        self._tool_executors: dict[str, ToolExecutor] = {
            "search": self._execute_search,
            "read": self._execute_read,
            "inspect": self._execute_inspect,
            "forget": self._execute_forget,
            "complete": self._execute_complete,
        }

    async def run(
        self,
        task: SubTask,
        *,
        claim_url: Callable[[str], Awaitable[bool]],
        on_url_already_attempted: Callable[[str], None] | None = None,
    ) -> ResearchAgentResult:
        """运行有界 Observe → Decide → Act 循环，返回方向级研究结论与轨迹。"""
        messages = self._initial_messages(task)
        run_state = _DirectionRunState()
        event_context = {
            "task_id": task["id"],
            "worker_id": task.get("worker_id", task["id"]),
            "worker_index": int(task.get("worker_index", task.get("sequence", 0))),
            "parent_task_id": task.get("parent_task_id", ""),
            "operation_id": task.get("operation_id", task["id"]),
        }
        for _turn in range(1, self.config.research_agent_max_turns + 1):
            try:
                decision, decision_call_id = await self._decide(
                    messages,
                    task,
                    run_state=run_state,
                )
            except Exception as exc:
                run_state.failures.append(f"direction_decision_failed: {exc}")
                run_state.stop_reason = "direction_decision_failed"
                run_state.stop_detail = str(exc)
                return self._result(
                    task,
                    status="failed",
                    run_state=run_state,
                )

            execution_context = ToolExecutionContext(
                task=task,
                decision=decision,
                tool_call_id=decision_call_id,
                messages=messages,
                run_state=run_state,
                event_context=event_context,
                claim_url=claim_url,
                on_url_already_attempted=on_url_already_attempted,
            )
            executor = self._tool_executors[decision.action]
            if await executor(execution_context):
                break
        return self._result(
            task,
            status="completed",
            run_state=run_state,
        )

    @staticmethod
    def _apply_decision(
        decision: ResearchDirectionDecision,
        run_state: _DirectionRunState,
    ) -> bool:
        """应用方向决策；返回 True 表示本方向不再执行下一步。"""
        run_state.remaining_gaps = list(
            dict.fromkeys(gap.strip() for gap in decision.remaining_gaps if gap.strip())
        )
        if decision.action == "complete":
            if run_state.evidences:
                run_state.answered_points = decision.answered_points
                run_state.conclusion = decision.conclusion.strip()
                run_state.stop_reason = "complete"
                run_state.stop_detail = decision.reason
            else:
                run_state.stop_reason = "blocked_without_evidence"
                run_state.stop_detail = decision.reason
                if not run_state.remaining_gaps:
                    run_state.remaining_gaps = [
                        "方向级 Agent 在没有可验证 Evidence 时宣称完成。"
                    ]
            return True
        return False

    async def _execute_complete(self, context: ToolExecutionContext) -> bool:
        """接受 ResearchDirectionComplete，并结束当前方向。"""
        stopped = self._apply_decision(context.decision, context.run_state)
        context.messages.append(
            self._tool_message(
                "direction_decision_accepted",
                {"decision": context.decision.model_dump()},
                tool_call_id=context.tool_call_id,
                name="ResearchDirectionComplete",
            )
        )
        return stopped

    async def _execute_search(self, context: ToolExecutionContext) -> bool:
        """执行 SearchSources；搜索结果由工具消息交回模型。"""
        await self._search_sources(
            context.task,
            context.decision,
            run_state=context.run_state,
            messages=context.messages,
            tool_call_id=context.tool_call_id,
            event_context=context.event_context,
        )
        return False

    async def _execute_read(self, context: ToolExecutionContext) -> bool:
        """执行 ReadSources；仅读取模型选中的候选来源。"""
        run_state = context.run_state
        selected = [
            run_state.candidates[candidate_id]
            for candidate_id in context.decision.candidate_ids
            if candidate_id in run_state.candidates
        ]
        unknown_ids = [
            candidate_id
            for candidate_id in context.decision.candidate_ids
            if candidate_id not in run_state.candidates
        ]
        if unknown_ids:
            run_state.failures.append(f"unknown_candidate_ids: {', '.join(unknown_ids)}")
        candidates: list[SearchCandidate] = []
        for candidate in selected:
            if candidate.candidate_id in run_state.selected_candidate_ids:
                continue
            if not await context.claim_url(candidate.url):
                if context.on_url_already_attempted:
                    context.on_url_already_attempted(candidate.url)
                run_state.skipped.append("url_already_attempted")
                continue
            run_state.selected_candidate_ids.add(candidate.candidate_id)
            run_state.read_urls.append(candidate.url)
            candidates.append(candidate)

        read_results = await self._read_candidates(context.task, candidates)
        accepted_evidence: list[Evidence] = []
        for candidate, read_result in zip(candidates, read_results, strict=True):
            url = candidate.url
            if isinstance(read_result, asyncio.CancelledError):
                raise read_result
            if isinstance(read_result, Exception):
                run_state.failures.append(f"{url}: {read_result}")
                self._emit(
                    "source_read_failed",
                    {**context.event_context, "research_direction": context.task["question"], "url": url, "error": str(read_result)[:500]},
                )
                continue
            result = SourceReaderToolResult.model_validate(read_result)
            if result.status == "completed":
                remaining = self.config.research_agent_max_evidences_per_direction - len(run_state.evidences)
                accepted = list(result.evidences)[:max(0, remaining)]
                run_state.evidences.extend(accepted)
                accepted_evidence.extend(accepted)
                if accepted and result.source_url:
                    run_state.source_refs.append(result.source_url)
            elif result.status == "skipped":
                reason = result.reason_code or "unknown"
                run_state.skipped.append(reason)
                self._emit(
                    "source_read_skipped",
                    {**context.event_context, "research_direction": context.task["question"], "url": url, "reason_code": reason},
                )
            else:
                error = result.error or "read_failed"
                run_state.failures.append(f"{url}: {error}")
                self._emit(
                    "source_read_failed",
                    {**context.event_context, "research_direction": context.task["question"], "url": url, "error": error},
                )
        context.messages.append(
            self._tool_message(
                "sources_read_completed",
                {
                    "candidate_ids": context.decision.candidate_ids,
                    "read_candidate_count": len(candidates),
                    "unknown_candidate_ids": unknown_ids,
                    "evidence": [
                        {
                            "claim": item.claim,
                            "quote": item.quote[: self.config.research_observation_quote_chars],
                            "source": item.source_url,
                            "support": item.support,
                        }
                        for item in accepted_evidence
                    ],
                    "total_evidence_count": len(run_state.evidences),
                    "skip_reasons": sorted(set(run_state.skipped)),
                    "recent_failures": run_state.failures[-4:],
                },
                tool_call_id=context.tool_call_id,
                name="ReadSources",
            )
        )
        return False

    async def _execute_inspect(self, context: ToolExecutionContext) -> bool:
        """返回当前方向工作集摘要，不改变研究状态。"""
        context.messages.append(
            self._tool_message(
                "working_set_snapshot",
                self._working_set_snapshot(context.run_state),
                tool_call_id=context.tool_call_id,
                name="ReadWorkingSet",
            )
        )
        return False

    async def _execute_forget(self, context: ToolExecutionContext) -> bool:
        """释放当前方向工作集中的 Evidence，但保留其全局可追溯记录。"""
        requested = list(dict.fromkeys(context.decision.evidence_ids))
        existing_ids = {item.evidence_id for item in context.run_state.evidences}
        before = len(context.run_state.evidences)
        forgotten = set(requested) & existing_ids
        context.run_state.evidences = [
            item for item in context.run_state.evidences if item.evidence_id not in forgotten
        ]
        removed = before - len(context.run_state.evidences)
        context.messages.append(
            self._tool_message(
                "working_set_updated",
                {
                    "forgotten_evidence_ids": [
                        evidence_id
                        for evidence_id in requested
                        if evidence_id in forgotten
                    ],
                    "unknown_evidence_ids": [
                        evidence_id
                        for evidence_id in requested
                        if evidence_id not in existing_ids
                    ],
                    "removed_count": removed,
                    **self._working_set_snapshot(context.run_state),
                },
                tool_call_id=context.tool_call_id,
                name="ForgetEvidence",
            )
        )
        return False

    def _working_set_snapshot(self, run_state: _DirectionRunState) -> dict[str, object]:
        """构造工作集轻量快照；完整 Evidence 仍通过 ReadSources 返回。"""
        return {
            "active_evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "claim": item.claim,
                    "support": item.support,
                    "confidence": item.confidence,
                }
                for item in run_state.evidences
            ],
            "active_evidence_count": len(run_state.evidences),
            "evidence_capacity_remaining": max(
                0,
                self.config.research_agent_max_evidences_per_direction
                - len(run_state.evidences),
            ),
        }

    async def _search_sources(
        self,
        task: SubTask,
        decision: ResearchDirectionDecision,
        *,
        run_state: _DirectionRunState,
        messages: list[BaseMessage],
        tool_call_id: str,
        event_context: dict[str, object],
    ) -> None:
        """搜索并返回候选目录；此方法不读取任何来源。"""
        remaining = self.config.research_agent_max_queries - len(run_state.queries)
        new_queries = self._new_queries(decision.queries, run_state.queries)[: max(0, remaining)]
        if not new_queries:
            error = "没有新的可执行检索式；请基于已有候选读取来源或调用 Complete。"
            run_state.failures.append(f"no_novel_queries: {error}")
            messages.append(
                self._tool_message(
                    "search_skipped",
                    {"reason": "no_novel_queries", "proposed_queries": decision.queries},
                    tool_call_id=tool_call_id,
                    name="SearchSources",
                )
            )
            return
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
            messages.append(
                self._tool_message(
                    "search_failed",
                    {"queries": new_queries, "error": error},
                    tool_call_id=tool_call_id,
                    name="SearchSources",
                )
            )
            return

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
            )
            run_state.candidates[candidate_id] = candidate
            candidates.append(candidate.model_dump())
        messages.append(
            self._tool_message(
                "search_sources_completed",
                {"queries": new_queries, "candidates": candidates},
                tool_call_id=tool_call_id,
                name="SearchSources",
            )
        )

    async def _decide(
        self,
        messages: list[BaseMessage],
        task: SubTask,
        *,
        run_state: _DirectionRunState,
    ) -> tuple[ResearchDirectionDecision, str]:
        """基于已有消息历史做决策，并保留模型原始 AIMessage。"""
        observation = {
            "research_direction": task["question"],
            "remaining_budget": {
                "queries": max(0, self.config.research_agent_max_queries - len(run_state.queries)),
                "sources_read": len(run_state.read_urls),
                "evidence": max(
                    0,
                    self.config.research_agent_max_evidences_per_direction
                    - len(run_state.evidences),
                ),
            },
        }
        messages.append(
            HumanMessage(
                content=(
                    "【系统研究观察；不是用户补充】\n" + json.dumps(observation, ensure_ascii=False)
                )
            )
        )
        prepared_messages = await self.context_policy.aprepare(messages, agent="researcher")
        response = await self._decision_runnable.ainvoke(prepared_messages)
        messages.append(response)
        decision = self._parse_direction_decision(response)
        call_id = str(response.tool_calls[0].get("id", ""))
        if not call_id:
            raise ValueError("ResearchAgent 工具调用缺少 id。")
        return decision, call_id

    def _emit(self, event_type: str, payload: dict[str, object]) -> None:
        """写入方向级 Agent 事件；事件只包含诊断元数据，不包含完整正文。"""
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
                node_id=context.node_id if context else "research_agent",
                component="research_agent",
                payload=payload,
            )
        )

    @staticmethod
    def _parse_direction_decision(response: AIMessage) -> ResearchDirectionDecision:
        """把方向级工具调用归一化为内部决策，不让自由文本进入执行循环。"""
        calls = response.tool_calls or []
        if len(calls) != 1:
            raise ValueError("ResearchAgent 必须且只能调用一个方向决策工具。")
        call = calls[0]
        args = call.get("args") or {}
        if call["name"] == "SearchSources":
            # 超出列表上限时保留有序的前 N 项，避免已有 Evidence 因总结字段
            # 过长而整项方向失败。
            args = {**args, "queries": list(args.get("queries") or [])[:2]}
            action = SearchSources.model_validate(args)
            return ResearchDirectionDecision(
                action="search",
                reason=action.reason,
                queries=action.queries,
            )
        if call["name"] == "ReadSources":
            args = {**args, "candidate_ids": list(args.get("candidate_ids") or [])[:8]}
            action = ReadSources.model_validate(args)
            return ResearchDirectionDecision(
                action="read",
                reason=action.reason,
                candidate_ids=action.candidate_ids,
            )
        if call["name"] == "ReadWorkingSet":
            action = ReadWorkingSet.model_validate(args)
            return ResearchDirectionDecision(action="inspect", reason=action.reason)
        if call["name"] == "ForgetEvidence":
            args = {**args, "evidence_ids": list(args.get("evidence_ids") or [])[:8]}
            action = ForgetEvidence.model_validate(args)
            return ResearchDirectionDecision(
                action="forget", reason=action.reason, evidence_ids=action.evidence_ids
            )
        if call["name"] == "ResearchDirectionComplete":
            args = {
                **args,
                "answered_points": list(args.get("answered_points") or [])[:4],
                "remaining_gaps": list(args.get("remaining_gaps") or [])[:4],
            }
            action = ResearchDirectionComplete.model_validate(args)
            return ResearchDirectionDecision(
                action="complete",
                reason=action.reason,
                answered_points=action.answered_points,
                conclusion=action.conclusion,
                remaining_gaps=action.remaining_gaps,
            )
        raise ValueError(f"ResearchAgent 调用了未知决策工具：{call['name']}")

    @staticmethod
    def _initial_messages(task: SubTask) -> list[BaseMessage]:
        return [
            SystemMessage(content=_RESEARCHER_SYSTEM_PROMPT),
            HumanMessage(content=f"【委派研究方向】\n{task['question']}"),
        ]

    @staticmethod
    def _tool_message(
        event: str,
        payload: dict[str, object],
        *,
        tool_call_id: str,
        name: str,
    ) -> ToolMessage:
        return ToolMessage(
            content=(
                "【系统工具执行结果；不是用户补充】\n"
                + json.dumps({"event": event, **payload}, ensure_ascii=False)
            ),
            name=name,
            tool_call_id=tool_call_id,
        )

    async def _read_candidates(self, task: SubTask, candidates: list[SearchCandidate]) -> list[object]:
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
        run_state: _DirectionRunState,
    ) -> ResearchAgentResult:
        task_result = ResearchDirectionResult(
            task_id=task["id"],
            round=int(task.get("round", 1)),
            task_index=int(task.get("sequence", 0)),
            question=task["question"],
            research_direction=task["question"],
            execution_status=status,
            coverage_status=(
                "sufficient"
                if run_state.stop_reason == "complete" and run_state.evidences
                else "partial"
                if run_state.evidences
                else "insufficient"
            ),
            evidence_count=len(run_state.evidences),
            source_count=len(run_state.source_refs),
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
            source_refs=list(dict.fromkeys(run_state.source_refs)),
            task_result=task_result,
        )
