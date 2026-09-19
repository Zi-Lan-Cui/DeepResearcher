"""方向级自主研究 Agent：在全局预算内完成一个受派方向的局部研究闭环。"""

import asyncio
import hashlib
from typing import Any, Literal, cast

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.errors import GraphRecursionError

from deepresearcher.agents.middleware import (
    AGENT_RECURSION_LIMIT,
    MiddlewareProfile,
    SubmissionGuard,
    build_agent_middleware,
)
from deepresearcher.agents.researcher.state import (
    DirectionRunState,
    ResearchRuntimeContext,
    evidence_observation_card,
)
from deepresearcher.agents.researcher.tools import build_researcher_tools
from deepresearcher.config import AgentConfig, language_directive
from deepresearcher.context.execution import AgentExecutionScope
from deepresearcher.context.runtime import get_runtime_environment
from deepresearcher.evidence.models import Evidence
from deepresearcher.evidence.validator import (
    normalize_text,
    quote_in_source,
    quote_matches_ignoring_punctuation,
    quote_verbatim_strict,
)
from deepresearcher.llm import LLMConfigurationError, LLMInvoker
from deepresearcher.observability.events import JsonlSink, emit_agent_event
from deepresearcher.observability.logger import get_logger
from deepresearcher.prompts import load_prompt, render_data_section
from deepresearcher.schemas import ResearchAgentResult, ResearchDirectionResult
from deepresearcher.schemas.limits import (
    SEARCH_RESULT_SNIPPET_PREVIEW_CHARS,
    SEARCH_RESULTS_PREVIEW_COUNT,
)
from deepresearcher.state import SubTask
from deepresearcher.tools import SearchTool, SourceReaderTool
from deepresearcher.tools.web.documents import DocumentView
from deepresearcher.tools.web.fetch.models import SourceReaderToolResult
from deepresearcher.tools.web.materials import ResearchMaterialStore, SearchResultSet
from deepresearcher.tools.web.search.models import (
    SearchCandidate,
    SearchResult,
    SearchToolResult,
    classify_source,
    describe_source,
)

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
        material_store: ResearchMaterialStore | None = None,
    ):
        if llm is None:
            raise LLMConfigurationError("ResearchAgent 需要已装配的 LLMInvoker。")
        if search_tool is None or reader_tool is None:
            raise ValueError("ResearchAgent 需要 SearchTool 和 SourceReaderTool。")
        self.llm = llm
        self.config = config
        self.search_tool = search_tool
        self.reader_tool = reader_tool
        self.material_store = material_store or getattr(reader_tool, "material_store", None)
        self.event_sink = event_sink
        self.logger = get_logger("deepresearcher.agents.researcher")
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
                                getattr(getattr(ctx, "run_state", None), "stop_reason", None)
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
        run_state = DirectionRunState(
            active_evidence_limit=self.config.research_agent_max_evidences_per_direction,
            evidence_archive_limit=(
                self.config.research_agent_max_evidence_candidates_per_direction
            ),
        )
        runtime = self._runtime_context(task, run_state)
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
            run_state.conclusion = ""
            run_state.remaining_gaps = list(
                dict.fromkeys([*run_state.remaining_gaps, fallback_gap, "未获得可用 Evidence。"])
            )
            run_state.stop_reason = "blocked_without_evidence"
        else:
            run_state.conclusion = (
                "本方向未完成模型综合；仅交付已选中的可验证 Evidence，"
                "具体事实以 Evidence claim 为准，不应外推。"
            )
            run_state.remaining_gaps = list(
                dict.fromkeys([*run_state.remaining_gaps, fallback_gap])
            )
            run_state.stop_reason = "fallback_complete"
        run_state.stop_detail = "系统已基于现有 Evidence 生成最小保守结果。"

    def _runtime_context(
        self,
        task: SubTask,
        run_state: DirectionRunState,
    ) -> ResearchRuntimeContext:
        scope = AgentExecutionScope.from_task(task, agent_name="ResearchAgent")
        event_context = {
            **scope.event_fields(),
            "worker_id": task.get("worker_id", task["id"]),
            "worker_index": int(task.get("worker_index", task.get("sequence", 0))),
        }
        evidence_commit_lock = asyncio.Lock()

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
            )

        async def list_search_results(
            search_id: str, offset: int, limit: int, reason: str
        ) -> dict[str, object]:
            return await self._list_search_results(
                task,
                search_id,
                offset,
                limit,
                reason,
                run_state=run_state,
            )

        async def grep_document(
            document_id: str,
            query: str,
            context_lines: int,
            offset: int,
            reason: str,
        ) -> dict[str, object]:
            return await self._grep_document(
                document_id,
                query,
                context_lines,
                offset,
                reason,
                run_state=run_state,
            )

        async def read_document(
            document_id: str,
            ranges: list[tuple[int, int]],
            reason: str,
        ) -> dict[str, object]:
            return await self._read_document(
                document_id,
                ranges,
                reason,
                run_state=run_state,
            )

        async def add_evidence(
            submissions: list[dict[str, object]], reason: str
        ) -> dict[str, object]:
            return await self._add_evidence(
                task,
                submissions,
                reason,
                run_state=run_state,
                event_context=event_context,
                commit_lock=evidence_commit_lock,
            )

        return ResearchRuntimeContext(
            task=task,
            scope=scope,
            run_state=run_state,
            search_sources=search_sources,
            list_search_results=list_search_results,
            read_sources=read_sources,
            grep_document=grep_document,
            read_document=read_document,
            add_evidence=add_evidence,
            event_context=event_context,
            evidence_commit_lock=evidence_commit_lock,
        )

    async def _read_sources(
        self,
        task: SubTask,
        candidate_ids: list[str],
        reason: str,
        *,
        run_state: DirectionRunState,
        event_context: dict[str, object],
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
            run_state.selected_candidate_ids.add(candidate.candidate_id)
            run_state.read_urls.append(candidate.url)
            candidates.append(candidate)

        read_results = await self._read_candidates(task, candidates)
        documents: list[DocumentView] = []
        inline_tokens = 0
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
                for document in result.documents:
                    view = document
                    if view.inline:
                        if (
                            inline_tokens + view.token_count
                            > self.config.document_inline_total_max_tokens
                        ):
                            view = view.model_copy(update={"inline": False, "content": ""})
                        else:
                            inline_tokens += view.token_count
                    run_state.documents[view.document_id] = view
                    documents.append(view)
                    if view.source_url:
                        run_state.source_refs.append(view.source_url)
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
            "documents": [item.model_dump(mode="json") for item in documents],
            "archive_evidence_count": len(run_state.evidences),
            "active_evidence_count": len(run_state.active_evidence_ids),
            "active_evidence_limit": run_state.active_evidence_limit,
            "archive_evidence_limit": run_state.evidence_archive_limit,
            "skip_reasons": sorted(set(run_state.skipped)),
            "recent_failures": run_state.failures[-4:],
        }

    async def _grep_document(
        self,
        document_id: str,
        query: str,
        context_lines: int,
        offset: int,
        reason: str,
        *,
        run_state: DirectionRunState,
    ) -> dict[str, object]:
        del reason
        if document_id not in run_state.documents:
            return {"status": "rejected", "reason": "unknown_document_id"}
        if self.material_store is None:
            return {"status": "failed", "reason": "material_store_unavailable"}
        result = await self.material_store.grep(
            document_id,
            query.strip(),
            context_lines=min(context_lines, self.config.document_grep_context_lines),
            max_matches=self.config.document_grep_max_matches,
            max_chars=self.config.document_grep_max_chars,
            offset=max(0, offset),
        )
        return {
            "status": "completed",
            "document_id": document_id,
            "query": result.query,
            "matches": [item.model_dump(mode="json") for item in result.matches],
            "match_count": len(result.matches),
            "total_matches": result.total_matches,
            "has_more": result.has_more,
            "next_offset": result.next_offset,
        }

    async def _read_document(
        self,
        document_id: str,
        ranges: list[tuple[int, int]],
        reason: str,
        *,
        run_state: DirectionRunState,
    ) -> dict[str, object]:
        del reason
        if document_id not in run_state.documents:
            return {"status": "rejected", "reason": "unknown_document_id"}
        if self.material_store is None:
            return {"status": "failed", "reason": "material_store_unavailable"}
        requested = ranges[: self.config.document_read_max_ranges]
        try:
            windows = await self.material_store.read(
                document_id,
                requested,
                max_lines=self.config.document_read_max_lines,
                max_chars=self.config.document_read_max_chars,
            )
        except ValueError as exc:
            return {"status": "rejected", "reason": str(exc)}
        return {
            "status": "completed",
            "document_id": document_id,
            "ranges": [item.model_dump(mode="json") for item in windows],
            "truncated_range_count": max(0, len(ranges) - len(requested)),
        }

    async def _add_evidence(
        self,
        task: SubTask,
        submissions: list[dict[str, object]],
        reason: str,
        *,
        run_state: DirectionRunState,
        event_context: dict[str, object],
        commit_lock: asyncio.Lock,
    ) -> dict[str, object]:
        if self.material_store is None:
            return {"status": "failed", "reason": "material_store_unavailable"}
        accepted: list[Evidence] = []
        rejected: list[dict[str, object]] = []
        duplicates: list[str] = []
        pending_ids: set[str] = set()
        candidates: list[tuple[int, str, Evidence]] = []
        accepted_via_normalization = 0  # 逐字被连字符/ligature 编码差异卡住、靠归一救回的条数
        for index, raw in enumerate(submissions):
            document_id = str(raw.get("document_id", ""))
            document = run_state.documents.get(document_id)
            if document is None:
                rejected.append({"index": index, "reason": "unknown_document_id"})
                continue
            claim = str(raw.get("claim", "")).strip()
            quote = str(raw.get("quote", "")).strip()
            try:
                source_text = await self.material_store.text(document_id)
            except (FileNotFoundError, KeyError) as exc:
                rejected.append({"index": index, "reason": str(exc)})
                continue
            # 入池不变式：quote 逐字（忽略空白 + 连字符/ligature 编码差异）出现在该来源正文里。
            # 宽松仍不过时再分一档：只差异标点/引号/破折号（词序列一致）→ quote_format_variant；
            # 词都不同 → quote_paraphrase。这样能真正区分"格式误杀"与"模型改述"。
            if not quote_in_source(source_text, quote):
                reason = (
                    "quote_format_variant"
                    if quote_matches_ignoring_punctuation(source_text, quote)
                    else "quote_paraphrase"
                )
                rejected.append({"index": index, "reason": reason})
                continue
            if not quote_verbatim_strict(source_text, quote):
                accepted_via_normalization += 1
            digest = hashlib.sha1(
                f"{document_id}\0{normalize_text(quote)}".encode("utf-8")
            ).hexdigest()[:16]
            evidence_id = f"{task['id']}-ev-{digest}"
            if evidence_id in pending_ids:
                duplicates.append(evidence_id)
                continue
            requested_support = str(raw.get("support", "direct"))
            support = self._bounded_support(requested_support, document.support_ceiling)
            confidence_value = raw.get("confidence", 0.0)
            if not isinstance(confidence_value, (int, float, str)):
                rejected.append({"index": index, "reason": "invalid_confidence"})
                continue
            try:
                confidence = float(confidence_value)
            except (TypeError, ValueError):
                rejected.append({"index": index, "reason": "invalid_confidence"})
                continue
            if not 0.0 <= confidence <= 1.0:
                rejected.append({"index": index, "reason": "invalid_confidence"})
                continue
            evidence = Evidence(
                evidence_id=evidence_id,
                subtask_id=task["id"],
                research_direction=task["question"],
                claim=claim,
                quote=quote,
                source_url=document.source_url,
                source_title=document.title,
                published_at=document.published_at,
                source_profile=describe_source(document.source_url),
                retrieval_method=cast(Any, document.retrieval_method),
                support=cast(Any, support),
                confidence=confidence,
            )
            candidates.append((index, document_id, evidence))
            pending_ids.add(evidence_id)

        ranked = sorted(candidates, key=lambda item: item[2].confidence, reverse=True)
        selected = ranked[: self.config.evidence_add_batch_size]
        # 原文读取和引用校验可并发；只有去重、容量与入池是短临界区。
        async with commit_lock:
            existing_ids = {item.evidence_id for item in run_state.evidences}
            per_source = {
                source_url: sum(1 for item in run_state.evidences if item.source_url == source_url)
                for source_url in {document.source_url for document in run_state.documents.values()}
            }
            for index, document_id, evidence in selected:
                if evidence.evidence_id in existing_ids:
                    duplicates.append(evidence.evidence_id)
                    continue
                document = run_state.documents[document_id]
                source_count = per_source.get(document.source_url, 0)
                if source_count >= self.config.evidence_max_per_source:
                    rejected.append({"index": index, "reason": "source_evidence_limit_reached"})
                    continue
                added = run_state.add_evidences([evidence])
                if not added:
                    rejected.append({"index": index, "reason": "evidence_archive_full"})
                    continue
                accepted.extend(added)
                existing_ids.add(evidence.evidence_id)
                per_source[document.source_url] = source_count + 1
        rejected_reasons: dict[str, int] = {}
        for item in rejected:
            key = str(item.get("reason", ""))[:40]
            rejected_reasons[key] = rejected_reasons.get(key, 0) + 1
        self._emit(
            "direction_evidence_added",
            {
                **event_context,
                "accepted_count": len(accepted),
                "rejected_count": len(rejected),
                "duplicate_count": len(duplicates),
                # 直方图归因证据为何被丢；quote_paraphrase=改述（宽松也不过）；
                # 另有 accepted_via_normalization 记"逐字但被连字符/ligature 卡、靠归一救回"的条数。
                "rejected_reasons": rejected_reasons,
                "accepted_via_normalization": accepted_via_normalization,
                # 模型提交证据时的理由：留作审计/归因的可解释信号，不再静默丢弃。
                "reason": reason.strip()[:400],
            },
        )
        return {
            "status": "completed" if accepted else "rejected",
            "accepted": [
                evidence_observation_card(
                    item, quote_chars=self.config.research_observation_quote_chars
                )
                for item in accepted
            ],
            "rejected": rejected,
            "duplicate_evidence_ids": duplicates,
            "truncated_submission_count": max(0, len(ranked) - len(selected)),
            "archive_evidence_count": len(run_state.evidences),
            "active_evidence_count": len(run_state.active_evidence_ids),
        }

    @staticmethod
    def _bounded_support(requested: str, ceiling: str) -> str:
        levels = ("insufficient", "partial", "direct")
        requested_index = levels.index(requested) if requested in levels else 0
        ceiling_index = levels.index(ceiling) if ceiling in levels else 0
        return levels[min(requested_index, ceiling_index)]

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
            if getattr(result, "provider_exhausted", False):
                # 把"原因 + 下一步"写清楚交给模型自然收尾，不加控制流分支：
                # 熔断后系统级不可用，重试/换向无意义。
                run_state.provider_exhausted = True
                return {
                    "status": "failed",
                    "provider_exhausted": True,
                    "user_code": result.provider_user_code,
                    "queries": new_queries,
                    "error": error,
                    "instruction": (
                        "搜索服务账户级不可用（额度耗尽或密钥无效），系统性、非本方向偶发。"
                        "不要再重试、换词或扩大范围。若已读到可用原文，立即调用 "
                        "ResearchDirectionComplete 提交已覆盖内容并把缺口写入 remaining_gaps；"
                        "若尚无原文，也直接 Complete 并标注证据不足。"
                    ),
                }
            return {"status": "failed", "queries": new_queries, "error": error}

        batch_key = "\0".join([task["id"], *result.queries])
        search_id = "search-" + hashlib.sha256(batch_key.encode("utf-8")).hexdigest()[:16]
        run_state.search_batches[search_id] = list(result.queries)
        if self.material_store is not None:
            await self.material_store.put_search_results(
                SearchResultSet(
                    search_id=search_id,
                    run_id=str(task.get("run_id") or task["id"]),
                    queries=list(result.queries),
                    results=[dict(item) for item in result.results],
                )
            )

        for item in result.results:
            url = str(item.get("url", "")).strip()
            if not url:
                continue
            candidate_id = "c-" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
            run_state.candidates[candidate_id] = SearchCandidate(
                candidate_id=candidate_id,
                title=str(item.get("title", "")),
                url=url,
                snippet=str(item.get("snippet", "")),
                raw_content=str(item.get("raw_content", "")),
                score=float(item.get("score", 0.0)),
                content_provider=str(item.get("content_provider", "")),
                published_at=str(item.get("published_at", "")),
                source_tier=classify_source(url),
                source_profile=describe_source(url),
            )
        raw_results = [dict(item) for item in result.results]
        # 预览与 ListSearchResults 共用同一卡片形状 + 同一 raw_results 下标偏移口径。
        preview, next_offset = self._candidate_cards(
            run_state, raw_results, start=0, count=SEARCH_RESULTS_PREVIEW_COUNT
        )
        return {
            "status": "completed",
            "queries": new_queries,
            "search_id": search_id,
            "result_count": len(raw_results),
            "candidates": preview,
            "next_offset": next_offset,
            "has_more": next_offset is not None,
        }

    async def _list_search_results(
        self,
        task: SubTask,
        search_id: str,
        offset: int,
        limit: int,
        reason: str,
        *,
        run_state: DirectionRunState,
    ) -> dict[str, object]:
        """只允许分页读取当前方向亲自产生的搜索结果。"""
        del reason
        queries = run_state.search_batches.get(search_id)
        if queries is None:
            return {"status": "rejected", "reason": "unknown_search_id"}
        if self.material_store is not None:
            try:
                stored = await self.material_store.get_search_results(
                    str(task.get("run_id") or task["id"]), search_id
                )
            except FileNotFoundError:
                return {"status": "failed", "reason": "search_results_expired"}
            raw_results = stored.results
        else:
            result = SearchToolResult.model_validate(
                await self.search_tool.arun_queries(task, queries=queries)
            )
            if result.status != "completed":
                return {"status": "failed", "reason": result.error or "search_cache_read_failed"}
            raw_results = [dict(item) for item in result.results]
        cards, next_offset = self._candidate_cards(
            run_state, raw_results, start=offset, count=limit
        )
        return {
            "status": "completed",
            "search_id": search_id,
            "offset": offset,
            "returned_count": len(cards),
            "result_count": len(raw_results),
            "candidates": cards,
            "next_offset": next_offset,
            "has_more": next_offset is not None,
        }

    def _candidate_cards(
        self,
        run_state: DirectionRunState,
        raw_results: list[dict[str, object]],
        *,
        start: int,
        count: int,
    ) -> tuple[list[dict[str, object]], int | None]:
        """按 raw_results 原始下标分页出候选卡；SearchSources 预览与 ListSearchResults 共用，
        保证同一候选在两处形状一致、且 offset 坐标同一（跳过空 url 不移动游标）。"""
        cards: list[dict[str, object]] = []
        for raw in raw_results[start : start + count]:
            url = str(raw.get("url", "")).strip()
            if not url:
                continue
            candidate = run_state.candidates.get(
                "c-" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
            )
            if candidate is not None:
                cards.append(self._search_candidate_card(candidate))
        next_start = start + count
        return cards, (next_start if next_start < len(raw_results) else None)

    @staticmethod
    def _search_candidate_card(candidate: SearchCandidate) -> dict[str, object]:
        """统一的候选卡形状：始终带（截断）snippet，供首屏即可判断是否值得 ReadSources。"""
        return {
            "candidate_id": candidate.candidate_id,
            "title": candidate.title,
            "url": candidate.url,
            "snippet": candidate.snippet[:SEARCH_RESULT_SNIPPET_PREVIEW_CHARS],
            "published_at": candidate.published_at,
            "source_tier": candidate.source_tier,
            "source_profile": candidate.source_profile.model_dump(mode="json"),
        }

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
                    render_data_section("运行时环境", get_runtime_environment().payload())
                    + "\n\n---\n\n"
                    + render_data_section("委派研究方向", {"question": task["question"]})
                )
            ),
            HumanMessage(content=render_data_section("系统研究观察（不是用户补充）", observation)),
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
            conclusion=run_state.conclusion,
            remaining_gaps=run_state.remaining_gaps,
            queries=run_state.queries,
            read_urls=run_state.read_urls,
            skip_reasons=sorted(set(run_state.skipped)),
            failures=run_state.failures[: self.config.research_failure_history_limit],
            stop_reason=run_state.stop_reason,
            stop_detail=run_state.stop_detail,
            provider_exhausted=run_state.provider_exhausted,
        )
        return ResearchAgentResult(
            evidences=run_state.evidences,
            selected_evidence_ids=[item.evidence_id for item in active_evidences],
            source_refs=list(dict.fromkeys(run_state.source_refs)),
            task_result=task_result,
        )
