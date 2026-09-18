"""Web search tool: retrieve and cache candidate sources for a research task."""

import asyncio
import time
from urllib.parse import urldefrag, urlsplit, urlunsplit

from deepresearcher.context.execution import AgentExecutionScope
from deepresearcher.observability.events import JsonlSink, make_tool_event
from deepresearcher.observability.logger import get_logger
from deepresearcher.observability.tracing.context import SpanContext, current_span_context
from deepresearcher.observability.tracing.recorder import TraceRecorder
from deepresearcher.schemas.limits import (
    SEARCH_RESULT_TITLE_PREVIEW_CHARS,
    SEARCH_RESULTS_AUDIT_PREVIEW_COUNT,
)
from deepresearcher.state import SubTask
from deepresearcher.tools.cache_keys import normalize_text, semantic_cache_key
from deepresearcher.tools.errors import ProviderExhaustedError, ToolConfigurationError
from deepresearcher.tools.web.search.client import SearchClient
from deepresearcher.tools.web.search.models import (
    SearchFailure,
    SearchResult,
    SearchToolResult,
    failed_search,
)


class SearchTool:
    """执行确定性的候选来源检索，不负责研究方向或结论判断。"""

    def __init__(
        self,
        client: SearchClient,
        *,
        trace_recorder: TraceRecorder | None = None,
        event_sink: JsonlSink | None = None,
    ):
        if client is None:
            raise ToolConfigurationError("SearchTool 需要已配置的 SearchClient。")
        self.client = client
        self.trace_recorder = trace_recorder
        self.event_sink = event_sink
        self.logger = get_logger("deepresearcher.tools.web_search")
        self._query_cache: dict[str, list[SearchResult]] = {}
        self._query_cache_lock = asyncio.Lock()

    async def arun(self, task: SubTask) -> SearchToolResult:
        return await self.arun_queries(task, queries=[task["question"]])

    async def arun_queries(self, task: SubTask, *, queries: list[str]) -> SearchToolResult:
        started = time.perf_counter()
        link: SpanContext | None = None
        search_queries = list(
            dict.fromkeys(query.strip() for query in queries if query.strip())
        ) or [task["question"]]
        execution = AgentExecutionScope.from_task(task, agent_name="ResearchAgent")
        task_context = {
            **execution.event_fields(),
            "worker_id": task.get("worker_id", task["id"]),
            "worker_index": int(task.get("worker_index", task.get("sequence", 0))),
            "provider": getattr(self.client, "provider_name", "unknown"),
            "effective_limit": getattr(self.client, "effective_limit", None),
        }
        if self.event_sink is not None:
            self.event_sink.write(
                make_tool_event(
                    "search",
                    "started",
                    payload={**task_context, "queries": search_queries},
                )
            )
        try:
            cached_queries: dict[str, list[SearchResult]] = {}
            missing_queries: list[str] = []
            async with self._query_cache_lock:
                for query in search_queries:
                    local_key = self._cache_key(query)
                    if local_key in self._query_cache:
                        cached_queries[query] = self._query_cache[local_key]
                    else:
                        missing_queries.append(query)

            async def search_one(query: str) -> list[SearchResult]:
                return list(await self.client.asearch(query))

            if self.trace_recorder is not None:
                with self.trace_recorder.span("search", kind="tool"):
                    link = (
                        current_span_context()
                    )  # search 事件归属 tool span,即使写出点在 with 之外
                    batches = await asyncio.gather(
                        *(search_one(query) for query in missing_queries),
                        return_exceptions=True,
                    )
            else:
                batches = await asyncio.gather(
                    *(search_one(query) for query in missing_queries),
                    return_exceptions=True,
                )
            if any(isinstance(item, asyncio.CancelledError) for item in batches):
                raise asyncio.CancelledError()
            failures = [
                SearchFailure(query=query, error=str(batch)[:500])
                for query, batch in zip(missing_queries, batches, strict=True)
                if isinstance(batch, BaseException)
            ]
            fetched_batches = [batch for batch in batches if isinstance(batch, list)]
            # 账户级不可用优先于"这次没搜到"：直接上抛专用错误，让 failed_search 打标志，
            # 上层据此停止重试、快速收尾（而不是把每条 query 当普通空结果反复试）。
            exhausted = next((b for b in batches if isinstance(b, ProviderExhaustedError)), None)
            if exhausted is not None:
                raise exhausted
            async with self._query_cache_lock:
                for query, batch in zip(missing_queries, batches, strict=True):
                    if isinstance(batch, list):
                        self._query_cache[self._cache_key(query)] = list(batch)
            results = [
                result
                for result_set in [*cached_queries.values(), *fetched_batches]
                for result in result_set
            ]
            if not results and failures:
                raise RuntimeError("；".join(f"{item.query}: {item.error}" for item in failures))
            ranked = self._rank_and_dedupe(results)
            self.logger.info(
                "search_completed task=%s provider=%s effective_limit=%s queries=%d candidates=%d queries=%r",
                task["id"],
                task_context["provider"],
                task_context["effective_limit"],
                len(search_queries),
                len(ranked),
                search_queries,
            )
            if self.event_sink is not None:
                self.event_sink.write(
                    make_tool_event(
                        "search",
                        "completed",
                        link=link,
                        duration_ms=(time.perf_counter() - started) * 1000,
                        payload={
                            **task_context,
                            "queries": search_queries,
                            "candidate_count": len(ranked),
                            "failed_query_count": len(failures),
                            "failed_queries": [item.model_dump() for item in failures],
                            "cache_hit_count": len(cached_queries),
                            "candidates": [
                                {
                                    "title": item.get("title", "")[:SEARCH_RESULT_TITLE_PREVIEW_CHARS],
                                    "url": item.get("url", ""),
                                    "score": item.get("score", 0.0),
                                }
                                for item in ranked[:SEARCH_RESULTS_AUDIT_PREVIEW_COUNT]
                            ],
                        },
                    )
                )
            return SearchToolResult(
                task_id=task["id"],
                status="completed",
                results=ranked,
                queries=search_queries,
                failures=failures,
            )
        except Exception as exc:
            self.logger.warning("search_failed task=%s error=%s", task["id"], exc)
            if self.event_sink is not None:
                self.event_sink.write(
                    make_tool_event(
                        "search",
                        "failed",
                        link=link,
                        error=str(exc),
                        duration_ms=(time.perf_counter() - started) * 1000,
                        payload={**task_context, "queries": search_queries},
                    )
                )
            return failed_search(
                task,
                exc,
                queries=search_queries,
                failures=failures if "failures" in locals() else None,
            )

    def _cache_key(self, query: str) -> str:
        return semantic_cache_key(
            normalize_text(query),
            getattr(self.client, "provider_name", "unknown"),
            getattr(self.client, "effective_limit", None),
        )

    @staticmethod
    def _canonical_url(url: str) -> str:
        url, _ = urldefrag(url.strip())
        parts = urlsplit(url)
        return urlunsplit(
            (parts.scheme, parts.netloc.lower(), parts.path.rstrip("/"), parts.query, "")
        )

    @classmethod
    def _rank_and_dedupe(cls, results):
        by_url = {}
        for result in results:
            url = result.get("url", "")
            if not url:
                continue
            key = cls._canonical_url(url)
            current = by_url.get(key)
            if current is None or result.get("score", 0.0) > current.get("score", 0.0):
                by_url[key] = result
        return sorted(
            by_url.values(),
            key=lambda item: item.get("score", 0.0),
            reverse=True,
        )
