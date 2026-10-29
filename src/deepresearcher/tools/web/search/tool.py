"""网络搜索工具：为研究任务检索并缓存候选来源。"""

import asyncio
import time
from collections import Counter

from deepresearcher.errors import clip_text
from deepresearcher.observability.events import JsonlSink, make_tool_event
from deepresearcher.observability.events.names import EventName
from deepresearcher.observability.logging_config import get_logger
from deepresearcher.observability.scope import AgentScope
from deepresearcher.observability.span import SpanContext, current_span_context
from deepresearcher.observability.tracing.spans import span
from deepresearcher.schemas import Task
from deepresearcher.schemas.limits import (
    SEARCH_RESULT_TITLE_PREVIEW_CHARS,
    SEARCH_RESULTS_AUDIT_PREVIEW_COUNT,
)
from deepresearcher.tools.cache_keys import canonical_url, normalize_text, semantic_cache_key
from deepresearcher.tools.errors import ProviderExhaustedError
from deepresearcher.tools.web.search.models import (
    SearchFailure,
    SearchResult,
    SearchToolResult,
    failed_search,
)
from deepresearcher.tools.web.search.service import SearchService


class SearchTool:
    """执行确定性的候选来源检索，不负责研究任务或结论判断。"""

    def __init__(
        self,
        client: SearchService,
        *,
        event_sink: JsonlSink | None = None,
    ):
        self.client = client
        self.event_sink = event_sink
        self.logger = get_logger("deepresearcher.tools.web_search")
        # build_graph 每个 run 新建实例，因此查询缓存只在单个 run 内有效，随 run 结束回收。
        self._query_cache: dict[str, list[SearchResult]] = {}
        self._query_cache_lock = asyncio.Lock()

    async def arun(self, task: Task) -> SearchToolResult:
        return await self.arun_queries(task, queries=[task["question"]])

    async def arun_queries(self, task: Task, *, queries: list[str]) -> SearchToolResult:
        started = time.perf_counter()
        link: SpanContext | None = None
        # 提前绑定空列表：早期异常路径进入 except 时也要能引用它。
        failures: list[SearchFailure] = []
        search_queries = list(
            dict.fromkeys(query.strip() for query in queries if query.strip())
        ) or [task["question"]]
        execution = AgentScope.from_task(task, agent_name="Researcher")
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
                    component="search_tool",
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
                # 每个 query 单独发起止事件，使进度能在整批 gather 返回前持续输出。
                if self.event_sink is not None:
                    self.event_sink.write(
                        make_tool_event(
                            "search",
                            "started",
                            component="search_tool",
                            link=link,
                            event_name=EventName.SEARCH_QUERY_STARTED,
                            payload={**task_context, "query": query},
                        )
                    )
                try:
                    results = list(await self.client.asearch(query))
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if self.event_sink is not None:
                        self.event_sink.write(
                            make_tool_event(
                                "search",
                                "failed",
                                component="search_tool",
                                link=link,
                                event_name=EventName.SEARCH_QUERY_FAILED,
                                error=str(exc)[:200],
                                payload={**task_context, "query": query},
                            )
                        )
                    raise
                if self.event_sink is not None:
                    self.event_sink.write(
                        make_tool_event(
                            "search",
                            "completed",
                            component="search_tool",
                            link=link,
                            event_name=EventName.SEARCH_QUERY_COMPLETED,
                            payload={
                                **task_context,
                                "query": query,
                                # 降级时实际响应的供应商，覆盖任务级的首选值。
                                "provider": _content_provider(results, task_context["provider"]),
                                "candidate_count": len(results),
                            },
                        )
                    )
                return results

            with span(
                "search",
                kind="tool",
                input={"queries": [query[:80] for query in missing_queries[:10]]},
            ):
                # 在此捕获上下文：搜索事件的写出点位于 with 之外，仍需归属本 tool span。
                link = current_span_context()
                batches = await asyncio.gather(
                    *(search_one(query) for query in missing_queries),
                    return_exceptions=True,
                )
            if any(isinstance(item, asyncio.CancelledError) for item in batches):
                raise asyncio.CancelledError()
            failures = [
                SearchFailure(query=query, error=clip_text(str(batch)))
                for query, batch in zip(missing_queries, batches, strict=True)
                if isinstance(batch, BaseException)
            ]
            fetched_batches = [batch for batch in batches if isinstance(batch, list)]
            # 账户级不可用优先于普通没搜到结果：直接上抛专用错误，让 failed_search 打标志，
            # 上层据此停止重试、快速收尾，不再把每条 query 当普通空结果反复重试。
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
                _content_provider(ranked, task_context["provider"]),
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
                        component="search_tool",
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
                                    "title": item.get("title", "")[
                                        :SEARCH_RESULT_TITLE_PREVIEW_CHARS
                                    ],
                                    "url": item.get("url", ""),
                                    "score": item.get("score", 0.0),
                                    "provider": item.get("content_provider", ""),
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
        except Exception as exc:  # noqa: BLE001 - 搜索失败写 failed 事件后继续
            self.logger.warning("search_failed task=%s error=%s", task["id"], exc)
            if self.event_sink is not None:
                self.event_sink.write(
                    make_tool_event(
                        "search",
                        "failed",
                        component="search_tool",
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
                failures=failures,
            )

    def _cache_key(self, query: str) -> str:
        return semantic_cache_key(
            normalize_text(query),
            getattr(self.client, "provider_name", "unknown"),
            getattr(self.client, "effective_limit", None),
        )

    @classmethod
    def _rank_and_dedupe(cls, results):
        by_url = {}
        for result in results:
            url = result.get("url", "")
            if not url:
                continue
            # 与缓存共用同一 canonical_url，去重时折叠尾斜杠。
            # 凭证 URL 归一为空串即无法共用身份，此时退回原串，既不错误合并也不丢失结果。
            key = canonical_url(url, strip_trailing_slash=True) or url
            current = by_url.get(key)
            if current is None or result.get("score", 0.0) > current.get("score", 0.0):
                by_url[key] = result
        return sorted(
            by_url.values(),
            key=lambda item: item.get("score", 0.0),
            reverse=True,
        )


def _content_provider(results, fallback: str) -> str:
    """本批结果贡献最多的供应商；结果为空时用 fallback（首选）。"""
    counts = Counter(
        str(item.get("content_provider")) for item in results if item.get("content_provider")
    )
    return counts.most_common(1)[0][0] if counts else fallback
