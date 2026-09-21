"""Source reader tool: fetch, normalize, and register source documents."""

import asyncio
import time
from typing import cast

from deepresearcher.observability.events import JsonlSink, make_tool_event
from deepresearcher.observability.execution import AgentExecutionScope
from deepresearcher.observability.tracing.context import SpanContext, current_span_context
from deepresearcher.observability.tracing.recorder import TraceRecorder
from deepresearcher.state import SubTask
from deepresearcher.tokens import get_token_estimator
from deepresearcher.tools.errors import (
    SourceUnavailableError,
    ToolConfigurationError,
    ToolParseError,
)
from deepresearcher.tools.web.documents import (
    DocumentOutlineItem,
    DocumentView,
)
from deepresearcher.tools.web.fetch.models import (
    SourceDocument,
    SourceReaderToolResult,
    failed_read,
    skipped_read,
)
from deepresearcher.tools.web.fetch.service import FetchService
from deepresearcher.tools.web.materials import ResearchMaterialStore
from deepresearcher.tools.web.parsing.models import DocumentBlock
from deepresearcher.tools.web.search.models import SearchResult
from deepresearcher.vocab import (
    RETRIEVAL_ORIGIN_FETCH,
    RETRIEVAL_SEARCH_SUMMARY,
    RETRIEVAL_TAVILY_RAW_CONTENT,
)


class SourceReaderTool:
    """抓取来源并登记正文；Evidence 只由 Researcher 显式提交。"""

    def __init__(
        self,
        fetcher: FetchService,
        *,
        trace_recorder: TraceRecorder | None = None,
        event_sink: JsonlSink | None = None,
        fetch_timeout: float = 30.0,
        parse_timeout: float = 20.0,
        material_store: ResearchMaterialStore,
        document_inline_max_tokens: int = 6_000,
    ):
        # fetcher 是 FetchService(跑整条 provider 链、带 material_fetch_key),
        # 不是单个原子 FetchProvider——两者同名协议曾让缓存契约在类型层隐形。
        if fetcher is None:
            raise ToolConfigurationError("SourceReaderTool 需要已配置的 FetchService。")
        self.fetcher = fetcher
        self.trace_recorder = trace_recorder
        self.event_sink = event_sink
        self.fetch_timeout = fetch_timeout
        self.parse_timeout = parse_timeout
        self.material_store = material_store
        self.document_inline_max_tokens = document_inline_max_tokens
        self.token_estimator = get_token_estimator()

    async def arun(self, task: SubTask, result: SearchResult) -> SourceReaderToolResult:
        """抓取、解析并登记正文，不在工具内部触发额外 LLM 抽取。"""
        started = time.perf_counter()
        execution = AgentExecutionScope.from_task(task, agent_name="ResearchAgent")
        task_context = {
            **execution.event_fields(),
            "worker_id": task.get("worker_id", task["id"]),
            "worker_index": int(task.get("worker_index", task.get("sequence", 0))),
        }
        return await self._read_into_material_store(
            task,
            result,
            started=started,
            task_context=task_context,
        )

    async def _read_into_material_store(
        self,
        task: SubTask,
        result: SearchResult,
        *,
        started: float,
        task_context: dict[str, object],
    ) -> SourceReaderToolResult:
        requested_url = str(result.get("url", ""))
        fetch_key = self._material_fetch_key(requested_url)
        link = current_span_context()
        if self.event_sink is not None:
            self.event_sink.write(
                make_tool_event(
                    "fetch",
                    "started",
                    event_name="source_fetch_started",
                    payload={**task_context, "requested_url": requested_url},
                )
            )

        if fetch_key:
            cached_ref = await self.material_store.resolve_fetch(fetch_key)
            if cached_ref is not None:
                inline = cached_ref.token_count <= self.document_inline_max_tokens
                content = await self.material_store.text(cached_ref.document_id) if inline else ""
                view = DocumentView(
                    **cached_ref.model_dump(),
                    inline=inline,
                    content=self._numbered_text(content) if inline else "",
                )
                self._emit_registered(
                    task_context,
                    requested_url=requested_url,
                    view=view,
                    started=started,
                    link=link,
                    cache_hit=True,
                )
                return SourceReaderToolResult(
                    task_id=task["id"],
                    status="completed",
                    source_url=cached_ref.source_url,
                    documents=[view],
                )

        try:
            if self.trace_recorder is not None:
                with self.trace_recorder.span("fetch", kind="tool"):
                    link = current_span_context()
                    document = await self.fetcher.afetch(
                        requested_url,
                        fetch_timeout=self.fetch_timeout,
                        parse_timeout=self.parse_timeout,
                    )
            else:
                document = await self.fetcher.afetch(
                    requested_url,
                    fetch_timeout=self.fetch_timeout,
                    parse_timeout=self.parse_timeout,
                )
            if document.get("error"):
                raise ToolParseError(str(document.get("error", "来源解析失败")))
            if not str(document.get("text", "")).strip():
                raise SourceUnavailableError("empty_content", "来源没有可读取正文。")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # 抓取失败时可登记搜索提供方已经返回的原文或摘要，
            # 但仍由 Researcher 阅读后决定是否提交 Evidence。
            document = self._search_content_document(result)
            if document is None:
                reason_code = (
                    exc.reason_code if isinstance(exc, SourceUnavailableError) else "fetch_failed"
                )
                if self.event_sink is not None:
                    self.event_sink.write(
                        make_tool_event(
                            "fetch",
                            "skipped" if isinstance(exc, SourceUnavailableError) else "failed",
                            link=link,
                            error=str(exc),
                            duration_ms=(time.perf_counter() - started) * 1_000,
                            payload={
                                **task_context,
                                "requested_url": requested_url,
                                "reason_code": reason_code,
                            },
                        )
                    )
                if isinstance(exc, SourceUnavailableError):
                    return skipped_read(
                        task,
                        source_url=requested_url,
                        reason_code=reason_code,
                        reason=str(exc),
                    )
                return failed_read(task, exc)

        published_at = str(result.get("published_at", "")).strip()
        if published_at:
            document = cast(SourceDocument, {**document, "published_at": published_at})
        text = str(document.get("text", "")).strip()
        token_count = self.token_estimator.count(text)
        put_args = {
            "text": text,
            "title": str(document.get("title") or result.get("title", "")),
            "source_url": str(document.get("final_url") or requested_url),
            "published_at": published_at,
            "retrieval_method": str(document.get("retrieval_method", RETRIEVAL_ORIGIN_FETCH)),
            "support_ceiling": str(document.get("support_ceiling", "direct")),
            "token_count": token_count,
            "outline": self._document_outline(
                text,
                cast(list[DocumentBlock], document.get("blocks", [])),
            ),
        }
        ref = await self.material_store.put(**put_args, fetch_key=fetch_key)
        inline = token_count <= self.document_inline_max_tokens
        view = DocumentView(
            **ref.model_dump(),
            inline=inline,
            content=self._numbered_text(text) if inline else "",
        )
        self._emit_registered(
            task_context,
            requested_url=requested_url,
            view=view,
            started=started,
            link=link,
            cache_hit=False,
        )
        return SourceReaderToolResult(
            task_id=task["id"],
            status="completed",
            source_url=ref.source_url,
            documents=[view],
        )

    def _emit_registered(
        self,
        task_context: dict[str, object],
        *,
        requested_url: str,
        view: DocumentView,
        started: float,
        link: SpanContext,
        cache_hit: bool,
    ) -> None:
        if self.event_sink is None:
            return
        self.event_sink.write(
            make_tool_event(
                "fetch",
                "completed",
                event_name="source_document_registered",
                link=link,
                duration_ms=(time.perf_counter() - started) * 1_000,
                payload={
                    **task_context,
                    "requested_url": requested_url,
                    "final_url": view.source_url,
                    "document_id": view.document_id,
                    "line_count": view.line_count,
                    "token_count": view.token_count,
                    "inline": view.inline,
                    "material_cache_hit": cache_hit,
                },
            )
        )

    def _material_fetch_key(self, url: str) -> str:
        # FetchService 提供 material_fetch_key;此处仍容错缺失,让无正文缓存需求的
        # 轻量 fetcher(含测试替身)退回"不缓存"而非当场炸。
        key_builder = getattr(self.fetcher, "material_fetch_key", None)
        return str(key_builder(url)) if callable(key_builder) else ""

    @staticmethod
    def _search_content_document(result: SearchResult) -> SourceDocument | None:
        raw_content = str(result.get("raw_content", "")).strip()
        snippet = str(result.get("snippet", "")).strip()
        if raw_content:
            text, method, ceiling = raw_content, RETRIEVAL_TAVILY_RAW_CONTENT, "direct"
        elif snippet:
            text, method, ceiling = snippet, RETRIEVAL_SEARCH_SUMMARY, "partial"
        else:
            return None
        return {
            "title": str(result.get("title", "")),
            "final_url": str(result.get("url", "")),
            "text": text,
            "blocks": [],
            "retrieval_method": method,
            "support_ceiling": ceiling,
        }

    @staticmethod
    def _numbered_text(text: str) -> str:
        return "\n".join(f"L{index}: {line}" for index, line in enumerate(text.splitlines(), 1))

    @staticmethod
    def _document_outline(
        text: str,
        blocks: list[DocumentBlock],
        *,
        limit: int = 64,
    ) -> list[DocumentOutlineItem]:
        """由解析器已有 heading 块生成紧凑行号目录，不额外调用模型。"""
        lines = text.splitlines()
        outline: list[DocumentOutlineItem] = []
        search_from = 0
        for block in blocks:
            if block.get("block_type") != "heading":
                continue
            heading = str(block.get("text", "")).strip()
            if not heading:
                continue
            matched = next(
                (index for index in range(search_from, len(lines)) if heading in lines[index]),
                None,
            )
            line = (matched + 1) if matched is not None else 1
            search_from = matched + 1 if matched is not None else search_from
            outline.append(DocumentOutlineItem(heading=heading, line=line))
            if len(outline) >= limit:
                break
        return outline
