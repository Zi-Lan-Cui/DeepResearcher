"""来源读取工具：抓取、归一并登记来源文档。"""

import asyncio
import time
from typing import cast

from deepresearcher.observability.events import JsonlSink, make_tool_event
from deepresearcher.observability.events.names import EventName
from deepresearcher.observability.scope import AgentScope
from deepresearcher.observability.span import SpanContext, current_span_context
from deepresearcher.observability.tracing.spans import span
from deepresearcher.schemas import Task
from deepresearcher.tokens import get_token_estimator
from deepresearcher.tools.errors import (
    SourceUnavailableError,
    ToolParseError,
    UnsafeUrlError,
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
        event_sink: JsonlSink | None = None,
        fetch_timeout: float = 30.0,
        parse_timeout: float = 20.0,
        material_store: ResearchMaterialStore,
        document_inline_max_tokens: int = 6_000,
    ):
        self.fetcher = fetcher
        self.event_sink = event_sink
        self.fetch_timeout = fetch_timeout
        self.parse_timeout = parse_timeout
        self.material_store = material_store
        self.document_inline_max_tokens = document_inline_max_tokens
        self.token_estimator = get_token_estimator()

    async def arun(self, task: Task, result: SearchResult) -> SourceReaderToolResult:
        """抓取、解析并登记正文，不在工具内部触发额外 LLM 抽取。"""
        started = time.perf_counter()
        execution = AgentScope.from_task(task, agent_name="Researcher")
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
        task: Task,
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
                    component="source_reader",
                    event_name=EventName.SOURCE_FETCH_STARTED,
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
            with span("fetch", kind="tool", input={"url": requested_url}):
                link = current_span_context()
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
        except Exception as exc:  # noqa: BLE001 - 抓取失败退回搜索原文
            # 抓取失败时可登记搜索提供方已经返回的原文或摘要，
            # 但仍由 Researcher 阅读后决定是否提交 Evidence。
            document = self._search_content_document(result)
            if document is not None and self.event_sink is not None:
                # 改用搜索内容成功也要记录：被拦/失败的 requested_url 与原因只存在于这里，
                # 登记事件本身分不清"正常抓取"与"搜索内容顶替"。
                self.event_sink.write(
                    make_tool_event(
                        "fetch",
                        "failed",  # 抓取本身确实失败；回退信息由 event_name 与 payload 承载
                        component="source_reader",
                        link=link,
                        error=str(exc),
                        duration_ms=(time.perf_counter() - started) * 1_000,
                        payload={
                            **task_context,
                            "requested_url": requested_url,
                            "reason_code": (
                                "ssrf_block" if isinstance(exc, UnsafeUrlError) else "fetch_failed"
                            ),
                            "fallback_retrieval_method": str(
                                cast(dict[str, object], document).get("retrieval_method", "")
                            ),
                        },
                        event_name=EventName.SOURCE_READ_FALLBACK,
                    )
                )
            if document is None:
                reason_code = (
                    exc.reason_code if isinstance(exc, SourceUnavailableError) else "fetch_failed"
                )
                if self.event_sink is not None:
                    self.event_sink.write(
                        make_tool_event(
                            "fetch",
                            "skipped" if isinstance(exc, SourceUnavailableError) else "failed",
                            component="source_reader",
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
                component="source_reader",
                event_name=EventName.SOURCE_DOCUMENT_REGISTERED,
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
        # FetchService 提供 material_fetch_key；此处仍容忍缺失，让无正文缓存需求、
        # 包括测试替身在内的轻量 fetcher 退回不缓存并继续运行，而不是直接报错。
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
