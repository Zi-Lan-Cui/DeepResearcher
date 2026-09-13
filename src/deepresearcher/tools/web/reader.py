"""Source reader tool: fetch, parse, and extract verified Evidence from one URL."""

import asyncio
import time
from typing import cast

from deepresearcher.context.execution import AgentExecutionScope
from deepresearcher.evidence import EvidenceExtractor
from deepresearcher.llm import LLMConfigurationError, LLMInvoker
from deepresearcher.observability.events import JsonlSink, make_tool_event
from deepresearcher.observability.logger import get_logger
from deepresearcher.observability.tracing.context import SpanContext, current_span_context
from deepresearcher.observability.tracing.recorder import TraceRecorder
from deepresearcher.state import SubTask
from deepresearcher.tools.cache import ToolCache
from deepresearcher.tools.errors import (
    SourceUnavailableError,
    ToolConfigurationError,
    ToolParseError,
)
from deepresearcher.tools.web.fetch.models import (
    SourceDocument,
    SourceReaderToolResult,
    failed_read,
    skipped_read,
)
from deepresearcher.tools.web.fetch.protocol import FetchProvider
from deepresearcher.tools.web.search.models import SearchResult


class SourceReaderTool:
    """读取单个来源并产出 Evidence，不决定研究是否充分。"""

    def __init__(
        self,
        fetcher: FetchProvider,
        *,
        llm: LLMInvoker,
        trace_recorder: TraceRecorder | None = None,
        event_sink: JsonlSink | None = None,
        context_window_tokens: int = 32_768,
        evidence_input_budget_tokens: int = 24_000,
        evidence_output_budget_tokens: int = 4_000,
        evidence_safety_margin_tokens: int = 2_000,
        evidence_chunk_concurrency: int = 2,
        evidence_max_per_source: int = 2,
        fetch_timeout: float = 30.0,
        parse_timeout: float = 20.0,
        evidence_extract_timeout: float = 120.0,
        tool_cache: ToolCache | None = None,
        evidence_cache_ttl_seconds: int = 0,
        extractor_prompt_version: str = "evidence-prompt-v1",
        evidence_schema_version: str = "evidence-schema-v1",
        chunking_version: str = "chunks-v1",
        model_id: str = "",
        input_usd_per_million: float = 0.0,
        output_usd_per_million: float = 0.0,
    ):
        if fetcher is None:
            raise ToolConfigurationError("SourceReaderTool 需要已配置的 FetchProvider。")
        if llm is None:
            raise LLMConfigurationError("SourceReaderTool 需要已装配的 LLMInvoker。")
        self.fetcher = fetcher
        self.trace_recorder = trace_recorder
        self.event_sink = event_sink
        self.fetch_timeout = fetch_timeout
        self.parse_timeout = parse_timeout
        self.evidence_extract_timeout = evidence_extract_timeout
        self.extractor = EvidenceExtractor(
            llm,
            context_window_tokens=context_window_tokens,
            input_budget_tokens=evidence_input_budget_tokens,
            output_budget_tokens=evidence_output_budget_tokens,
            safety_margin_tokens=evidence_safety_margin_tokens,
            chunk_concurrency=evidence_chunk_concurrency,
            max_evidences=evidence_max_per_source,
            event_sink=event_sink,
            tool_cache=tool_cache,
            cache_ttl_seconds=evidence_cache_ttl_seconds,
            extractor_prompt_version=extractor_prompt_version,
            evidence_schema_version=evidence_schema_version,
            chunking_version=chunking_version,
            model_id=model_id,
            input_usd_per_million=input_usd_per_million,
            output_usd_per_million=output_usd_per_million,
        )
        self.logger = get_logger("deepresearcher.tools.source_reader")

    async def arun(self, task: SubTask, result: SearchResult) -> SourceReaderToolResult:
        started = time.perf_counter()
        execution = AgentExecutionScope.from_task(task, agent_name="ResearchAgent")
        task_context = {
            **execution.event_fields(),
            "worker_id": task.get("worker_id", task["id"]),
            "worker_index": int(task.get("worker_index", task.get("sequence", 0))),
        }
        if self.event_sink is not None:
            self.event_sink.write(
                make_tool_event(
                    "fetch",
                    "started",
                    event_name="source_fetch_started",
                    payload={**task_context, "requested_url": result.get("url", "")},
                )
            )
        link = current_span_context()
        try:
            if self.trace_recorder is not None:
                with self.trace_recorder.span("fetch", kind="tool"):
                    # fetch 事件全部归属 fetch span；span 内取一次身份，
                    # 失败路径同样带着它（span 已结束仍要能关联）。
                    link = current_span_context()
                    document = await self.fetcher.afetch(
                        result.get("url", ""),
                        fetch_timeout=self.fetch_timeout,
                        parse_timeout=self.parse_timeout,
                    )
            else:
                document = await self.fetcher.afetch(
                    result.get("url", ""),
                    fetch_timeout=self.fetch_timeout,
                    parse_timeout=self.parse_timeout,
                )
            if document.get("error"):
                raise ToolParseError(str(document.get("error", "来源解析失败")))
            published_at = str(result.get("published_at", "")).strip()
            if published_at:
                # 搜索引擎给出的时效信息：附加于文档之上供抽取提示参考，
                # 不来自页面本身，因此绝不能写进 L2 抓取缓存的正文。
                document = cast(SourceDocument, {**document, "published_at": published_at})
            text = document.get("text", "")
            if not text:
                raise SourceUnavailableError("empty_content", "来源没有可读取正文。")
            source_url = document.get("final_url", result.get("url", ""))
            self.logger.info(
                "fetch_completed task=%s url=%s final_url=%s status=%s text_chars=%d",
                task["id"],
                result.get("url", ""),
                source_url,
                document.get("status_code"),
                len(text),
            )
            if self.event_sink is not None:
                self.event_sink.write(
                    make_tool_event(
                        "fetch",
                        "completed",
                        event_name="source_fetch_completed",
                        link=link,
                        duration_ms=(time.perf_counter() - started) * 1000,
                        payload={
                            **task_context,
                            "requested_url": result.get("url", ""),
                            "final_url": source_url,
                            "status_code": document.get("status_code"),
                            "content_type": document.get("content_type", ""),
                            "text_chars": len(text),
                            "raw_bytes": document.get("raw_bytes", 0),
                            "fetch_duration_ms": document.get("fetch_duration_ms", 0),
                            "parse_duration_ms": document.get("parse_duration_ms", 0),
                            "cache_hit": bool(document.get("cache_hit", False)),
                        },
                    )
                )
            if self.event_sink is not None:
                self.event_sink.write(
                    make_tool_event(
                        "evidence_extract",
                        "started",
                        event_name="evidence_extraction_started",
                        link=link,
                        payload={
                            "task_id": task["id"],
                            "source_url": source_url,
                            "text_chars": len(text),
                        },
                    )
                )
            try:
                extraction = await asyncio.wait_for(
                    self.extractor.aextract_result(task, document, result),
                    timeout=self.evidence_extract_timeout,
                )
            except Exception as exc:
                if isinstance(exc, asyncio.TimeoutError):
                    exc = TimeoutError(
                        f"evidence_extract_timeout（超过 {self.evidence_extract_timeout:.1f}s）"
                    )
                self.logger.warning(
                    "evidence_extraction_failed task=%s url=%s error=%s",
                    task["id"],
                    source_url,
                    exc,
                )
                if self.event_sink is not None:
                    self.event_sink.write(
                        make_tool_event(
                            "evidence_extract",
                            "failed",
                            event_name="evidence_extraction_failed",
                            link=link,
                            duration_ms=(time.perf_counter() - started) * 1000,
                            error=str(exc),
                            payload={"task_id": task["id"], "source_url": source_url},
                        )
                    )
                return failed_read(task, exc)
            if extraction.failed_chunk_count == extraction.chunk_count and extraction.chunk_count:
                reason = f"Evidence 抽取失败：{extraction.failed_chunk_count} 个 chunk 全部失败。"
                self.logger.warning(
                    "evidence_extraction_failed task=%s url=%s chunks=%d failed_chunks=%d",
                    task["id"],
                    source_url,
                    extraction.chunk_count,
                    extraction.failed_chunk_count,
                )
                return failed_read(task, RuntimeError(reason))
            evidences = extraction.evidences
            if not evidences:
                reason = "正文存在，但不足以支持当前子问题的可验证 Evidence。"
                self.logger.info(
                    "source_skipped task=%s url=%s reason_code=evidence_empty",
                    task["id"],
                    result.get("url", ""),
                )
                if self.event_sink is not None:
                    self.event_sink.write(
                        make_tool_event(
                            "evidence_extract",
                            "skipped",
                            event_name="evidence_validation_skipped",
                            link=link,
                            duration_ms=(time.perf_counter() - started) * 1000,
                            error=reason,
                            payload={
                                "task_id": task["id"],
                                "requested_url": result.get("url", ""),
                                "final_url": source_url,
                                "status_code": document.get("status_code"),
                                "text_chars": len(text),
                                "extraction_strategy": extraction.strategy,
                                "chunk_count": extraction.chunk_count,
                                "candidate_chars": extraction.candidate_chars,
                                "failed_chunk_count": extraction.failed_chunk_count,
                                "validation_rejected_count": extraction.validation_rejected_count,
                                "cache_hit": extraction.cache_hit,
                                "reason_code": "evidence_empty",
                            },
                        )
                    )
                return skipped_read(
                    task,
                    source_url=source_url,
                    reason_code="evidence_empty",
                    reason=reason,
                )
            self.logger.info(
                "evidence_extracted task=%s url=%s final_url=%s evidence_count=%d text_chars=%d strategy=%s chunks=%d",
                task["id"],
                result.get("url", ""),
                source_url,
                len(evidences),
                len(text),
                extraction.strategy,
                extraction.chunk_count,
            )
            if self.event_sink is not None:
                self.event_sink.write(
                    make_tool_event(
                        "evidence_extract",
                        "completed",
                        event_name="source_reader_completed",
                        link=link,
                        duration_ms=(time.perf_counter() - started) * 1000,
                        payload={
                            "task_id": task["id"],
                            "requested_url": result.get("url", ""),
                            "final_url": source_url,
                            "status_code": document.get("status_code"),
                            "extraction_strategy": extraction.strategy,
                            "chunk_count": extraction.chunk_count,
                            "candidate_chars": extraction.candidate_chars,
                            "raw_bytes": document.get("raw_bytes", 0),
                            "evidence_count": len(evidences),
                            "failed_chunk_count": extraction.failed_chunk_count,
                            "validation_rejected_count": extraction.validation_rejected_count,
                            "cache_hit": extraction.cache_hit,
                            # 事件会进入持久化记录并可能被 SSE 消费；
                            # audit_chunk 只留在 checkpoint/评测产物中。
                            "evidences": [item.agent_payload() for item in evidences],
                        },
                    )
                )
            return SourceReaderToolResult(
                task_id=task["id"],
                status="completed",
                source_url=source_url,
                # 再由任务配额统一裁剪，不能在单来源工具层静默丢弃。
                evidences=evidences,
            )
        except TimeoutError as exc:
            stage = str(exc) or "source_timeout"
            self.logger.warning(
                "source_stage_timeout task=%s url=%s stage=%s",
                task["id"],
                result.get("url", ""),
                stage,
            )
            if self.event_sink is not None:
                self.event_sink.write(
                    make_tool_event(
                        "source_reader",
                        "failed",
                        event_name="source_timeout",
                        link=link,
                        error=stage,
                        duration_ms=(time.perf_counter() - started) * 1000,
                        payload={
                            "task_id": task["id"],
                            "source_url": result.get("url", ""),
                            "timeout_stage": stage,
                        },
                    )
                )
            return failed_read(task, exc)
        except SourceUnavailableError as exc:
            fallback = await self._extract_search_content_fallback(
                task,
                result,
                fetch_error=str(exc),
                started=started,
                link=link,
            )
            if fallback is not None:
                return fallback
            self.logger.info(
                "source_skipped task=%s url=%s reason_code=%s",
                task["id"],
                result.get("url", ""),
                exc.reason_code,
            )
            if self.event_sink is not None:
                self.event_sink.write(
                    make_tool_event(
                        "fetch",
                        "skipped",
                        link=link,
                        duration_ms=(time.perf_counter() - started) * 1000,
                        error=str(exc),
                        payload={
                            "task_id": task["id"],
                            "requested_url": result.get("url", ""),
                            "reason_code": exc.reason_code,
                        },
                    )
                )
            return skipped_read(
                task,
                source_url=result.get("url", ""),
                reason_code=exc.reason_code,
                reason=str(exc),
            )
        except Exception as exc:
            fallback = await self._extract_search_content_fallback(
                task,
                result,
                fetch_error=str(exc),
                started=started,
                link=link,
            )
            if fallback is not None:
                return fallback
            self.logger.warning(
                "fetch_failed task=%s url=%s error=%s", task["id"], result.get("url", ""), exc
            )
            if self.event_sink is not None:
                self.event_sink.write(
                    make_tool_event(
                        "fetch",
                        "failed",
                        link=link,
                        error=str(exc),
                        duration_ms=(time.perf_counter() - started) * 1000,
                        payload={
                            "task_id": task["id"],
                            "requested_url": result.get("url", ""),
                        },
                    )
                )
            return failed_read(task, exc)

    async def _extract_search_content_fallback(
        self,
        task: SubTask,
        result: SearchResult,
        *,
        fetch_error: str,
        started: float,
        link: SpanContext,
    ) -> SourceReaderToolResult | None:
        """网页读取失败时，谨慎使用搜索提供商返回的内容。

        `raw_content` 是提供商提取的来源正文，保留为 direct，但明确记录获取方式；
        普通搜索摘要只能产生 partial Evidence，默认不会进入 Writer。
        """
        raw_content = str(result.get("raw_content", "")).strip()
        snippet = str(result.get("snippet", "")).strip()
        if raw_content:
            text, retrieval_method, support_ceiling = raw_content, "tavily_raw_content", "direct"
        elif snippet:
            text, retrieval_method, support_ceiling = snippet, "search_summary", "partial"
        else:
            return None

        source_url = str(result.get("url", ""))
        provider = str(result.get("content_provider", "search"))
        document: SourceDocument = {
            "title": str(result.get("title", "")),
            "final_url": source_url,
            "text": text,
            "blocks": [
                {
                    "block_id": f"{retrieval_method}-0001",
                    "block_type": "paragraph",
                    "text": text,
                    "heading_path": [],
                    "order": 0,
                }
            ],
            "retrieval_method": retrieval_method,
            "support_ceiling": support_ceiling,
            "published_at": str(result.get("published_at", "")),
        }
        self.logger.info(
            "source_content_fallback task=%s url=%s provider=%s method=%s chars=%d fetch_error=%s",
            task["id"],
            source_url,
            provider,
            retrieval_method,
            len(text),
            fetch_error,
        )
        if self.event_sink is not None:
            self.event_sink.write(
                make_tool_event(
                    "source_content_fallback",
                    "started",
                    link=link,
                    payload={
                        "task_id": task["id"],
                        "source_url": source_url,
                        "provider": provider,
                        "retrieval_method": retrieval_method,
                        "support_ceiling": support_ceiling,
                        "text_chars": len(text),
                        "fetch_error": fetch_error[:300],
                    },
                )
            )
        try:
            extraction = await asyncio.wait_for(
                self.extractor.aextract_result(task, document, result),
                timeout=self.evidence_extract_timeout,
            )
        except Exception as exc:
            if isinstance(exc, asyncio.TimeoutError):
                exc = TimeoutError(
                    f"evidence_extract_timeout（超过 {self.evidence_extract_timeout:.1f}s）"
                )
            self.logger.warning(
                "fallback_evidence_extraction_failed task=%s url=%s method=%s error=%s",
                task["id"],
                source_url,
                retrieval_method,
                exc,
            )
            if self.event_sink is not None:
                self.event_sink.write(
                    make_tool_event(
                        "source_content_fallback",
                        "failed",
                        link=link,
                        duration_ms=(time.perf_counter() - started) * 1000,
                        error=str(exc),
                        payload={
                            "task_id": task["id"],
                            "source_url": source_url,
                            "retrieval_method": retrieval_method,
                        },
                    )
                )
                return failed_read(task, exc)

        if extraction.failed_chunk_count == extraction.chunk_count and extraction.chunk_count:
            reason = f"Evidence 抽取失败：{extraction.failed_chunk_count} 个 chunk 全部失败。"
            self.logger.warning(
                "fallback_evidence_extraction_failed task=%s url=%s chunks=%d failed_chunks=%d",
                task["id"],
                source_url,
                extraction.chunk_count,
                extraction.failed_chunk_count,
            )
            return failed_read(task, RuntimeError(reason))

        evidences = extraction.evidences
        if not evidences:
            reason_code = f"{retrieval_method}_evidence_empty"
            if self.event_sink is not None:
                self.event_sink.write(
                    make_tool_event(
                        "source_content_fallback",
                        "skipped",
                        link=link,
                        duration_ms=(time.perf_counter() - started) * 1000,
                        payload={
                            "task_id": task["id"],
                            "source_url": source_url,
                            "retrieval_method": retrieval_method,
                            "reason_code": reason_code,
                        },
                    )
                )
            return skipped_read(
                task,
                source_url=source_url,
                reason_code=reason_code,
                reason="搜索提供商返回的内容不足以支持当前子问题的 Evidence。",
            )

        if self.event_sink is not None:
            self.event_sink.write(
                make_tool_event(
                    "source_content_fallback",
                    "completed",
                    link=link,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    payload={
                        "task_id": task["id"],
                        "source_url": source_url,
                        "retrieval_method": retrieval_method,
                        "support_ceiling": support_ceiling,
                        "evidence_count": len(evidences),
                        "cache_hit": extraction.cache_hit,
                    },
                )
            )
        return SourceReaderToolResult(
            task_id=task["id"],
            status="completed",
            source_url=source_url,
            evidences=evidences,
        )
