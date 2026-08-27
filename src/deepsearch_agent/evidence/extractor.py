"""从完整来源或结构化分块中抽取可验证 Evidence。"""

import asyncio
import hashlib
import math
from dataclasses import dataclass
from typing import Literal, cast

from langchain_core.messages import HumanMessage, SystemMessage

from deepsearch_agent.evidence.models import Evidence, EvidenceExtraction
from deepsearch_agent.evidence.tokens import TokenEstimator, get_token_estimator
from deepsearch_agent.evidence.validator import validate_evidence
from deepsearch_agent.llm import LLMConfigurationError, LLMInvoker, ainvoke_structured
from deepsearch_agent.observability.events import JsonlSink, make_audit_event, make_tool_event
from deepsearch_agent.observability.logger import get_logger
from deepsearch_agent.observability.tracing.context import current_context
from deepsearch_agent.parsers.models import DocumentBlock, ParsedDocument
from deepsearch_agent.state import SubTask
from deepsearch_agent.tools.search import SearchResult


class EvidenceExtractor:
    def __init__(
        self,
        llm: LLMInvoker,
        *,
        input_budget_tokens: int = 24_000,
        context_window_tokens: int = 32_768,
        output_budget_tokens: int = 4_000,
        safety_margin_tokens: int = 2_000,
        chunk_concurrency: int = 2,
        max_evidences: int | None = None,
        estimator: TokenEstimator | None = None,
        event_sink: JsonlSink | None = None,
    ):
        if llm is None:
            raise LLMConfigurationError("EvidenceExtractor 需要已装配的 LLMInvoker。")
        self.llm = llm
        self.input_budget_tokens = min(
            input_budget_tokens,
            context_window_tokens - output_budget_tokens - safety_margin_tokens,
        )
        if self.input_budget_tokens <= 0:
            raise ValueError("EvidenceExtractor 的 token 输入预算必须小于上下文窗口预留空间。")
        self.estimator = estimator or get_token_estimator()
        self.chunk_concurrency = chunk_concurrency
        self.max_evidences = max_evidences
        self.event_sink = event_sink
        self.logger = get_logger("deepsearch_agent.evidence.extractor")

    async def aextract(
        self, task: SubTask, document: ParsedDocument, result: SearchResult
    ) -> list[Evidence]:
        return (await self.aextract_result(task, document, result)).evidences

    async def aextract_result(
        self, task: SubTask, document: ParsedDocument, result: SearchResult
    ) -> "ExtractionResult":
        blocks = cast(list[DocumentBlock], document.get("blocks", []))
        if not blocks and document.get("text"):
            blocks = cast(
                list[DocumentBlock],
                [
                    {
                        "block_id": "b-0000",
                        "block_type": "paragraph",
                        "text": document["text"],
                        "heading_path": [],
                        "order": 0,
                    }
                ],
            )
        if not blocks:
            return ExtractionResult([], "empty_document", 0, 0)
        chunks, strategy = self._build_chunks(blocks)
        if not chunks:
            return ExtractionResult([], strategy, 0, 0)
        extracted_by_chunk, failed_chunk_count = await self._extract_chunks(
            task, document, result, chunks
        )
        evidences: list[Evidence] = []
        seen_quotes: set[str] = set()
        validation_rejected_count = 0
        source_url = document.get("final_url", result.get("url", ""))
        source_fingerprint = hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:10]
        for chunk, extracted in zip(chunks, extracted_by_chunk, strict=True):
            for item in extracted.evidences:
                if self.max_evidences is not None and len(evidences) >= self.max_evidences:
                    break
                quote_key = " ".join(item.quote.lower().split())
                if not quote_key or quote_key in seen_quotes:
                    continue
                evidence = Evidence(
                    # 同一 task 会读取多个来源；来源指纹避免每个来源都从 ev-1
                    # 开始而在 State reducer 中发生 ID 碰撞。
                    evidence_id=f"{task['id']}-src-{source_fingerprint}-ev-{len(evidences) + 1}",
                    subtask_id=task["id"],
                    research_direction=task["question"],
                    claim=item.claim,
                    quote=item.quote,
                    source_url=source_url,
                    source_title=document.get("title", result.get("title", "")),
                    retrieval_method=cast(
                        Literal["origin_fetch", "tavily_raw_content", "search_summary"],
                        document.get("retrieval_method", "origin_fetch"),
                    ),
                    support=self._cap_support(
                        cast(Literal["direct", "partial", "insufficient"], item.support),
                        cast(
                            Literal["direct", "partial", "insufficient"],
                            document.get("support_ceiling", "direct"),
                        ),
                    ),
                    confidence=item.confidence,
                )
                try:
                    evidences.append(validate_evidence(evidence, chunk))
                    seen_quotes.add(quote_key)
                except ValueError as exc:
                    validation_rejected_count += 1
                    self.logger.warning(
                        "evidence_candidate_rejected task=%s chunk_blocks=%s reason=%s quote_chars=%d",
                        task["id"],
                        [block.get("block_id", "") for block in chunk],
                        exc,
                        len(item.quote),
                    )
                    continue
            if self.max_evidences is not None and len(evidences) >= self.max_evidences:
                break
        return ExtractionResult(
            evidences=evidences,
            strategy=strategy,
            chunk_count=len(chunks),
            candidate_chars=sum(len(block.get("text", "")) for chunk in chunks for block in chunk),
            failed_chunk_count=failed_chunk_count,
            validation_rejected_count=validation_rejected_count,
        )

    @staticmethod
    def _cap_support(
        extracted_support: Literal["direct", "partial", "insufficient"],
        source_support_ceiling: Literal["direct", "partial", "insufficient"],
    ) -> Literal["direct", "partial", "insufficient"]:
        """搜索摘要不能被抽取器升级为比来源材料更强的 Evidence。"""
        rank = {"insufficient": 0, "partial": 1, "direct": 2}
        return cast(
            Literal["direct", "partial", "insufficient"],
            min(
                (extracted_support, source_support_ceiling),
                key=lambda level: rank.get(level, 0),
            ),
        )

    async def _extract_chunks(
        self, task, document, result, chunks: list[list[DocumentBlock]]
    ) -> tuple[list[EvidenceExtraction], int]:
        semaphore = asyncio.Semaphore(self.chunk_concurrency)

        # 全文过长时每个 chunk 都要一次模型调用。把总配额均分为每块的
        # 输出上限，避免模型产生大量最终不会进入 State 的候选 Evidence。
        per_chunk_limit = None
        if self.max_evidences is not None:
            per_chunk_limit = max(1, math.ceil(self.max_evidences / len(chunks)))

        async def extract_one(index: int, chunk: list[DocumentBlock]):
            async with semaphore:
                chunk_ids = [block.get("block_id", "") for block in chunk]
                started = asyncio.get_running_loop().time()
                if self.event_sink is not None:
                    context = current_context()
                    self.event_sink.write(
                        make_tool_event(
                            "evidence_extract",
                            "started",
                            event_name="evidence_chunk_started",
                            trace_id=context.trace_id if context else None,
                            span_id=context.span_id if context else None,
                            payload={"task_id": task["id"], "chunk_index": index, "chunk_count": len(chunks), "block_ids": chunk_ids},
                        )
                    )
                try:
                    llm_started = asyncio.get_running_loop().time()
                    extracted = await self._aextract_with_llm(
                        task, document, result, chunk, max_evidences=per_chunk_limit
                    )
                except asyncio.CancelledError:
                    self.logger.warning("evidence_chunk_cancelled task=%s chunk=%d/%d", task["id"], index, len(chunks))
                    raise
                except Exception as exc:
                    elapsed = asyncio.get_running_loop().time() - started
                    self.logger.warning(
                        "evidence_chunk_failed task=%s chunk=%d/%d duration_ms=%.2f error_type=%s error=%s",
                        task["id"], index, len(chunks), elapsed * 1000, type(exc).__name__, exc,
                    )
                    if self.event_sink is not None:
                        context = current_context()
                        self.event_sink.write(
                            make_tool_event(
                                "evidence_extract", "failed",
                                event_name="evidence_chunk_failed",
                                trace_id=context.trace_id if context else None,
                                span_id=context.span_id if context else None,
                                duration_ms=elapsed * 1000, error=str(exc),
                                payload={"task_id": task["id"], "chunk_index": index, "chunk_count": len(chunks), "block_ids": chunk_ids, "error_type": type(exc).__name__},
                            )
                        )
                    return EvidenceExtraction(), True
                response_json = extracted.model_dump_json(ensure_ascii=False)
                llm_duration_ms = (
                    asyncio.get_running_loop().time() - llm_started
                ) * 1000
                self.logger.info(
                    "evidence_llm_completed task=%s source_url=%s chunk=%d/%d "
                    "llm_duration_ms=%.2f response_type=%s response_chars=%d candidate_count=%d",
                    task["id"],
                    document.get("final_url", result.get("url", "")),
                    index,
                    len(chunks),
                    llm_duration_ms,
                    type(extracted).__name__,
                    len(response_json),
                    len(extracted.evidences),
                )
                if self.event_sink is not None:
                    context = current_context()
                    self.event_sink.write(
                        make_audit_event(
                            "evidence_llm_response",
                            trace_id=context.trace_id if context else None,
                            span_id=context.span_id if context else None,
                            run_id=context.run_id if context else None,
                            session_id=context.session_id if context else None,
                            node_id=context.node_id if context else "evidence_extract",
                            component="evidence_extractor",
                            payload={
                                "task_id": task["id"],
                                "source_url": document.get("final_url", result.get("url", "")),
                                "chunk_index": index,
                                "chunk_count": len(chunks),
                                "llm_duration_ms": round(llm_duration_ms, 2),
                                "response_type": type(extracted).__name__,
                                "response_chars": len(response_json),
                                "candidate_count": len(extracted.evidences),
                                "response_preview": response_json[:1_000],
                            },
                        )
                    )
                elapsed = asyncio.get_running_loop().time() - started
                if self.event_sink is not None:
                    context = current_context()
                    self.event_sink.write(
                        make_tool_event(
                            "evidence_extract", "completed",
                            event_name="evidence_chunk_completed",
                            trace_id=context.trace_id if context else None,
                            span_id=context.span_id if context else None,
                            duration_ms=elapsed * 1000,
                            payload={"task_id": task["id"], "chunk_index": index, "chunk_count": len(chunks), "block_ids": chunk_ids, "candidate_count": len(extracted.evidences)},
                        )
                    )
                return extracted, False

        outcomes = await asyncio.gather(
            *(extract_one(index, chunk) for index, chunk in enumerate(chunks, 1)),
            return_exceptions=True,
        )
        extracted: list[EvidenceExtraction] = []
        failed_count = 0
        for outcome in outcomes:
            if isinstance(outcome, asyncio.CancelledError):
                raise outcome
            if isinstance(outcome, BaseException):
                # 防御性处理：extract_one 已将普通异常转成结果；若未来出现未处理
                # 的异常，仍不能让一个 chunk 破坏其他已完成 chunk。
                failed_count += 1
                extracted.append(EvidenceExtraction())
                continue
            chunk_result, failed = outcome
            extracted.append(chunk_result)
            failed_count += int(failed)
        self.logger.info(
            "evidence_chunks_completed task=%s chunks=%d failed_chunks=%d candidate_count=%d",
            task["id"], len(chunks), failed_count,
            sum(len(item.evidences) for item in extracted),
        )
        return extracted, failed_count

    def _build_chunks(self, blocks: list[DocumentBlock]) -> tuple[list[list[DocumentBlock]], str]:
        expanded = [part for block in blocks for part in self._split_large_block(block)]
        total_tokens = self._blocks_tokens(expanded)
        if total_tokens <= self.input_budget_tokens:
            return [expanded], "full_document"
        chunks: list[list[DocumentBlock]] = []
        current: list[DocumentBlock] = []
        current_tokens = 0
        for block in expanded:
            block_tokens = self._block_tokens(block)
            if current and current_tokens + block_tokens > self.input_budget_tokens:
                chunks.append(current)
                current, current_tokens = [], 0
            current.append(block)
            current_tokens += block_tokens
        if current:
            chunks.append(current)
        return chunks, "structured_chunks"

    def _split_large_block(self, block: DocumentBlock) -> list[DocumentBlock]:
        text = block.get("text", "")
        if self.estimator.count(text) <= self.input_budget_tokens:
            return [block]
        parts = []
        start = 0
        index = 0
        while start < len(text):
            end = self._fit_text_end(text[start:])
            if end <= 0:
                end = len(text[start:])
            part = dict(block)
            part["block_id"] = f"{block['block_id']}-part{index + 1}"
            part["text"] = text[start : start + end]
            parts.append(part)
            start += end
            index += 1
        return parts

    def _fit_text_end(self, text: str) -> int:
        """寻找不超过输入预算的最长原文前缀，不重新编码正文。"""
        low, high = 1, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if self.estimator.count(text[:middle]) <= self.input_budget_tokens:
                low = middle
            else:
                high = middle - 1
        return low

    def _block_tokens(self, block: DocumentBlock) -> int:
        heading = " > ".join(block.get("heading_path", []))
        return self.estimator.count(f"[{block['block_id']}] {heading}\n{block['text']}")

    def _blocks_tokens(self, blocks: list[DocumentBlock]) -> int:
        return sum(self._block_tokens(block) for block in blocks)

    async def _aextract_with_llm(
        self, task, document, result, selected, *, max_evidences: int | None = None
    ):
        context = "\n\n".join(
            f"[{block['block_id']}] {' > '.join(block.get('heading_path', []))}\n{block['text']}"
            for block in selected
        )
        system_prompt = (
            "【运行背景】你是深度研究流水线的 Evidence 抽取器。输入是某个来源的局部原文与一个研究子问题；"
            "你的输出会被 Writer 和审阅器当作唯一允许引用的事实基础。"
            "【任务目标】挑选能直接回答该子问题的少量原文证据，并把每条证据压缩为可核验的 claim 与逐字 quote。"
            "【判定标准】quote 必须逐字来自给定原文，且应包含使 claim 成立的主体、条件、数字、时间、范围与限制。"
            "claim 只能忠实改写 quote，不得补充背景知识、评价、因果、常识推断或标题中未被正文支持的信息。"
            "若原文只有标题、导航、验证码、无关内容，或无法直接支撑子问题，返回空 evidences；空结果优于猜测。"
            "【输出约束】只返回 JSON object，不要输出 Markdown、代码围栏或解释。JSON 示例："
            '{"evidences":[{"claim":"原文支持的结论","quote":"原文逐字引文",'
            '"support":"direct","confidence":0.8}]}'
            '；没有足够证据时返回：{"evidences":[]}。'
        )
        if max_evidences is not None:
            system_prompt += (
                f"最多返回 {max_evidences} 条 Evidence；优先选择最直接回答当前子问题、"
                "信息密度最高且包含必要限定条件的证据。"
            )
        user_prompt = (
            f"研究子问题：{task['question']}\n"
            f"来源标题：{document.get('title', result.get('title', ''))}\n"
            f"候选原文：\n{context}"
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        return await ainvoke_structured(
            self.llm,
            EvidenceExtraction,
            messages,
            request_kwargs={"extra_body": {"thinking": {"type": "disabled"}}},
        )


@dataclass(frozen=True)
class ExtractionResult:
    evidences: list[Evidence]
    strategy: str
    chunk_count: int
    candidate_chars: int
    failed_chunk_count: int = 0
    validation_rejected_count: int = 0
