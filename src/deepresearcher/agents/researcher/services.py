"""Researcher 任务级工具的业务实现：不感知 langgraph 的纯 async 函数。"""

import asyncio
import hashlib
from typing import Any, cast

from deepresearcher.agents.researcher.state import (
    ResearcherDeps,
    ResearcherTaskState,
    evidence_observation_card,
)
from deepresearcher.errors import clip_text
from deepresearcher.evidence.models import Evidence, normalize_title_whitespace
from deepresearcher.evidence.validator import SourceIndex, collapse_whitespace
from deepresearcher.observability.events.names import EventName
from deepresearcher.schemas import Task, TaskStopReason
from deepresearcher.schemas.limits import (
    SEARCH_RESULT_SNIPPET_PREVIEW_CHARS,
    SEARCH_RESULTS_PREVIEW_COUNT,
)
from deepresearcher.tools.web.documents import DocumentView
from deepresearcher.tools.web.fetch.models import SourceReaderToolResult
from deepresearcher.tools.web.materials import SearchResultSet
from deepresearcher.tools.web.search.models import (
    SearchCandidate,
    SearchResult,
    SearchToolResult,
    classify_source,
    describe_source,
)
from deepresearcher.vocab import SUPPORT_ORDER


async def search_sources(
    deps: ResearcherDeps,
    task: Task,
    task_state: ResearcherTaskState,
    event_context: dict[str, object],
    proposed_queries: list[str],
    reason: str,
) -> dict[str, object]:
    """搜索并返回候选目录；此函数不读取任何来源。"""
    del reason
    remaining = deps.config.research_agent_max_queries - len(task_state.queries)
    new_queries = _new_queries(deps, proposed_queries, task_state.queries)[: max(0, remaining)]
    if not new_queries:
        error = "没有新的可执行检索式；请基于已有候选读取来源或调用 Complete。"
        task_state.failures.append(f"no_novel_queries: {error}")
        return {
            "status": "skipped",
            "reason": "no_novel_queries",
            "proposed_queries": proposed_queries,
        }
    task_state.queries.extend(new_queries)
    result = SearchToolResult.model_validate(
        await deps.search_tool.arun_queries(task, queries=new_queries)
    )
    task_state.failures.extend(
        f"search query={item.query}: {item.error}" for item in result.failures
    )
    deps.emit(
        EventName.TASK_SEARCH_COMPLETED,
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
        task_state.failures.append(f"search: {error}")
        if getattr(result, "provider_exhausted", False):
            # 把"原因 + 下一步"写清楚交给模型自然收尾，不加控制流分支：
            # 熔断后系统级不可用，重试/换向无意义。
            task_state.provider_exhausted = True
            return {
                "status": "failed",
                "provider_exhausted": True,
                "user_code": result.provider_user_code,
                "queries": new_queries,
                "error": error,
                "instruction": (
                    "搜索服务账户级不可用（额度耗尽或密钥无效），系统性、非本任务偶发。"
                    "不要再重试、换词或扩大范围。若已读到可用原文，立即调用 "
                    "TaskComplete 提交已覆盖内容，并在 conclusion 里用自然语言写清"
                    "已查清什么、还差什么；若尚无原文，也直接 Complete 并标注证据不足。"
                ),
            }
        return {"status": "failed", "queries": new_queries, "error": error}

    batch_key = "\0".join([task["id"], *result.queries])
    search_id = "search-" + hashlib.sha256(batch_key.encode("utf-8")).hexdigest()[:16]
    task_state.search_batches[search_id] = list(result.queries)
    if deps.material_store is not None:
        await deps.material_store.put_search_results(
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
        task_state.candidates[candidate_id] = SearchCandidate(
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
    # 预览与 ListSearchResults 共用同一卡片结构 + 同一 raw_results 下标偏移基准。
    preview, next_offset = _candidate_cards(
        task_state, raw_results, start=0, count=SEARCH_RESULTS_PREVIEW_COUNT
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


async def list_search_results(
    deps: ResearcherDeps,
    task: Task,
    task_state: ResearcherTaskState,
    search_id: str,
    offset: int,
    limit: int,
    reason: str,
) -> dict[str, object]:
    """只允许分页读取当前任务亲自产生的搜索结果。"""
    del reason
    queries = task_state.search_batches.get(search_id)
    if queries is None:
        return {"status": "rejected", "reason": "unknown_search_id"}
    if deps.material_store is not None:
        try:
            stored = await deps.material_store.get_search_results(
                str(task.get("run_id") or task["id"]), search_id
            )
        except FileNotFoundError:
            return {"status": "failed", "reason": "search_results_expired"}
        raw_results = stored.results
    else:
        result = SearchToolResult.model_validate(
            await deps.search_tool.arun_queries(task, queries=queries)
        )
        if result.status != "completed":
            return {"status": "failed", "reason": result.error or "search_cache_read_failed"}
        raw_results = [dict(item) for item in result.results]
    cards, next_offset = _candidate_cards(task_state, raw_results, start=offset, count=limit)
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


async def read_sources(
    deps: ResearcherDeps,
    task: Task,
    task_state: ResearcherTaskState,
    event_context: dict[str, object],
    candidate_ids: list[str],
    reason: str,
) -> dict[str, object]:
    """读取模型选中的候选来源，并返回紧凑的工具结果。"""
    del reason
    selected_ids = list(dict.fromkeys(candidate_ids))
    selected = [
        task_state.candidates[item] for item in selected_ids if item in task_state.candidates
    ]
    unknown_ids = [item for item in selected_ids if item not in task_state.candidates]
    if unknown_ids:
        task_state.failures.append(f"unknown_candidate_ids: {', '.join(unknown_ids)}")
    candidates: list[SearchCandidate] = []
    for candidate in selected:
        if candidate.candidate_id in task_state.selected_candidate_ids:
            continue
        task_state.selected_candidate_ids.add(candidate.candidate_id)
        task_state.read_urls.append(candidate.url)
        candidates.append(candidate)

    read_results = await _read_candidates(deps, task, candidates)
    documents: list[DocumentView] = []
    inline_tokens = 0
    for candidate, read_result in zip(candidates, read_results, strict=True):
        url = candidate.url
        if isinstance(read_result, asyncio.CancelledError):
            raise read_result
        if isinstance(read_result, Exception):
            task_state.failures.append(f"{url}: {read_result}")
            deps.emit(
                EventName.SOURCE_READ_FAILED,
                {
                    **event_context,
                    "research_direction": task["question"],
                    "url": url,
                    "error": clip_text(str(read_result)),
                },
            )
            continue
        result = SourceReaderToolResult.model_validate(read_result)
        if result.status == "completed":
            for document in result.documents:
                view = document
                if view.inline:
                    if (
                        view.token_count + inline_tokens
                        > deps.config.document_inline_total_max_tokens
                    ):
                        view = view.model_copy(update={"inline": False, "content": ""})
                    else:
                        inline_tokens += view.token_count
                task_state.documents[view.document_id] = view
                documents.append(view)
                if view.source_url:
                    task_state.source_refs.append(view.source_url)
        elif result.status == "skipped":
            reason_code = result.reason_code or "unknown"
            task_state.skipped.append(reason_code)
            deps.emit(
                EventName.SOURCE_READ_SKIPPED,
                {
                    **event_context,
                    "research_direction": task["question"],
                    "url": url,
                    "reason_code": reason_code,
                },
            )
        else:
            error = result.error or "read_failed"
            task_state.failures.append(f"{url}: {error}")
            deps.emit(
                EventName.SOURCE_READ_FAILED,
                {
                    **event_context,
                    "research_direction": task["question"],
                    "url": url,
                    "error": error,
                },
            )
    receipt: dict[str, object] = {
        "candidate_ids": selected_ids,
        "read_candidate_count": len(candidates),
        "unknown_candidate_ids": unknown_ids,
        "documents": [item.model_dump(mode="json") for item in documents],
        "archive_evidence_count": len(task_state.evidences),
        "active_evidence_count": len(task_state.active_evidence_ids),
        "active_evidence_limit": task_state.active_evidence_limit,
        "archive_evidence_limit": task_state.evidence_archive_limit,
        "skip_reasons": sorted(set(task_state.skipped)),
        "recent_failures": task_state.failures[-4:],
    }
    if unknown_ids:
        # 与 unknown 回执配对：给出可用的候选 id，模型无需回头重查搜索目录。
        receipt["available_candidate_ids"] = list(task_state.candidates)[-_AVAILABLE_IDS_WINDOW:]
    return receipt


# 有效 id 清单的条数上限：够模型改引用，又不至于在上下文里堆长。
_AVAILABLE_IDS_WINDOW = 12


def _unknown_document_receipt(task_state: "ResearcherTaskState") -> dict[str, object]:
    """未知 document_id 的驳回回执：附当前有效清单，让模型改引用而不是猜。"""
    return {
        "status": "rejected",
        "reason": "unknown_document_id",
        "available_document_ids": list(task_state.documents)[-_AVAILABLE_IDS_WINDOW:],
    }


async def grep_document(
    deps: ResearcherDeps,
    task_state: ResearcherTaskState,
    document_id: str,
    patterns: list[str],
    segments: list[tuple[int, int]],
    context_lines: int,
    reason: str,
) -> dict[str, object]:
    """多关键词一次归并检索：命中按文档行号成流，segments 从流上切多段。

    参数 segments 为 (offset, count) 列表，offset 指归并流位置；回执带每词命中
    窗口总数（pattern_totals），供模型判断哪个词值得追加段落深挖。
    """
    del reason
    if document_id not in task_state.documents:
        return _unknown_document_receipt(task_state)
    if deps.material_store is None:
        return {"status": "failed", "reason": "material_store_unavailable"}
    cleaned_patterns = [item.strip() for item in patterns if item.strip()]
    if not cleaned_patterns:
        return {"status": "rejected", "reason": "empty_patterns"}
    result = await deps.material_store.grep_patterns(
        document_id,
        cleaned_patterns,
        context_lines=min(context_lines, deps.config.document_grep_context_lines),
        segments=[(max(0, offset), count) for offset, count in segments],
        max_matches_per_segment=deps.config.document_grep_max_matches,
        max_chars=deps.config.document_grep_max_chars,
        max_line_chars=deps.config.document_grep_max_line_chars,
    )
    return {
        "status": "completed",
        "document_id": document_id,
        "pattern_totals": result.pattern_totals,
        "total_windows": result.total_windows,
        "segments": [item.model_dump(mode="json") for item in result.segments],
        "has_more": result.has_more,
    }


async def read_document(
    deps: ResearcherDeps,
    task_state: ResearcherTaskState,
    document_id: str,
    ranges: list[tuple[int, int]],
    reason: str,
) -> dict[str, object]:
    del reason
    if document_id not in task_state.documents:
        return _unknown_document_receipt(task_state)
    if deps.material_store is None:
        return {"status": "failed", "reason": "material_store_unavailable"}
    requested = ranges[: deps.config.document_read_max_ranges]
    try:
        windows = await deps.material_store.read(
            document_id,
            requested,
            max_lines=deps.config.document_read_max_lines,
            max_chars=deps.config.document_read_max_chars,
        )
    except ValueError as exc:
        return {"status": "rejected", "reason": str(exc)}
    return {
        "status": "completed",
        "document_id": document_id,
        "ranges": [item.model_dump(mode="json") for item in windows],
        "truncated_range_count": max(0, len(ranges) - len(requested)),
    }


async def read_documents(
    deps: ResearcherDeps,
    task_state: ResearcherTaskState,
    targets: list[tuple[str, list[tuple[int, int]]]],
    reason: str,
) -> dict[str, object]:
    """跨文档批量读取：任一文档号未登记即整批驳回并附已登记清单，全部在场才逐文档并发执行。

    参数 targets 为 (document_id, [(start_line, end_line), ...]) 列表。返回 documents 键
    逐文档回执，行段与行数预算对每份文档独立生效，批次文档数上限由 ReadDocument 入参约束。
    """
    if deps.material_store is None:
        return {"status": "failed", "reason": "material_store_unavailable"}
    if any(document_id not in task_state.documents for document_id, _ in targets):
        return _unknown_document_receipt(task_state)
    per_document = await asyncio.gather(
        *(
            read_document(deps, task_state, document_id, ranges, reason)
            for document_id, ranges in targets
        )
    )
    return {"status": "completed", "documents": list(per_document)}


async def _validate_submissions(
    deps: ResearcherDeps,
    task: Task,
    task_state: ResearcherTaskState,
    submissions: list[dict[str, object]],
) -> tuple[list[tuple[int, str, Evidence]], list[dict[str, object]], list[str]]:
    """锁外逐条校验提交：原文读取、锚定回填 quote、批内去重、置信度解析。

    返回 (candidates, rejected, duplicates)。产物只是候选 Evidence 与拒因清单，不触碰
    task_state——写入的并发/容量判断全部留给 add_evidence 的同步临界区。
    """
    rejected: list[dict[str, object]] = []
    duplicates: list[str] = []
    pending_ids: set[str] = set()
    candidates: list[tuple[int, str, Evidence]] = []
    # 批内缓存：同一文档的多条引用共用一次原文读取与一次全文归一。
    source_indexes: dict[str, SourceIndex] = {}
    assert deps.material_store is not None  # 调用方已守卫
    for index, raw in enumerate(submissions):
        document_id = str(raw.get("document_id", ""))
        document = task_state.documents.get(document_id)
        if document is None:
            rejected.append({"index": index, "reason": "unknown_document_id"})
            continue
        claim = str(raw.get("claim", "")).strip()
        quote_start = str(raw.get("quote_start", "")).strip()
        quote_end = str(raw.get("quote_end", "")).strip()
        source_index = source_indexes.get(document_id)
        if source_index is None:
            try:
                source_text = await deps.material_store.text(document_id)
            except (FileNotFoundError, KeyError) as exc:
                rejected.append({"index": index, "reason": str(exc)})
                continue
            # 全文归一逐字符跑 NFKC，是这条路径唯一的 CPU 热点：丢线程池，别让事件
            # 循环连同租约确认与 SSE 一起停住。
            source_index = await asyncio.to_thread(SourceIndex.scan, source_text)
            source_indexes[document_id] = source_index
        # quote 必须是原文严格子串——由锚定解析从原文逐字回填，模型不手抄整段。
        # 锚未命中、有歧义或跨度超限都是提交错误，回执理由码，anchor_not_found 附最接近的原文句。
        resolution = source_index.resolve(quote_start, quote_end)
        if resolution.reason:
            rejection: dict[str, object] = {"index": index, "reason": resolution.reason}
            if resolution.reason == "anchor_not_found":
                # 逐句比对相似度是这条路径上最重的一步，丢线程池，别卡住事件循环。
                hint = await asyncio.to_thread(source_index.nearest_passage, quote_start)
                if hint:
                    rejection["nearby_original_text"] = hint
            rejected.append(rejection)
            continue
        quote = resolution.quote
        digest = hashlib.sha1(
            f"{document_id}\0{collapse_whitespace(quote)}".encode("utf-8")
        ).hexdigest()[:16]
        evidence_id = f"{task['id']}-ev-{digest}"
        if evidence_id in pending_ids:
            duplicates.append(evidence_id)
            continue
        requested_support = str(raw.get("support", "direct"))
        support = _bounded_support(requested_support, document.support_ceiling)
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
            task_id=task["id"],
            research_direction=task["question"],
            claim=claim,
            quote=quote,
            source_url=document.source_url,
            source_title=normalize_title_whitespace(document.title),
            published_at=document.published_at,
            source_profile=describe_source(document.source_url),
            retrieval_method=document.retrieval_method,
            support=cast(Any, support),
            confidence=confidence,
        )
        candidates.append((index, document_id, evidence))
        pending_ids.add(evidence_id)
    return candidates, rejected, duplicates


async def add_evidence(
    deps: ResearcherDeps,
    task: Task,
    task_state: ResearcherTaskState,
    event_context: dict[str, object],
    submissions: list[dict[str, object]],
    reason: str,
) -> dict[str, object]:
    """向当前任务证据池中批量添加证据"""
    if deps.material_store is None:
        return {"status": "failed", "reason": "material_store_unavailable"}
    accepted: list[Evidence] = []
    candidates, rejected, duplicates = await _validate_submissions(
        deps, task, task_state, submissions
    )
    ranked = sorted(candidates, key=lambda item: item[2].confidence, reverse=True)
    selected = ranked[: deps.config.evidence_add_batch_size]
    existing_ids = {item.evidence_id for item in task_state.evidences}
    per_source = {
        source_url: sum(1 for item in task_state.evidences if item.source_url == source_url)
        for source_url in {document.source_url for document in task_state.documents.values()}
    }
    for index, document_id, evidence in selected:
        if evidence.evidence_id in existing_ids:
            duplicates.append(evidence.evidence_id)
            continue
        document = task_state.documents[document_id]
        source_count = per_source.get(document.source_url, 0)
        if source_count >= deps.config.evidence_max_per_source:
            rejected.append({"index": index, "reason": "source_evidence_limit_reached"})
            continue
        added = task_state.add_evidences([evidence])
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
    deps.emit(
        EventName.TASK_EVIDENCE_ADDED,
        {
            **event_context,
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "duplicate_count": len(duplicates),
            "rejected_reasons": rejected_reasons,
            "reason": reason.strip()[:400],
        },
    )
    receipt: dict[str, object] = {
        "status": "completed" if accepted else "rejected",
        "accepted": [
            evidence_observation_card(
                item, quote_chars=deps.config.research_observation_quote_chars
            )
            for item in accepted
        ],
        "rejected": rejected,
        "duplicate_evidence_ids": duplicates,
        "truncated_submission_count": max(0, len(ranked) - len(selected)),
        "archive_evidence_count": len(task_state.evidences),
        "active_evidence_count": len(task_state.active_evidence_ids),
    }
    if any(item.get("reason") == "unknown_document_id" for item in rejected):
        # 引用错 id 的批次统一给有效清单，避免模型拿同一批幻觉 id 反复重试。
        receipt["available_document_ids"] = list(task_state.documents)[-_AVAILABLE_IDS_WINDOW:]
    return receipt


async def _read_candidates(
    deps: ResearcherDeps, task: Task, candidates: list[SearchCandidate]
) -> list[object]:
    semaphore = asyncio.Semaphore(deps.config.research_agent_read_concurrency)

    async def read_one(candidate: SearchCandidate) -> object:
        async with semaphore:
            url = candidate.url
            try:
                result = cast(SearchResult, candidate.model_dump())
                return await asyncio.wait_for(
                    deps.reader_tool.arun(task, result),
                    timeout=deps.config.source_total_timeout,
                )
            except asyncio.TimeoutError:
                return TimeoutError(
                    "来源读取超时 "
                    f"（超过来源总时限 {deps.config.source_total_timeout:.1f}s）：{url}"
                )

    return list(
        await asyncio.gather(
            *(read_one(candidate) for candidate in candidates), return_exceptions=True
        )
    )


def _new_queries(deps: ResearcherDeps, proposed: list[str], seen: list[str]) -> list[str]:
    known = {item.casefold().strip() for item in seen}
    return list(
        dict.fromkeys(
            query.strip()[: deps.config.research_query_chars]
            for query in proposed
            if query.strip() and query.casefold().strip() not in known
        )
    )


def _candidate_cards(
    task_state: ResearcherTaskState,
    raw_results: list[dict[str, object]],
    *,
    start: int,
    count: int,
) -> tuple[list[dict[str, object]], int | None]:
    """按 raw_results 原始下标分页出候选卡，SearchSources 预览与 ListSearchResults 共用。
    同一候选在两处结构一致，offset 基准相同：跳过空 url 时不推进 offset。"""
    cards: list[dict[str, object]] = []
    for raw in raw_results[start : start + count]:
        url = str(raw.get("url", "")).strip()
        if not url:
            continue
        candidate = task_state.candidates.get(
            "c-" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
        )
        if candidate is not None:
            cards.append(_search_candidate_card(candidate))
    next_start = start + count
    return cards, (next_start if next_start < len(raw_results) else None)


def _search_candidate_card(candidate: SearchCandidate) -> dict[str, object]:
    """统一的候选卡结构：始终带截断后的 snippet，让模型在预览阶段就能判断是否值得读取。"""
    return {
        "candidate_id": candidate.candidate_id,
        "title": candidate.title,
        "url": candidate.url,
        "snippet": candidate.snippet[:SEARCH_RESULT_SNIPPET_PREVIEW_CHARS],
        "published_at": candidate.published_at,
        "source_tier": candidate.source_tier,
        "source_profile": candidate.source_profile.model_dump(mode="json"),
    }


def _bounded_support(requested: str, ceiling: str) -> str:
    levels = SUPPORT_ORDER
    requested_index = levels.index(requested) if requested in levels else 0
    ceiling_index = levels.index(ceiling) if ceiling in levels else 0
    return levels[min(requested_index, ceiling_index)]


def complete_task(
    task_state: ResearcherTaskState,
    reason: str,
    selected_evidence_ids: list[str],
    conclusion: str,
) -> dict[str, object]:
    """TaskComplete 业务：校验选定集、落定 stop_reason，返回完整回执。"""
    requested = list(dict.fromkeys(selected_evidence_ids))
    active = set(task_state.active_evidence_ids)
    invalid = [item for item in requested if item not in active]
    if invalid:
        task_state.failures.append("completion_unknown_evidence_ids: " + ", ".join(invalid))
        return {"status": "rejected", "invalid_evidence_ids": invalid}
    if requested:
        task_state.active_evidence_ids = set(requested)
    active_evidences = task_state.active_evidences()
    if active_evidences:
        task_state.conclusion = conclusion.strip()
        task_state.stop_reason = TaskStopReason.COMPLETE
    else:
        # 无证据时的模型自述无从校验，整段换成对当前状态的如实说明。
        task_state.conclusion = "本任务未取得可锚定的 Evidence，因此不给出研究结论。当前状态：" + (
            reason.strip() or "未说明具体的停止原因。"
        )
        task_state.stop_reason = TaskStopReason.BLOCKED_WITHOUT_EVIDENCE
    task_state.stop_detail = reason
    return {
        "status": "accepted",
        "stop_reason": task_state.stop_reason,
        "selected_evidence_ids": [item.evidence_id for item in active_evidences],
    }
