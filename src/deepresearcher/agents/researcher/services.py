"""ResearchAgent 方向级工具的业务实现:不感知 langgraph 的纯 async 函数。

每个函数依赖显式入参:deps 是 agent 构造期的稳定零件(跨 run 共享只读),
task/loop_state/event_context/commit_lock 是本轮 run 的名词,由 tools.py
从 ResearcherLoopContext 转交。工具回执的字符串协议留在 tools.py,这里只回 dict。
"""

import asyncio
import hashlib
from typing import Any, cast

from deepresearcher.agents.researcher.state import (
    ResearcherDeps,
    ResearcherLoopState,
    evidence_observation_card,
)
from deepresearcher.evidence.models import Evidence
from deepresearcher.evidence.validator import (
    collapse_whitespace,
    quote_in_source,
    quote_matches_ignoring_punctuation,
    quote_verbatim_strict,
)
from deepresearcher.schemas.limits import (
    SEARCH_RESULT_SNIPPET_PREVIEW_CHARS,
    SEARCH_RESULTS_PREVIEW_COUNT,
)
from deepresearcher.state import SubTask
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
    task: SubTask,
    loop_state: ResearcherLoopState,
    event_context: dict[str, object],
    proposed_queries: list[str],
    reason: str,
) -> dict[str, object]:
    """搜索并返回候选目录；此函数不读取任何来源。"""
    del reason
    remaining = deps.config.research_agent_max_queries - len(loop_state.queries)
    new_queries = _new_queries(deps, proposed_queries, loop_state.queries)[: max(0, remaining)]
    if not new_queries:
        error = "没有新的可执行检索式；请基于已有候选读取来源或调用 Complete。"
        loop_state.failures.append(f"no_novel_queries: {error}")
        return {
            "status": "skipped",
            "reason": "no_novel_queries",
            "proposed_queries": proposed_queries,
        }
    loop_state.queries.extend(new_queries)
    result = SearchToolResult.model_validate(
        await deps.search_tool.arun_queries(task, queries=new_queries)
    )
    loop_state.failures.extend(
        f"search query={item.query}: {item.error}" for item in result.failures
    )
    deps.emit(
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
        loop_state.failures.append(f"search: {error}")
        if getattr(result, "provider_exhausted", False):
            # 把"原因 + 下一步"写清楚交给模型自然收尾，不加控制流分支：
            # 熔断后系统级不可用，重试/换向无意义。
            loop_state.provider_exhausted = True
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
    loop_state.search_batches[search_id] = list(result.queries)
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
        loop_state.candidates[candidate_id] = SearchCandidate(
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
    preview, next_offset = _candidate_cards(
        loop_state, raw_results, start=0, count=SEARCH_RESULTS_PREVIEW_COUNT
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
    task: SubTask,
    loop_state: ResearcherLoopState,
    search_id: str,
    offset: int,
    limit: int,
    reason: str,
) -> dict[str, object]:
    """只允许分页读取当前方向亲自产生的搜索结果。"""
    del reason
    queries = loop_state.search_batches.get(search_id)
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
    cards, next_offset = _candidate_cards(loop_state, raw_results, start=offset, count=limit)
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
    task: SubTask,
    loop_state: ResearcherLoopState,
    event_context: dict[str, object],
    candidate_ids: list[str],
    reason: str,
) -> dict[str, object]:
    """读取模型选中的候选来源，并返回紧凑的工具结果。"""
    del reason
    selected_ids = list(dict.fromkeys(candidate_ids))
    selected = [
        loop_state.candidates[item] for item in selected_ids if item in loop_state.candidates
    ]
    unknown_ids = [item for item in selected_ids if item not in loop_state.candidates]
    if unknown_ids:
        loop_state.failures.append(f"unknown_candidate_ids: {', '.join(unknown_ids)}")
    candidates: list[SearchCandidate] = []
    for candidate in selected:
        if candidate.candidate_id in loop_state.selected_candidate_ids:
            continue
        loop_state.selected_candidate_ids.add(candidate.candidate_id)
        loop_state.read_urls.append(candidate.url)
        candidates.append(candidate)

    read_results = await _read_candidates(deps, task, candidates)
    documents: list[DocumentView] = []
    inline_tokens = 0
    for candidate, read_result in zip(candidates, read_results, strict=True):
        url = candidate.url
        if isinstance(read_result, asyncio.CancelledError):
            raise read_result
        if isinstance(read_result, Exception):
            loop_state.failures.append(f"{url}: {read_result}")
            deps.emit(
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
                        view.token_count + inline_tokens
                        > deps.config.document_inline_total_max_tokens
                    ):
                        view = view.model_copy(update={"inline": False, "content": ""})
                    else:
                        inline_tokens += view.token_count
                loop_state.documents[view.document_id] = view
                documents.append(view)
                if view.source_url:
                    loop_state.source_refs.append(view.source_url)
        elif result.status == "skipped":
            reason_code = result.reason_code or "unknown"
            loop_state.skipped.append(reason_code)
            deps.emit(
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
            loop_state.failures.append(f"{url}: {error}")
            deps.emit(
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
        "archive_evidence_count": len(loop_state.evidences),
        "active_evidence_count": len(loop_state.active_evidence_ids),
        "active_evidence_limit": loop_state.active_evidence_limit,
        "archive_evidence_limit": loop_state.evidence_archive_limit,
        "skip_reasons": sorted(set(loop_state.skipped)),
        "recent_failures": loop_state.failures[-4:],
    }


async def grep_document(
    deps: ResearcherDeps,
    loop_state: ResearcherLoopState,
    document_id: str,
    query: str,
    context_lines: int,
    offset: int,
    reason: str,
) -> dict[str, object]:
    del reason
    if document_id not in loop_state.documents:
        return {"status": "rejected", "reason": "unknown_document_id"}
    if deps.material_store is None:
        return {"status": "failed", "reason": "material_store_unavailable"}
    result = await deps.material_store.grep(
        document_id,
        query.strip(),
        context_lines=min(context_lines, deps.config.document_grep_context_lines),
        max_matches=deps.config.document_grep_max_matches,
        max_chars=deps.config.document_grep_max_chars,
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


async def read_document(
    deps: ResearcherDeps,
    loop_state: ResearcherLoopState,
    document_id: str,
    ranges: list[tuple[int, int]],
    reason: str,
) -> dict[str, object]:
    del reason
    if document_id not in loop_state.documents:
        return {"status": "rejected", "reason": "unknown_document_id"}
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


async def add_evidence(
    deps: ResearcherDeps,
    task: SubTask,
    loop_state: ResearcherLoopState,
    event_context: dict[str, object],
    commit_lock: asyncio.Lock,
    submissions: list[dict[str, object]],
    reason: str,
) -> dict[str, object]:
    if deps.material_store is None:
        return {"status": "failed", "reason": "material_store_unavailable"}
    accepted: list[Evidence] = []
    rejected: list[dict[str, object]] = []
    duplicates: list[str] = []
    pending_ids: set[str] = set()
    candidates: list[tuple[int, str, Evidence]] = []
    accepted_via_normalization = 0  # 逐字被连字符/ligature 编码差异卡住、靠归一救回的条数
    # 批内 memo：同文档多条引用只取一次原文(Redis 后端下省 N-1 次全量 GET)。
    source_texts: dict[str, str] = {}
    for index, raw in enumerate(submissions):
        document_id = str(raw.get("document_id", ""))
        document = loop_state.documents.get(document_id)
        if document is None:
            rejected.append({"index": index, "reason": "unknown_document_id"})
            continue
        claim = str(raw.get("claim", "")).strip()
        quote = str(raw.get("quote", "")).strip()
        source_text = source_texts.get(document_id)
        if source_text is None:
            try:
                source_text = await deps.material_store.text(document_id)
            except (FileNotFoundError, KeyError) as exc:
                rejected.append({"index": index, "reason": str(exc)})
                continue
            source_texts[document_id] = source_text
        # 入池不变式：quote 逐字（忽略空白 + 连字符/ligature 编码差异）出现在该来源正文里。
        # 宽松仍不过时再分一档：只差异标点/引号/破折号（词序列一致）→ quote_format_variant；
        # 词都不同 → quote_paraphrase。这样能真正区分"格式误杀"与"模型改述"。
        if not quote_in_source(source_text, quote):
            rejection = (
                "quote_format_variant"
                if quote_matches_ignoring_punctuation(source_text, quote)
                else "quote_paraphrase"
            )
            rejected.append({"index": index, "reason": rejection})
            continue
        if not quote_verbatim_strict(source_text, quote):
            accepted_via_normalization += 1
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
            subtask_id=task["id"],
            research_direction=task["question"],
            claim=claim,
            quote=quote,
            source_url=document.source_url,
            source_title=document.title,
            published_at=document.published_at,
            source_profile=describe_source(document.source_url),
            retrieval_method=document.retrieval_method,
            support=cast(Any, support),
            confidence=confidence,
        )
        candidates.append((index, document_id, evidence))
        pending_ids.add(evidence_id)

    ranked = sorted(candidates, key=lambda item: item[2].confidence, reverse=True)
    selected = ranked[: deps.config.evidence_add_batch_size]
    # 原文读取和引用校验可并发；只有去重、容量与入池是短临界区。
    async with commit_lock:
        existing_ids = {item.evidence_id for item in loop_state.evidences}
        per_source = {
            source_url: sum(1 for item in loop_state.evidences if item.source_url == source_url)
            for source_url in {document.source_url for document in loop_state.documents.values()}
        }
        for index, document_id, evidence in selected:
            if evidence.evidence_id in existing_ids:
                duplicates.append(evidence.evidence_id)
                continue
            document = loop_state.documents[document_id]
            source_count = per_source.get(document.source_url, 0)
            if source_count >= deps.config.evidence_max_per_source:
                rejected.append({"index": index, "reason": "source_evidence_limit_reached"})
                continue
            added = loop_state.add_evidences([evidence])
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
                item, quote_chars=deps.config.research_observation_quote_chars
            )
            for item in accepted
        ],
        "rejected": rejected,
        "duplicate_evidence_ids": duplicates,
        "truncated_submission_count": max(0, len(ranked) - len(selected)),
        "archive_evidence_count": len(loop_state.evidences),
        "active_evidence_count": len(loop_state.active_evidence_ids),
    }


async def _read_candidates(
    deps: ResearcherDeps, task: SubTask, candidates: list[SearchCandidate]
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
    loop_state: ResearcherLoopState,
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
        candidate = loop_state.candidates.get(
            "c-" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
        )
        if candidate is not None:
            cards.append(_search_candidate_card(candidate))
    next_start = start + count
    return cards, (next_start if next_start < len(raw_results) else None)


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


def _bounded_support(requested: str, ceiling: str) -> str:
    levels = SUPPORT_ORDER
    requested_index = levels.index(requested) if requested in levels else 0
    ceiling_index = levels.index(ceiling) if ceiling in levels else 0
    return levels[min(requested_index, ceiling_index)]
