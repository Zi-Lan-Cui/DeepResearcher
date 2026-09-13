import asyncio
from datetime import datetime, timedelta, timezone

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import func, select, update

from deepresearcher.config import LLMConfig, SearchConfig
from deepresearcher.evidence.extractor import EvidenceExtractor
from deepresearcher.evidence.models import EvidenceExtraction, ExtractedEvidence
from deepresearcher.service.persistence.database import init_db, make_engine, make_session_factory
from deepresearcher.service.persistence.models import Run, RunUsage, ToolCacheEntry, User
from deepresearcher.service.persistence.tool_cache import PostgresToolCache
from deepresearcher.service.usage import (
    UsageRuntime,
    UsageStore,
    bind_usage_runtime,
    reset_usage_runtime,
)
from deepresearcher.tools.cache import CacheValue
from deepresearcher.tools.web import DirectHttpFetchProvider, FetchService
from deepresearcher.tools.web.search.service import SearchTool

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def cache_db(tmp_path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'cache.db'}")
    sessions = make_session_factory(engine)
    await init_db(engine)
    async with sessions() as session:
        user = User(email="cache@example.com", password_hash="x")
        session.add(user)
        await session.flush()
        session.add(Run(id="run-cache", user_id=user.id, query="cache"))
        await session.commit()
    try:
        yield sessions
    finally:
        await engine.dispose()


async def test_persistent_cache_singleflight_ttl_and_failure(cache_db):
    cache = PostgresToolCache(cache_db)
    calls = 0

    async def compute():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return CacheValue(value={"ok": True}, metrics={"saved_external_requests": 1})

    first, second = await asyncio.gather(
        cache.get_or_compute(
            "search", "same", ttl_seconds=60, schema_version="v1", compute=compute
        ),
        cache.get_or_compute(
            "search", "same", ttl_seconds=60, schema_version="v1", compute=compute
        ),
    )
    assert calls == 1
    assert sorted((first.hit, second.hit)) == [False, True]

    async with cache_db() as session:
        row = await session.get(ToolCacheEntry, ("search", "same"))
        assert row is not None and row.hit_count == 1
        await session.execute(
            update(ToolCacheEntry)
            .where(ToolCacheEntry.namespace == "search", ToolCacheEntry.cache_key == "same")
            .values(expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))
        )
        await session.commit()
    assert await cache.delete_expired() == 1

    failures = 0

    async def fail():
        nonlocal failures
        failures += 1
        raise RuntimeError("provider failed")

    for _ in range(2):
        with pytest.raises(RuntimeError, match="provider failed"):
            await cache.get_or_compute(
                "search", "failure", ttl_seconds=60, schema_version="v1", compute=fail
            )
    assert failures == 2
    async with cache_db() as session:
        assert await session.get(ToolCacheEntry, ("search", "failure")) is None


async def test_cancelled_compute_is_not_cached(cache_db):
    cache = PostgresToolCache(cache_db)

    async def cancel():
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await cache.get_or_compute(
            "fetch", "cancel", ttl_seconds=60, schema_version="v1", compute=cancel
        )
    async with cache_db() as session:
        assert await session.get(ToolCacheEntry, ("fetch", "cancel")) is None


async def test_cache_storage_failure_fails_open():
    def broken_factory():
        raise RuntimeError("database unavailable")

    calls = 0

    async def compute():
        nonlocal calls
        calls += 1
        return CacheValue(value={"fresh": True})

    result = await PostgresToolCache(broken_factory).get_or_compute(
        "search", "key", ttl_seconds=60, schema_version="v1", compute=compute
    )
    assert result.value == {"fresh": True}
    assert result.hit is False
    assert calls == 1


class _SearchClient:
    provider_name = "fake"
    effective_limit = 5

    def __init__(self):
        self.calls = 0

    async def asearch(self, query):
        self.calls += 1
        return [
            {
                "title": query,
                "url": "https://example.com/result",
                "snippet": "result",
                "score": 0.9,
            }
        ]


def _task(task_id="task-1", question="Redis 缓存"):
    return {
        "id": task_id,
        "question": question,
        "type": "search",
        "status": "pending",
        "assigned_agent": "search",
    }


async def test_l1_search_cache_survives_tool_recreation(cache_db):
    cache = PostgresToolCache(cache_db)
    client = _SearchClient()
    first = SearchTool(client, tool_cache=cache, cache_ttl_seconds=60, cache_version="search-v1")
    second = SearchTool(client, tool_cache=cache, cache_ttl_seconds=60, cache_version="search-v1")

    one = await first.arun(_task("task-1", "  Redis   缓存 "))
    two = await second.arun(_task("task-2", "redis 缓存"))
    assert one.status == two.status == "completed"
    assert client.calls == 1

    changed = SearchTool(client, tool_cache=cache, cache_ttl_seconds=60, cache_version="search-v2")
    await changed.arun(_task("task-3", "redis 缓存"))
    assert client.calls == 2


class _HttpClient:
    def __init__(self, *, content_type="text/html"):
        self.calls = 0
        self.content_type = content_type

    async def arequest(self, _method, url, **_kwargs):
        self.calls += 1
        return httpx.Response(
            200,
            content=b"<html><title>Cached</title><body><p>stable text</p></body></html>",
            headers={"content-type": self.content_type},
            request=httpx.Request("GET", url),
        )


async def test_l2_fetch_parse_cache_survives_fetcher_recreation(cache_db, monkeypatch):
    async def inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("deepresearcher.tools.web.fetch.providers.direct.asyncio.to_thread", inline)
    cache = PostgresToolCache(cache_db)
    client = _HttpClient()
    kwargs = {
        "tool_cache": cache,
        "cache_ttl_seconds": 60,
        "fetch_policy_version": "fetch-v1",
        "parser_version": "parser-v1",
    }
    first = await FetchService([DirectHttpFetchProvider(SearchConfig(), client)], **kwargs).afetch(
        "https://EXAMPLE.com/page#part"
    )
    second = await FetchService([DirectHttpFetchProvider(SearchConfig(), client)], **kwargs).afetch(
        "https://example.com/page"
    )
    assert client.calls == 1
    assert first.get("cache_hit") is False
    assert second.get("cache_hit") is True
    assert second["text"] == first["text"]
    assert second["source_url"] == "https://example.com/page"


async def test_l2_failed_parse_is_not_cached(cache_db, monkeypatch):
    async def inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("deepresearcher.tools.web.fetch.providers.direct.asyncio.to_thread", inline)
    cache = PostgresToolCache(cache_db)
    client = _HttpClient(content_type="application/octet-stream")
    fetcher = FetchService(
        [DirectHttpFetchProvider(SearchConfig(), client)],
        tool_cache=cache,
        cache_ttl_seconds=60,
    )
    assert (await fetcher.afetch("https://example.com/binary"))["status"] == "failed"
    assert (await fetcher.afetch("https://example.com/binary"))["status"] == "failed"
    assert client.calls == 2


def _document():
    return {
        "title": "Source",
        "final_url": "https://example.com/source",
        "text": "Redis can cache data.",
        "content_hash": "a" * 64,
        "blocks": [
            {
                "block_id": "b-1",
                "block_type": "paragraph",
                "text": "Redis can cache data.",
                "heading_path": [],
                "order": 1,
            }
        ],
    }


def _extractor(cache):
    return EvidenceExtractor(
        object(),
        tool_cache=cache,
        cache_ttl_seconds=60,
        extractor_prompt_version="prompt-v1",
        evidence_schema_version="schema-v1",
        chunking_version="chunks-v1",
        model_id="model-v1",
        input_usd_per_million=1,
        output_usd_per_million=2,
    )


async def test_l3_cache_revalidates_and_rebinds_evidence_identity(cache_db):
    cache = PostgresToolCache(cache_db)
    calls = 0

    async def extract(_task, _document, _result, _chunks):
        nonlocal calls
        calls += 1
        return (
            [
                EvidenceExtraction(
                    evidences=[
                        ExtractedEvidence(
                            claim="Redis caches data",
                            quote="Redis can cache data.",
                            confidence=0.9,
                        )
                    ]
                )
            ],
            0,
        )

    first_extractor = _extractor(cache)
    first_extractor._extract_chunks = extract
    first = await first_extractor.aextract_result(
        _task("task-old"), _document(), {"url": "https://example.com/source"}
    )

    second_extractor = _extractor(cache)
    second_extractor._extract_chunks = extract
    usage_token = bind_usage_runtime(UsageRuntime("run-cache", UsageStore(cache_db), LLMConfig()))
    try:
        second = await second_extractor.aextract_result(
            _task("task-new"), _document(), {"url": "https://example.com/source"}
        )
    finally:
        reset_usage_runtime(usage_token)

    assert calls == 1
    assert first.cache_hit is False and second.cache_hit is True
    assert first.evidences[0].subtask_id == "task-old"
    assert second.evidences[0].subtask_id == "task-new"
    assert first.evidences[0].evidence_id != second.evidences[0].evidence_id
    assert second.evidences[0].locator.block_ids == ["b-1"]

    async with cache_db() as session:
        run = await session.get(Run, "run-cache")
        cache_events = await session.scalar(
            select(func.count()).select_from(RunUsage).where(RunUsage.category == "cache")
        )
    assert run is not None
    assert run.cache_hit_count == 1
    assert run.saved_llm_call_count == 1
    assert run.saved_tokens > 0
    assert cache_events == 1


async def test_l3_all_failed_chunks_are_not_cached(cache_db):
    cache = PostgresToolCache(cache_db)
    calls = 0

    async def fail_chunks(_task, _document, _result, _chunks):
        nonlocal calls
        calls += 1
        return [EvidenceExtraction()], 1

    for task_id in ("task-a", "task-b"):
        extractor = _extractor(cache)
        extractor._extract_chunks = fail_chunks
        result = await extractor.aextract_result(
            _task(task_id), _document(), {"url": "https://example.com/source"}
        )
        assert result.failed_chunk_count == 1
    assert calls == 2


async def test_l3_confirmed_empty_result_is_cached(cache_db):
    cache = PostgresToolCache(cache_db)
    calls = 0

    async def empty_chunks(_task, _document, _result, _chunks):
        nonlocal calls
        calls += 1
        return [EvidenceExtraction()], 0

    outcomes = []
    for task_id in ("task-empty-a", "task-empty-b"):
        extractor = _extractor(cache)
        extractor._extract_chunks = empty_chunks
        outcomes.append(
            await extractor.aextract_result(
                _task(task_id), _document(), {"url": "https://example.com/source"}
            )
        )
    assert calls == 1
    assert outcomes[0].cache_hit is False and outcomes[1].cache_hit is True
    assert outcomes[0].evidences == outcomes[1].evidences == []
