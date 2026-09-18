import pytest

import deepresearcher.service.persistence.redis_material_store as rms_module
from deepresearcher.service.persistence.redis_material_store import (
    RedisResearchMaterialStore,
)
from deepresearcher.tools.web.materials import MemoryResearchMaterialStore, SearchResultSet


@pytest.mark.asyncio
async def test_memory_material_store_keeps_searches_and_readable_documents():
    store = MemoryResearchMaterialStore()
    result_set = SearchResultSet(
        search_id="search-1",
        run_id="run-1",
        queries=["test query"],
        results=[{"title": "One", "url": "https://example.com", "raw_content": "hidden"}],
    )
    await store.put_search_results(result_set)

    restored = await store.get_search_results("run-1", "search-1")
    assert restored.results[0]["raw_content"] == "hidden"
    with pytest.raises(FileNotFoundError):
        await store.get_search_results("run-2", "search-1")

    ref = await store.put(
        text="alpha\nbeta keyword\ngamma",
        title="Document",
        source_url="https://example.com",
        token_count=3,
        fetch_key="fetch-1",
    )
    assert await store.resolve_fetch("fetch-1") == ref
    assert (await store.read(ref.document_id, [(2, 3)], max_lines=10, max_chars=1_000))[
        0
    ].content == "L2: beta keyword\nL3: gamma"
    result = await store.grep(
        ref.document_id,
        "keyword",
        context_lines=1,
        max_matches=5,
        max_chars=1_000,
    )
    assert result.matches[0].content == "L1: alpha\nL2: beta keyword\nL3: gamma"
    assert result.total_matches == 1 and not result.has_more and result.next_offset == 0


@pytest.mark.asyncio
async def test_grep_reports_total_has_more_and_pages_via_offset_with_whole_windows():
    """截断可见、可翻页;字符预算触顶时整窗丢弃,不截半行。"""
    store = MemoryResearchMaterialStore()
    ref = await store.put(
        text="\n".join(f"L{index} match" for index in range(1, 21)),
        title="paging",
        source_url="https://example.com",
    )

    # 首页:context_lines=0 每窗单行;max_matches=3 → 返回前 3 窗 + 显式 has_more。
    first = await store.grep(
        ref.document_id, "match", context_lines=0, max_matches=3, max_chars=1_000
    )
    assert first.total_matches == 20
    assert [m.start_line for m in first.matches] == [1, 2, 3]
    assert first.has_more and first.next_offset == 3

    # 续页:带 next_offset → 取后续窗口,与首页不重叠、不遗漏。
    second = await store.grep(
        ref.document_id,
        "match",
        context_lines=0,
        max_matches=3,
        max_chars=1_000,
        offset=first.next_offset,
    )
    assert [m.start_line for m in second.matches] == [4, 5, 6]
    assert second.has_more and second.next_offset == 6

    # 字符预算触顶:第二个窗口装不下 → 整窗跳过(has_more 保留),首窗仍完整未截半行。
    # 每行原文 "L{n} match"，grep 再加 "L{n}: " 前缀；首窗 (context_lines=2, 命中第1行)
    # = "L1: L1 match\nL2: L2 match\nL3: L3 match" 共 38 字符;max_chars=40 容首窗拒次窗。
    tight = await store.grep(
        ref.document_id, "match", context_lines=2, max_matches=3, max_chars=40
    )
    assert len(tight.matches) == 1
    assert tight.matches[0].content == "L1: L1 match\nL2: L2 match\nL3: L3 match"  # 完整,非截半
    assert tight.has_more and tight.next_offset == 1


class _Pipeline:
    def __init__(self, redis):
        self.redis = redis
        self.commands = []

    def set(self, key, value, *, ex):
        self.commands.append((key, value, ex))
        return self

    async def execute(self):
        for key, value, _ex in self.commands:
            self.redis.values[key] = value


class _Redis:
    def __init__(self):
        self.values = {}
        self.closed = False

    async def set(self, key, value, *, ex):
        del ex
        self.values[key] = value

    async def get(self, key):
        return self.values.get(key)

    async def delete(self, key):
        self.values.pop(key, None)

    def pipeline(self, *, transaction):
        del transaction
        return _Pipeline(self)

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_redis_material_store_uses_run_scoped_search_key_and_fetch_index():
    redis = _Redis()
    store = RedisResearchMaterialStore(
        redis,
        key_prefix="test:material",
        search_ttl_seconds=60,
        document_ttl_seconds=60,
    )
    await store.put_search_results(
        SearchResultSet(
            search_id="search-1",
            run_id="run-1",
            queries=["query"],
            results=[{"url": "https://example.com"}],
        )
    )
    assert "test:material:search:run-1:search-1" in redis.values
    restored = await store.get_search_results("run-1", "search-1")
    assert restored.results == [{"url": "https://example.com"}]

    ref = await store.put(
        text="first\nsecond",
        title="Document",
        source_url="https://example.com",
        fetch_key="fetch-1",
    )
    assert (await store.resolve_fetch("fetch-1")).document_id == ref.document_id
    assert await store.text(ref.document_id) == "first\nsecond"
    await store.close()
    assert redis.closed is True


@pytest.mark.asyncio
async def test_search_cache_hit_records_saved_external_request(monkeypatch):
    """回归:material_search 命中必须带 saved_external_requests，否则 cache-reuse 断言误判。"""
    events: list[tuple[str, str, dict | None]] = []

    async def fake_event(*, namespace: str, status: str, detail: dict | None = None) -> None:
        events.append((namespace, status, detail))

    monkeypatch.setattr(rms_module, "record_cache_event", fake_event)
    redis = _Redis()
    store = RedisResearchMaterialStore(
        redis, key_prefix="t", search_ttl_seconds=60, document_ttl_seconds=60
    )
    await store.put_search_results(
        SearchResultSet(search_id="s1", run_id="r1", queries=["q"], results=[{"url": "https://e.com"}])
    )
    events.clear()  # 去掉 put 的 write 事件

    await store.get_search_results("r1", "s1")  # 命中缓存
    hits = [e for e in events if e[0] == "material_search" and e[1] == "hit"]
    assert hits, "search 命中未记录 cache hit 事件"
    assert hits[0][2] == {"saved_external_requests": 1}
    await store.close()
