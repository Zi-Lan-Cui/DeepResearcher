import pytest

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
    assert (
        await store.grep(
            ref.document_id,
            ["keyword"],
            context_lines=1,
            max_matches=5,
            max_chars=1_000,
        )
    )[0].content == "L1: alpha\nL2: beta keyword\nL3: gamma"


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
