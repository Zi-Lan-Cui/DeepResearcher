import pytest

import deepresearcher.service.persistence.redis_material_store as rms_module
from deepresearcher.service.persistence.redis_material_store import (
    RedisResearchMaterialStore,
)
from deepresearcher.tools.web.materials import MemoryResearchMaterialStore, SearchResultSet
from deepresearcher.tools.web.materials.store import grep_lines, read_lines


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
        max_line_chars=2_000,
    )
    assert result.matches[0].content == "L1: alpha\nL2: beta keyword\nL3: gamma"
    assert result.total_matches == 1 and not result.has_more and result.next_offset == 0


@pytest.mark.asyncio
async def test_grep_reports_total_has_more_and_pages_via_offset_with_whole_windows():
    """截断可见、可翻页;字符预算容不下新窗口时整窗留给下一页,单行超行预算则裁切并标注。"""
    store = MemoryResearchMaterialStore()
    ref = await store.put(
        text="\n".join(f"L{index} match" for index in range(1, 21)),
        title="paging",
        source_url="https://example.com",
    )

    # 首页:context_lines=0 时每窗单行,max_matches=3 返回前 3 窗并显式给出 has_more。
    first = await store.grep(
        ref.document_id,
        "match",
        context_lines=0,
        max_matches=3,
        max_chars=1_000,
        max_line_chars=2_000,
    )
    assert first.total_matches == 20
    assert [m.start_line for m in first.matches] == [1, 2, 3]
    assert first.has_more and first.next_offset == 3

    # 续页:带 next_offset 取后续窗口,与首页不重叠、不遗漏。
    second = await store.grep(
        ref.document_id,
        "match",
        context_lines=0,
        max_matches=3,
        max_chars=1_000,
        max_line_chars=2_000,
        offset=first.next_offset,
    )
    assert [m.start_line for m in second.matches] == [4, 5, 6]
    assert second.has_more and second.next_offset == 6

    # 字符预算触顶:第二个窗口连首行都装不下时整窗留给下一页,has_more 保留,首窗完整。
    # 每行原文 "L{n} match"，grep 再加 "L{n}: " 前缀；首窗 (context_lines=2, 命中第1行)
    # = "L1: L1 match\nL2: L2 match\nL3: L3 match" 共 38 字符;max_chars=40 容首窗拒次窗。
    tight = await store.grep(
        ref.document_id,
        "match",
        context_lines=2,
        max_matches=3,
        max_chars=40,
        max_line_chars=2_000,
    )
    assert len(tight.matches) == 1
    assert tight.matches[0].content == "L1: L1 match\nL2: L2 match\nL3: L3 match"
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
        SearchResultSet(
            search_id="s1", run_id="r1", queries=["q"], results=[{"url": "https://e.com"}]
        )
    )
    events.clear()

    await store.get_search_results("r1", "s1")
    hits = [e for e in events if e[0] == "material_search" and e[1] == "hit"]
    assert hits, "search 命中未记录 cache hit 事件"
    assert hits[0][2] == {"saved_external_requests": 1}
    await store.close()


def test_grep_lines_clips_oversized_line_within_line_budget():
    """单行巨型文本裁到行预算并显式标注。"""
    giant_line = "keyword " + "x" * 40_000
    result = grep_lines(
        [giant_line],
        "keyword",
        context_lines=2,
        max_matches=12,
        max_chars=12_000,
        max_line_chars=2_000,
    )
    assert len(result.matches) == 1
    content = result.matches[0].content
    assert len(content) <= 2_000
    assert "本行其余" in content
    assert "ReadDocument" in content
    assert result.matches[0].tail_omitted_lines == 0
    assert result.has_more is False


def test_grep_lines_window_tail_cut_is_marked_and_skipped_on_next_page():
    """窗口尾部因页预算裁掉:行号标注 + tail_omitted_lines;下一页从下一窗口起,不回补被裁尾行。"""
    # 命中行相隔 10,避免相邻命中窗口重叠;上下文行同样长，凑出"单窗放不进一页"的裁尾。
    lines = [
        ("alpha " if index in (0, 10) else "body ") + f"{index} " + "z" * 390 for index in range(20)
    ]
    result = grep_lines(
        lines,
        "alpha",
        context_lines=2,
        max_matches=12,
        max_chars=1_200,
        max_line_chars=2_000,
    )
    assert result.total_matches == 2
    first = result.matches[0]
    assert first.start_line == 1
    assert first.end_line == 2
    assert first.tail_omitted_lines == 1
    assert "L3-L3" in first.content
    assert len(first.content) <= 1_200
    assert result.has_more is True
    assert result.next_offset == len(result.matches)

    second_page = grep_lines(
        lines,
        "alpha",
        context_lines=2,
        max_matches=12,
        max_chars=1_200,
        max_line_chars=2_000,
        offset=result.next_offset,
    )
    assert second_page.matches[0].start_line == 9
    assert second_page.matches[0].end_line == 10
    assert second_page.matches[0].tail_omitted_lines == 3
    assert second_page.has_more is False


def test_grep_lines_page_total_never_exceeds_char_budget():
    """整页字符总和不超过 max_chars:预算不足时少发窗口,不超额发出。"""
    lines = [f"needle {index} " + "w" * 300 for index in range(30)]
    result = grep_lines(
        lines,
        "needle",
        context_lines=1,
        max_matches=12,
        max_chars=1_500,
        max_line_chars=2_000,
    )
    assert result.matches
    assert sum(len(item.content) for item in result.matches) <= 1_500


def test_read_lines_marks_clipped_oversized_line():
    """ReadDocument 整行超预算时裁切标注,不静默截断。"""
    lines = ["head", "big " + "q" * 3_000]
    ranges = read_lines(lines, [(2, 2)], max_lines=240, max_chars=1_000)
    assert len(ranges) == 1
    content = ranges[0].content
    assert len(content) <= 1_000
    assert "本行其余" in content
