import asyncio

import pytest

from deepresearcher.tools.web.documents import LocalDocumentStore


def test_local_document_store_deduplicates_and_reads_bounded_ranges(tmp_path):
    store = LocalDocumentStore(tmp_path)
    first = asyncio.run(
        store.put(
            text="第一行\n第二行关键词\n第三行\n第四行",
            title="测试文档",
            source_url="https://example.com/a",
            token_count=12,
        )
    )
    second = asyncio.run(
        store.put(
            text="第一行\n第二行关键词\n第三行\n第四行",
            title="测试文档",
            source_url="https://example.com/a",
            token_count=12,
        )
    )

    assert first.document_id == second.document_id
    assert first.line_count == 4
    assert store.path_for(first.document_id).is_file()
    ranges = asyncio.run(store.read(first.document_id, [(2, 4)], max_lines=2, max_chars=1_000))
    assert [(item.start_line, item.end_line) for item in ranges] == [(2, 3)]
    assert ranges[0].content == "L2: 第二行关键词\nL3: 第三行"


def test_local_document_store_grep_returns_literal_context(tmp_path):
    store = LocalDocumentStore(tmp_path)
    ref = asyncio.run(
        store.put(
            text="开头\nQuantum Error Correction\n结论\nquantum network",
            title="Paper",
            source_url="https://example.com/paper",
        )
    )

    matches = asyncio.run(
        store.grep(
            ref.document_id,
            ["quantum"],
            context_lines=1,
            max_matches=1,
            max_chars=1_000,
        )
    )

    assert len(matches) == 1
    assert matches[0].start_line == 1
    assert matches[0].end_line == 3
    assert "L2: Quantum Error Correction" in matches[0].content


def test_local_document_store_keeps_source_identity_separate(tmp_path):
    store = LocalDocumentStore(tmp_path)
    first = asyncio.run(store.put(text="相同正文", title="A", source_url="https://example.com/a"))
    second = asyncio.run(store.put(text="相同正文", title="B", source_url="https://example.com/b"))

    assert first.content_hash == second.content_hash
    assert first.document_id != second.document_id


def test_local_document_store_rejects_path_like_document_id(tmp_path):
    store = LocalDocumentStore(tmp_path)

    with pytest.raises(ValueError, match="document_id"):
        store.path_for("../../etc/passwd")
