import asyncio

import pytest

from deepresearcher.config import LLMRetryConfig
from deepresearcher.evidence import EvidenceExtractor
from deepresearcher.evidence.models import Evidence, EvidenceExtraction, ExtractedEvidence
from deepresearcher.evidence.retrieval import BM25Retriever, select_blocks
from deepresearcher.llm import LLMConfigurationError, structured
from deepresearcher.observability.events import JsonlSink


def test_evidence_audit_chunk_is_persisted_but_hidden_from_agent() -> None:
    evidence = Evidence(
        evidence_id="ev-1",
        subtask_id="task-1",
        research_direction="测试",
        claim="结论",
        quote="可核对原文",
        source_url="https://example.com",
        audit_chunk="前文 可核对原文 后文",
    )

    assert evidence.model_dump()["audit_chunk"] == "前文 可核对原文 后文"
    assert "audit_chunk" not in evidence.agent_payload()


def test_validator_persists_quote_locator() -> None:
    from deepresearcher.evidence.validator import validate_evidence

    evidence = Evidence(
        evidence_id="ev-locator",
        subtask_id="task-1",
        research_direction="测试",
        claim="A 的延迟为 20ms",
        quote="A 的平均延迟为 20ms。",
        source_url="https://example.com",
    )

    validated = validate_evidence(evidence, blocks())

    assert validated.locator.block_ids == ["b-1"]
    assert validated.locator.heading_path == ["性能测试"]


def test_audit_chunk_keeps_quote_and_is_bounded() -> None:
    quote = "需要保留的原文"
    source_blocks = [
        {
            "block_id": "b1",
            "block_type": "paragraph",
            "text": "甲" * 20_000 + quote + "乙" * 20_000,
            "heading_path": ["标题"],
            "order": 0,
        }
    ]

    chunk = EvidenceExtractor._audit_chunk(source_blocks, quote)

    assert quote in chunk
    assert len(chunk) <= 16_004


def blocks():
    return [
        {
            "block_id": "b-0",
            "block_type": "heading",
            "text": "性能测试",
            "heading_path": ["性能测试"],
            "order": 0,
        },
        {
            "block_id": "b-1",
            "block_type": "paragraph",
            "text": "在相同硬件环境下，A 的平均延迟为 20ms。",
            "heading_path": ["性能测试"],
            "order": 1,
        },
        {
            "block_id": "b-2",
            "block_type": "paragraph",
            "text": "该结果仅适用于英文数据集。",
            "heading_path": ["性能测试"],
            "order": 2,
        },
        {
            "block_id": "b-3",
            "block_type": "paragraph",
            "text": "文章还讨论了部署成本。",
            "heading_path": ["成本"],
            "order": 3,
        },
    ]


class CharEstimator:
    """测试专用确定性估算器，避免断言依赖 tokenizer 词表。"""

    def count(self, text: str) -> int:
        return len(text)


def test_lexical_retrieval_keeps_adjacent_qualification():
    selected = select_blocks("A 平均延迟", blocks(), top_k=1, window=1)
    assert [block["block_id"] for block in selected] == ["b-0", "b-1", "b-2"]


def test_evidence_extractor_requires_llm_at_construction():
    with pytest.raises(LLMConfigurationError):
        EvidenceExtractor(None)


def test_llm_empty_evidence_does_not_fall_back_to_source_title(monkeypatch):
    async def empty_extraction(llm, schema, messages, **kwargs):
        return EvidenceExtraction(evidences=[])

    monkeypatch.setattr("deepresearcher.evidence.extractor.ainvoke_structured", empty_extraction)
    document = {
        "title": "只有标题的来源",
        "final_url": "https://example.com",
        "text": "这段内容没有支持当前研究任务的事实。",
        "blocks": blocks(),
    }
    result = {"title": "搜索标题", "url": "https://example.com", "snippet": "", "score": 0.8}

    evidences = asyncio.run(
        EvidenceExtractor(llm=object()).aextract(
            {
                "id": "r1-1",
                "question": "不存在的主题",
                "type": "search",
                "status": "pending",
                "assigned_agent": "search",
            },
            document,
            result,
        )
    )

    assert evidences == []


def test_short_document_sends_every_block_to_extraction_prompt(monkeypatch):
    seen_contexts = []

    async def extract_each_chunk(llm, schema, messages, **kwargs):
        context = messages[-1].content
        seen_contexts.append(context)
        return EvidenceExtraction(evidences=[])

    monkeypatch.setattr("deepresearcher.evidence.extractor.ainvoke_structured", extract_each_chunk)
    document = {
        "title": "长文",
        "final_url": "https://example.com/long",
        "text": "证据甲\n无关段落\n证据乙",
        "blocks": [
            {
                "block_id": "b-1",
                "block_type": "paragraph",
                "text": "证据甲",
                "heading_path": ["甲"],
                "order": 1,
            },
            {
                "block_id": "b-2",
                "block_type": "paragraph",
                "text": "无关段落",
                "heading_path": ["中"],
                "order": 2,
            },
            {
                "block_id": "b-3",
                "block_type": "paragraph",
                "text": "证据乙",
                "heading_path": ["乙"],
                "order": 3,
            },
        ],
    }
    result = {"title": "长文", "url": "https://example.com/long", "snippet": "", "score": 0.8}
    extractor = EvidenceExtractor(
        llm=object(),
        input_budget_tokens=1_000,
        full_context_max_tokens=1_000,
        estimator=CharEstimator(),
    )

    outcome = asyncio.run(
        extractor.aextract_result(
            {
                "id": "r1-1",
                "question": "不匹配的查询词",
                "type": "search",
                "status": "pending",
                "assigned_agent": "search",
            },
            document,
            result,
        )
    )

    assert outcome.strategy == "full"
    assert outcome.chunk_count == 1
    assert len(seen_contexts) == 1
    assert all(text in seen_contexts[0] for text in ("证据甲", "无关段落", "证据乙"))


def test_long_document_sends_only_recalled_blocks_and_validator_sees_same_blocks(monkeypatch):
    captured: dict[str, object] = {}

    class OneBlockRetriever:
        def select(self, blocks, query, top_k, window):
            assert query == "目标问题"
            return [blocks[1]]

    async def fake_invoke(llm, schema, messages, **kwargs):
        captured["prompt"] = messages[-1].content
        return EvidenceExtraction(evidences=[ExtractedEvidence(claim="目标事实", quote="目标证据")])

    def fake_validate(evidence, selected):
        captured["validator_blocks"] = selected
        return evidence

    monkeypatch.setattr("deepresearcher.evidence.extractor.ainvoke_structured", fake_invoke)
    monkeypatch.setattr("deepresearcher.evidence.extractor.validate_evidence", fake_validate)
    document_blocks = [
        {
            "block_id": "b-1",
            "block_type": "paragraph",
            "text": "无关内容" * 20,
            "heading_path": [],
            "order": 0,
        },
        {
            "block_id": "b-2",
            "block_type": "paragraph",
            "text": "目标证据",
            "heading_path": ["目标章节"],
            "order": 1,
        },
    ]
    outcome = asyncio.run(
        EvidenceExtractor(
            llm=object(),
            estimator=CharEstimator(),
            full_context_max_tokens=80,
            retriever=OneBlockRetriever(),
        ).aextract_result(
            {
                "id": "r1-1",
                "question": "目标问题",
                "type": "search",
                "status": "pending",
                "assigned_agent": "search",
            },
            {
                "title": "长文",
                "final_url": "https://example.com/long",
                "text": "\n".join(block["text"] for block in document_blocks),
                "blocks": document_blocks,
            },
            {"title": "长文", "url": "https://example.com/long", "snippet": "", "score": 0.8},
        )
    )

    assert outcome.strategy == "bm25_recall"
    assert "目标证据" in str(captured["prompt"])
    assert "无关内容" not in str(captured["prompt"])
    assert [block["block_id"] for block in captured["validator_blocks"]] == ["b-2"]


def test_multi_subquestion_recall_unions_and_deduplicates_blocks(monkeypatch):
    seen_queries: list[str] = []
    captured = {}

    class MultiRetriever:
        def select(self, blocks, query, top_k, window):
            seen_queries.append(query)
            return [blocks[0], blocks[1]] if "子问题甲" in query else [blocks[1], blocks[2]]

    async def fake_invoke(llm, schema, messages, **kwargs):
        captured["prompt"] = messages[-1].content
        return EvidenceExtraction(evidences=[])

    monkeypatch.setattr("deepresearcher.evidence.extractor.ainvoke_structured", fake_invoke)
    document_blocks = [
        {
            "block_id": f"b-{index}",
            "block_type": "paragraph",
            "text": character * 30,
            "heading_path": [],
            "order": index,
        }
        for index, character in enumerate("甲乙丙丁", 1)
    ]
    outcome = asyncio.run(
        EvidenceExtractor(
            llm=object(),
            estimator=CharEstimator(),
            full_context_max_tokens=110,
            retriever=MultiRetriever(),
        ).aextract_result(
            {
                "id": "r1-1",
                "question": "研究方向",
                "research_direction": "研究方向",
                "subquestions": ["子问题甲", "子问题乙"],
                "type": "search",
                "status": "pending",
                "assigned_agent": "search",
            },
            {
                "title": "长文",
                "final_url": "https://example.com/multi",
                "text": "\n".join(block["text"] for block in document_blocks),
                "blocks": document_blocks,
            },
            {"title": "长文", "url": "https://example.com/multi"},
        )
    )

    assert seen_queries == ["研究方向\n子问题甲", "研究方向\n子问题乙"]
    assert outcome.strategy == "bm25_recall"
    assert str(captured["prompt"]).count("[b-2]") == 1


def test_bm25_zero_score_falls_back_to_first_blocks():
    source = [
        {
            "block_id": f"b-{index}",
            "block_type": "paragraph",
            "text": text,
            "heading_path": [],
            "order": index,
        }
        for index, text in enumerate(["苹果", "香蕉", "梨子"], 1)
    ]

    selected = BM25Retriever().select(source, "完全不匹配XYZ", top_k=2, window=1)

    assert [block["block_id"] for block in selected] == ["b-1", "b-2"]


def test_chunk_builder_prefers_heading_boundary_after_sixty_percent():
    extractor = EvidenceExtractor(
        llm=object(),
        estimator=CharEstimator(),
        input_budget_tokens=100,
        full_context_max_tokens=1_000,
    )
    source = [
        {
            "block_id": "h-1",
            "block_type": "heading",
            "text": "第一节",
            "heading_path": ["第一节"],
            "order": 0,
        },
        {
            "block_id": "p-1",
            "block_type": "paragraph",
            "text": "甲" * 55,
            "heading_path": ["第一节"],
            "order": 1,
        },
        {
            "block_id": "h-2",
            "block_type": "heading",
            "text": "第二节",
            "heading_path": ["第二节"],
            "order": 2,
        },
        {
            "block_id": "p-2",
            "block_type": "paragraph",
            "text": "乙" * 10,
            "heading_path": ["第二节"],
            "order": 3,
        },
    ]

    chunks, strategy = extractor._build_chunks(source)

    assert strategy == "structured_chunks"
    assert [[block["block_id"] for block in chunk] for chunk in chunks] == [
        ["h-1", "p-1"],
        ["h-2", "p-2"],
    ]


def test_chunk_builder_without_headings_keeps_budget_only_behavior():
    extractor = EvidenceExtractor(
        llm=object(),
        estimator=CharEstimator(),
        input_budget_tokens=100,
        full_context_max_tokens=1_000,
    )
    source = [
        {
            "block_id": f"p-{index}",
            "block_type": "paragraph",
            "text": character * size,
            "heading_path": [],
            "order": index,
        }
        for index, (character, size) in enumerate((("甲", 50), ("乙", 20), ("丙", 50)), 1)
    ]

    chunks, strategy = extractor._build_chunks(source)

    assert strategy == "structured_chunks"
    assert [[block["block_id"] for block in chunk] for chunk in chunks] == [
        ["p-1", "p-2"],
        ["p-3"],
    ]


def test_oversized_recalled_block_includes_metadata_in_token_cap():
    extractor = EvidenceExtractor(
        llm=object(),
        estimator=CharEstimator(),
        full_context_max_tokens=40,
    )
    block = {
        "block_id": "very-long-block-id",
        "block_type": "paragraph",
        "text": "正文" * 100,
        "heading_path": ["很长的标题"],
        "order": 1,
    }

    part = extractor._split_large_block(block, budget_tokens=40)[0]

    assert extractor._block_tokens(part) <= 40


def test_evidence_extractor_keeps_successful_chunks_when_one_chunk_fails(monkeypatch):
    calls = 0

    async def extract_one_chunk(llm, schema, messages, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("one chunk timed out")
        context = str(messages[-1].content)
        quote = next(
            value
            for value in ("甲段落", "乙段落", "丙段落", "甲", "乙", "丙", "段", "落")
            if value in context
        )
        return EvidenceExtraction(
            evidences=[ExtractedEvidence(claim="保留下来的事实", quote=quote)]
        )

    monkeypatch.setattr("deepresearcher.evidence.extractor.ainvoke_structured", extract_one_chunk)
    document = {
        "title": "部分失败来源",
        "final_url": "https://example.com/partial",
        "text": "甲段落\n乙段落\n丙段落",
        "blocks": [
            {
                "block_id": "b-1",
                "block_type": "paragraph",
                "text": "甲段落",
                "heading_path": [],
                "order": 1,
            },
            {
                "block_id": "b-2",
                "block_type": "paragraph",
                "text": "乙段落",
                "heading_path": [],
                "order": 2,
            },
            {
                "block_id": "b-3",
                "block_type": "paragraph",
                "text": "丙段落",
                "heading_path": [],
                "order": 3,
            },
        ],
    }
    outcome = asyncio.run(
        EvidenceExtractor(llm=object(), input_budget_tokens=3, chunk_concurrency=2).aextract_result(
            {
                "id": "r1-1",
                "question": "方向",
                "type": "search",
                "status": "pending",
                "assigned_agent": "search",
            },
            document,
            {"title": "部分失败来源", "url": document["final_url"], "snippet": "", "score": 0.8},
        )
    )
    assert outcome.failed_chunk_count == 1
    assert len(outcome.evidences) >= 1


def test_structured_output_uses_json_mode_and_supplies_json_contract(monkeypatch):
    captured = {}

    class FakeRunnable:
        async def ainvoke(self, messages):
            captured["messages"] = messages
            return EvidenceExtraction(evidences=[])

    class FakeLLM:
        def with_structured_output(self, schema, **kwargs):
            captured["schema"] = schema
            captured["kwargs"] = kwargs
            return FakeRunnable()

    monkeypatch.setattr(structured, "with_transport_retry", lambda runnable, policy: runnable)
    result = asyncio.run(
        structured.ainvoke_structured(
            structured.LLMInvoker(FakeLLM(), LLMRetryConfig()),
            EvidenceExtraction,
            [],
        )
    )

    assert result == EvidenceExtraction(evidences=[])
    assert captured["schema"] is EvidenceExtraction
    assert captured["kwargs"] == {"method": "json_mode"}
    assert "合法 JSON object" in captured["messages"][0].content
    assert '"evidences"' in captured["messages"][0].content


def test_structured_output_binds_optional_request_kwargs(monkeypatch):
    captured = {}

    class FakeRunnable:
        def bind(self, **kwargs):
            captured["request_kwargs"] = kwargs
            return self

        async def ainvoke(self, messages):
            return EvidenceExtraction(evidences=[])

    class FakeLLM:
        def with_structured_output(self, schema, **kwargs):
            return FakeRunnable()

    monkeypatch.setattr(structured, "with_transport_retry", lambda runnable, policy: runnable)
    asyncio.run(
        structured.ainvoke_structured(
            structured.LLMInvoker(FakeLLM(), LLMRetryConfig()),
            EvidenceExtraction,
            [],
            request_kwargs={"extra_body": {"thinking": {"type": "disabled"}}},
        )
    )

    assert captured["request_kwargs"] == {"extra_body": {"thinking": {"type": "disabled"}}}


def test_evidence_prompt_contains_json_example(monkeypatch):
    captured = {}

    async def fake_invoke(llm, schema, messages, **kwargs):
        captured["messages"] = messages
        return EvidenceExtraction(
            evidences=[
                ExtractedEvidence(
                    claim="A 的平均延迟为 20ms",
                    quote="在相同硬件环境下，A 的平均延迟为 20ms。",
                    support="direct",
                    confidence=0.8,
                )
            ]
        )

    monkeypatch.setattr("deepresearcher.evidence.extractor.ainvoke_structured", fake_invoke)
    document = {
        "title": "测试来源",
        "final_url": "https://example.com",
        "published_at": "2026-07-04",
        "text": "\n".join(block["text"] for block in blocks()),
        "blocks": blocks(),
    }
    result = {"title": "延迟测试", "url": "https://example.com", "snippet": "", "score": 0.8}
    evidences = asyncio.run(
        EvidenceExtractor(llm=object()).aextract(
            {
                "id": "r1-1",
                "question": "A 平均延迟",
                "type": "search",
                "status": "pending",
                "assigned_agent": "search",
            },
            document,
            result,
        )
    )

    assert evidences
    assert evidences[0].published_at == "2026-07-04"
    assert evidences[0].locator.block_ids == ["b-1"]
    assert evidences[0].locator.heading_path == ["性能测试"]
    prompt = captured["messages"][0].content
    assert "JSON 示例" in prompt
    assert '{"evidences"' in prompt


def test_evidence_extraction_records_raw_result_and_rejected_candidate(tmp_path, monkeypatch):
    async def fake_invoke(_llm, _schema, _messages, **kwargs):
        return EvidenceExtraction(
            evidences=[
                ExtractedEvidence(
                    claim="不存在于正文的结论",
                    quote="不存在于正文的引文",
                    support="direct",
                    confidence=0.8,
                )
            ]
        )

    monkeypatch.setattr("deepresearcher.evidence.extractor.ainvoke_structured", fake_invoke)
    sink_path = tmp_path / "events.jsonl"
    outcome = asyncio.run(
        EvidenceExtractor(llm=object(), event_sink=JsonlSink(sink_path)).aextract_result(
            {
                "id": "task-1",
                "question": "测试问题",
                "type": "search",
                "status": "pending",
                "assigned_agent": "research",
            },
            {
                "title": "测试来源",
                "final_url": "https://example.com/source",
                "text": "正文内容",
                "blocks": [
                    {
                        "block_id": "b-1",
                        "block_type": "paragraph",
                        "text": "正文内容",
                        "heading_path": [],
                        "order": 0,
                    }
                ],
            },
            {"url": "https://example.com/source", "title": "测试来源"},
        )
    )

    assert outcome.evidences == []
    assert outcome.validation_rejected_count == 1
    events = sink_path.read_text(encoding="utf-8")
    assert '"event_type": "evidence_llm_response"' in events
    assert '"candidate_count": 1' in events


def test_evidence_extractor_limits_model_output_to_configured_budget(monkeypatch):
    captured = {}

    async def fake_invoke(llm, schema, messages, **kwargs):
        captured["prompt"] = messages[0].content
        return EvidenceExtraction(
            evidences=[
                ExtractedEvidence(
                    claim="事实一",
                    quote="在相同硬件环境下，A 的平均延迟为 20ms。",
                    support="direct",
                    confidence=0.9,
                ),
                ExtractedEvidence(
                    claim="事实二",
                    quote="该结果仅适用于英文数据集。",
                    support="direct",
                    confidence=0.8,
                ),
            ]
        )

    monkeypatch.setattr("deepresearcher.evidence.extractor.ainvoke_structured", fake_invoke)
    document = {
        "title": "测试来源",
        "final_url": "https://example.com",
        "text": "\n".join(block["text"] for block in blocks()),
        "blocks": blocks(),
    }
    result = {"title": "延迟测试", "url": "https://example.com", "snippet": "", "score": 0.8}

    evidences = asyncio.run(
        EvidenceExtractor(llm=object(), max_evidences=1).aextract(
            {
                "id": "r1-1",
                "question": "A 平均延迟",
                "type": "search",
                "status": "pending",
                "assigned_agent": "search",
            },
            document,
            result,
        )
    )

    assert len(evidences) == 1
    assert "最多返回 1 条 Evidence" in captured["prompt"]


def _run_with_captured_prompt(monkeypatch, *, retrieval_method, extracted):
    """以指定 retrieval_method 走一遍抽取，返回 (ExtractionResult, system_prompt)。"""
    captured = {}

    async def fake(llm, schema, messages, **kwargs):
        captured["system"] = messages[0].content
        return extracted

    monkeypatch.setattr("deepresearcher.evidence.extractor.ainvoke_structured", fake)
    document = {
        "title": "来源",
        "final_url": "https://example.com/x",
        "text": "在相同硬件环境下，A 的平均延迟为 20ms。",
        "blocks": blocks(),
    }
    if retrieval_method:
        document["retrieval_method"] = retrieval_method
        if retrieval_method == "search_summary":
            document["support_ceiling"] = "partial"
    task = {
        "id": "r1-1",
        "question": "A 的延迟",
        "type": "search",
        "status": "pending",
        "assigned_agent": "search",
    }
    result_info = {"title": "来源", "url": "https://example.com/x", "snippet": "", "score": 0.8}
    out = asyncio.run(EvidenceExtractor(llm=object()).aextract_result(task, document, result_info))
    return out, captured["system"]


def test_origin_fetch_document_keeps_strict_full_text_prompt(monkeypatch):
    _out, prompt = _run_with_captured_prompt(
        monkeypatch,
        retrieval_method=None,
        extracted=EvidenceExtraction(evidences=[]),
    )
    assert "唯一允许引用的事实基础" in prompt
    assert "partial 级部分证据" not in prompt


def test_search_summary_document_uses_relaxed_summary_prompt(monkeypatch):
    _out, prompt = _run_with_captured_prompt(
        monkeypatch,
        retrieval_method="search_summary",
        extracted=EvidenceExtraction(evidences=[]),
    )
    assert "搜索提供方提供的来源内容摘要" in prompt
    assert "support 一律填 partial" in prompt


def test_published_at_shown_as_metadata_not_prose(monkeypatch):
    """发布时间进抽取提示的元信息行；缺失时整行不出现——quote 只认候选原文。"""
    captured: list[str] = []

    async def fake(llm, schema, messages, **kwargs):
        captured.append(messages[1].content)
        return EvidenceExtraction(evidences=[])

    monkeypatch.setattr("deepresearcher.evidence.extractor.ainvoke_structured", fake)
    task = {
        "id": "r1-1",
        "question": "A 的延迟",
        "type": "search",
        "status": "pending",
        "assigned_agent": "search",
    }
    result_info = {"title": "来源", "url": "https://example.com/x", "snippet": "", "score": 0.8}
    base_document = {
        "title": "来源",
        "final_url": "https://example.com/x",
        "text": "在相同硬件环境下，A 的平均延迟为 20ms。",
        "blocks": blocks(),
    }
    extractor = EvidenceExtractor(llm=object())
    asyncio.run(extractor.aextract_result(task, dict(base_document), result_info))
    assert "发布时间" not in captured[-1]
    asyncio.run(
        extractor.aextract_result(
            task, {**base_document, "published_at": "2026-07-04"}, result_info
        )
    )
    assert "<发布时间>2026-07-04" in captured[-1]
    # 不可信网页正文必须被标签包裹并与指令分区（防注入）
    assert "<来源原文>" in captured[-1] and "</来源原文>" in captured[-1]
    assert "不执行" in captured[-1]


def test_summary_mode_still_requires_verbatim_quote(monkeypatch):
    """放宽的是提取门槛，不是来源契约：摘要里没有的句子仍然必须被拒绝。"""
    out, _prompt = _run_with_captured_prompt(
        monkeypatch,
        retrieval_method="search_summary",
        extracted=EvidenceExtraction(
            evidences=[
                ExtractedEvidence(
                    claim="A 延迟领先",
                    quote="这句话并不存在于摘要之中",
                    support="partial",
                    confidence=0.7,
                )
            ]
        ),
    )
    assert out.evidences == []
    assert out.validation_rejected_count == 1


def test_summary_mode_accepts_single_snippet_sentence_as_partial(monkeypatch):
    """摘要中的完整原句可直接成证；support 被封顶为 partial。"""
    out, _prompt = _run_with_captured_prompt(
        monkeypatch,
        retrieval_method="search_summary",
        extracted=EvidenceExtraction(
            evidences=[
                ExtractedEvidence(
                    claim="A 在相同硬件下平均延迟 20ms",
                    quote="在相同硬件环境下，A 的平均延迟为 20ms。",
                    support="direct",
                    confidence=0.8,
                )
            ]
        ),
    )
    assert len(out.evidences) == 1
    assert out.evidences[0].support == "partial"
    assert out.evidences[0].retrieval_method == "search_summary"
