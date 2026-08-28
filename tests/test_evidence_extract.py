import asyncio

import pytest

from deepsearch_agent.config import LLMRetryConfig
from deepsearch_agent.evidence import EvidenceExtractor
from deepsearch_agent.evidence.models import EvidenceExtraction, ExtractedEvidence
from deepsearch_agent.evidence.retrieval import select_blocks
from deepsearch_agent.llm import LLMConfigurationError, structured
from deepsearch_agent.observability.events import JsonlSink


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


def test_lexical_retrieval_keeps_adjacent_qualification():
    selected = select_blocks("A 平均延迟", blocks(), top_k=1, window=1)
    assert [block["block_id"] for block in selected] == ["b-0", "b-1", "b-2"]


def test_evidence_extractor_requires_llm_at_construction():
    with pytest.raises(LLMConfigurationError):
        EvidenceExtractor(None)


def test_llm_empty_evidence_does_not_fall_back_to_source_title(monkeypatch):
    async def empty_extraction(llm, schema, messages, **kwargs):
        return EvidenceExtraction(evidences=[])

    monkeypatch.setattr("deepsearch_agent.evidence.extractor.ainvoke_structured", empty_extraction)
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


def test_long_document_is_extracted_from_structured_chunks_not_bm25_selection(monkeypatch):
    seen_contexts = []

    async def extract_each_chunk(llm, schema, messages, **kwargs):
        context = messages[-1].content
        seen_contexts.append(context)
        if "证据甲" in context:
            return EvidenceExtraction(
                evidences=[
                    ExtractedEvidence(
                        claim="事实甲", quote="证据甲", support="direct", confidence=0.9
                    )
                ]
            )
        if "证据乙" in context:
            return EvidenceExtraction(
                evidences=[
                    ExtractedEvidence(
                        claim="事实乙", quote="证据乙", support="direct", confidence=0.9
                    )
                ]
            )
        return EvidenceExtraction(evidences=[])

    monkeypatch.setattr(
        "deepsearch_agent.evidence.extractor.ainvoke_structured", extract_each_chunk
    )
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
    extractor = EvidenceExtractor(llm=object(), input_budget_tokens=5, chunk_concurrency=2)

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

    assert outcome.strategy == "structured_chunks"
    assert outcome.chunk_count == 3
    assert {item.claim for item in outcome.evidences} == {"事实甲", "事实乙"}
    assert len(seen_contexts) == 3


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

    monkeypatch.setattr(
        "deepsearch_agent.evidence.extractor.ainvoke_structured", extract_one_chunk
    )
    document = {
        "title": "部分失败来源",
        "final_url": "https://example.com/partial",
        "text": "甲段落\n乙段落\n丙段落",
        "blocks": [
            {"block_id": "b-1", "block_type": "paragraph", "text": "甲段落", "heading_path": [], "order": 1},
            {"block_id": "b-2", "block_type": "paragraph", "text": "乙段落", "heading_path": [], "order": 2},
            {"block_id": "b-3", "block_type": "paragraph", "text": "丙段落", "heading_path": [], "order": 3},
        ],
    }
    outcome = asyncio.run(
        EvidenceExtractor(llm=object(), input_budget_tokens=3, chunk_concurrency=2).aextract_result(
            {"id": "r1-1", "question": "方向", "type": "search", "status": "pending", "assigned_agent": "search"},
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

    monkeypatch.setattr("deepsearch_agent.evidence.extractor.ainvoke_structured", fake_invoke)
    document = {
        "title": "测试来源",
        "final_url": "https://example.com",
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

    monkeypatch.setattr("deepsearch_agent.evidence.extractor.ainvoke_structured", fake_invoke)
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

    monkeypatch.setattr("deepsearch_agent.evidence.extractor.ainvoke_structured", fake_invoke)
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

    monkeypatch.setattr("deepsearch_agent.evidence.extractor.ainvoke_structured", fake)
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
