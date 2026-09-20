"""各 Agent 的 Evidence 可见性契约：新增字段必须在这里显式决策。"""

from deepresearcher.agents.researcher.state import evidence_observation_card
from deepresearcher.agents.supervisor.state import evidence_card as supervisor_evidence_card
from deepresearcher.agents.writer.state import evidence_detail_card, evidence_index_card
from deepresearcher.evidence.models import Evidence
from deepresearcher.nodes.reviewer import reviewer_evidence_card
from deepresearcher.reporting.validation import validate_and_bind
from deepresearcher.schemas import Citation, SourceProfile


def _evidence() -> Evidence:
    return Evidence(
        evidence_id="ev-1",
        subtask_id="task-1",
        research_direction="研究方向",
        claim="已核验事实",
        quote="一段足够长的可验证原文。",
        source_url="https://Docs.Example.com/report?id=1",
        source_title="示例报告",
        published_at="2026-09-01",
        source_profile=SourceProfile(
            source_type="organization",
            authority_tier="primary",
            publication_status="official",
            primary_source=True,
        ),
        retrieval_method="origin_fetch",
        support="partial",
        confidence=0.82,
    )


def test_researcher_evidence_projection_contract() -> None:
    card = evidence_observation_card(_evidence(), quote_chars=8)

    assert set(card) == {
        "evidence_id",
        "claim",
        "quote",
        "support",
        "confidence",
        "source_title",
        "source_domain",
        "source_profile",
        "published_at",
    }
    assert card["quote"] == "一段足够长的可验"
    assert card["source_domain"] == "docs.example.com"
    assert {"source_url", "retrieval_method", "locator", "audit_chunk", "subtask_id"}.isdisjoint(
        card
    )


def test_supervisor_evidence_projection_contract() -> None:
    card = supervisor_evidence_card(_evidence())

    assert set(card) == {
        "evidence_id",
        "claim",
        "support",
        "confidence",
        "source_title",
        "source_domain",
        "source_profile",
        "research_direction",
        "published_at",
    }
    assert card["source_domain"] == "docs.example.com"
    assert {"quote", "source_url", "retrieval_method", "locator", "audit_chunk"}.isdisjoint(card)


def test_writer_evidence_projection_contracts() -> None:
    evidence = _evidence()
    index = evidence_index_card(evidence)
    detail = evidence_detail_card(evidence)

    assert set(index) == {
        "evidence_id",
        "claim",
        "support",
        "source_title",
        "source_domain",
        "source_type",
        "published_at",
    }
    assert {"quote", "source_url", "confidence", "retrieval_method", "locator"}.isdisjoint(index)
    assert set(detail) == {
        "evidence_id",
        "claim",
        "quote",
        "source_url",
        "source_title",
        "support",
        "source_type",
        "authority_tier",
        "published_at",
    }
    assert {"confidence", "retrieval_method", "locator", "audit_chunk", "subtask_id"}.isdisjoint(
        detail
    )


def test_reviewer_evidence_projection_contract() -> None:
    citation = Citation(
        id="ev-1",
        url="https://research.example.org/paper",
        title="研究论文",
        quote="原文",
        claim="事实",
        support="partial",
        published_at="2026-08-01",
        source_profile=SourceProfile(source_type="academic", authority_tier="primary"),
    )
    card = reviewer_evidence_card(citation)

    assert set(card) == {
        "id",
        "fact",
        "quote",
        "support",
        "source_title",
        "source_domain",
        "source_type",
        "authority_tier",
        "published_at",
    }
    assert card["source_domain"] == "research.example.org"
    assert {"url", "source_profile"}.isdisjoint(card)


def test_writer_binding_preserves_support_and_compact_source_metadata() -> None:
    _, _, citations = validate_and_bind("已核验事实。[[cite:ev-1]]", {"ev-1": _evidence()})

    assert citations[0].support == "partial"
    assert citations[0].published_at == "2026-09-01"
    assert citations[0].source_profile.authority_tier == "primary"
