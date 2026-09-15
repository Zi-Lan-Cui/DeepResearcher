import pytest
from pydantic import ValidationError

from deepresearcher.config import AgentConfig
from deepresearcher.schemas.limits import (
    EVIDENCE_REFERENCES_HARD_LIMIT,
    REPORT_MARKDOWN_HARD_LIMIT_CHARS,
)
from deepresearcher.schemas.reporting import MarkdownReportDraft, ResearchAspect, ResearchSynthesis


def _synthesis(aspect_count: int) -> ResearchSynthesis:
    aspects = [
        ResearchAspect(
            aspect_id=f"aspect-{index}",
            topic=f"主题 {index}",
            role="支撑结论",
            status="covered",
            summary="已有证据支撑。",
            evidence_ids=[f"evidence-{index}"],
        )
        for index in range(aspect_count)
    ]
    return ResearchSynthesis(
        revision=1,
        based_on_working_set_revision=1,
        answer_goal="回答研究问题",
        overall_summary="当前材料可以形成报告。",
        aspects=aspects,
        decision_rationale="按证据支撑程度组织。",
    )


def test_reporting_schemas_do_not_enforce_old_business_limits():
    synthesis = _synthesis(10)
    draft = MarkdownReportDraft(
        markdown="x" * 30_000,
        selected_evidence_ids=[f"evidence-{index}" for index in range(40)],
    )

    assert len(synthesis.aspects) == 10
    assert len(synthesis.selected_evidence_ids) == 10
    assert len(draft.markdown) == 30_000
    assert len(draft.selected_evidence_ids) == 40


def test_agent_config_accepts_normal_values_above_old_schema_caps():
    config = AgentConfig(
        supervisor_max_active_evidences=60,
        research_agent_max_evidences_per_direction=32,
        research_agent_max_evidence_candidates_per_direction=64,
        writer_max_markdown_chars=50_000,
    )

    assert config.supervisor_max_active_evidences == 60
    assert config.writer_max_markdown_chars == 50_000


def test_schema_and_config_still_reject_anomalous_payloads():
    with pytest.raises(ValidationError):
        MarkdownReportDraft(markdown="x" * (REPORT_MARKDOWN_HARD_LIMIT_CHARS + 1))
    with pytest.raises(ValueError, match="SUPERVISOR_MAX_ACTIVE_EVIDENCES"):
        AgentConfig(supervisor_max_active_evidences=EVIDENCE_REFERENCES_HARD_LIMIT + 1)
    with pytest.raises(ValueError, match="WRITER_MAX_MARKDOWN_CHARS"):
        AgentConfig(writer_max_markdown_chars=REPORT_MARKDOWN_HARD_LIMIT_CHARS + 1)
