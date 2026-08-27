from langchain_core.messages import AIMessage

from deepsearch_agent.agents.researcher import ResearchAgent
from deepsearch_agent.evidence.models import Evidence
from deepsearch_agent.state import StateInvariantError, merge_evidences


def _evidence(evidence_id: str, claim: str) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        subtask_id="task-0001",
        research_direction="测试方向",
        claim=claim,
        quote=claim,
        source_url="https://example.com/source",
    )


def test_merge_evidences_keeps_same_sequence_from_different_sources():
    merged = merge_evidences(
        [],
        [
            _evidence("task-0001-src-a1b2c3d4e5-ev-1", "事实一"),
            _evidence("task-0001-src-f6a7b8c9d0-ev-1", "事实二"),
        ],
    )

    assert [item.claim for item in merged] == ["事实一", "事实二"]


def test_merge_evidences_rejects_conflicting_duplicate_id():
    try:
        merge_evidences([], [_evidence("same-id", "事实一"), _evidence("same-id", "事实二")])
    except StateInvariantError as exc:
        assert "same-id" in str(exc)
    else:
        raise AssertionError("相同 Evidence ID 的不同内容必须被拒绝")


def test_direction_complete_decision_is_bounded_before_schema_validation():
    response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "ResearchDirectionComplete",
                "id": "complete-1",
                "args": {
                    "reason": "已有足够材料",
                    "answered_points": ["p1", "p2", "p3", "p4", "p5"],
                    "remaining_gaps": ["g1", "g2", "g3", "g4", "g5"],
                },
            }
        ],
    )

    decision = ResearchAgent._parse_direction_decision(response)

    assert decision.answered_points == ["p1", "p2", "p3", "p4"]
    assert decision.remaining_gaps == ["g1", "g2", "g3", "g4"]
