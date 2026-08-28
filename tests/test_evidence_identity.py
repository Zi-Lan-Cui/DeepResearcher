from deepsearch_agent.agents.researcher.tools import build_researcher_tools
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


def test_researcher_registers_completion_as_a_standard_tool():
    tools = {item.name: item for item in build_researcher_tools()}

    assert "ResearchDirectionComplete" in tools
    assert "ReadSources" in tools
