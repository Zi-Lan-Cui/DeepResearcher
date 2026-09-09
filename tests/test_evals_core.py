"""evals harness 的核心回归：契约校验、题集文件、确定性 scorer、judge、聚合。

evals/ 不在 ruff/pyright 门检内（刻意），但它的正确性必须由门检内的测试兜住：
这些断言失败 = 评测数字不可信 = 整个评测体系失去意义。
"""

import csv
import dataclasses
import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # evals 非安装包，按路径导入

from evals import aggregate, drb  # noqa: E402
from evals.deterministic import Artifact, score_artifact  # noqa: E402
from evals.judge import judge_case, parse_verdict  # noqa: E402
from evals.schemas import (  # noqa: E402
    Criterion,
    CriterionResult,
    EvalCase,
    cases_to_jsonl,
    load_cases,
    validate_case,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _case(**overrides):
    base = dict(
        case_id="x-1",
        track="behavior",
        prompt="p",
        criteria=(
            Criterion(
                id="gate-ref",
                dimension="grounding",
                text="t",
                scorer="deterministic:citation_integrity",
            ),
        ),
    )
    base.update(overrides)
    return EvalCase(**base)


# ---- schemas ----


def test_behavior_seed_set_loads_and_validates():
    cases = load_cases(REPO_ROOT / "evals" / "cases" / "behavior.jsonl")
    assert len(cases) == 15
    for case in cases:
        assert validate_case(case) == [], case.case_id
    groups = {case.group for case in cases}
    assert groups == {"B", "C", "D", "F"}


def test_cases_jsonl_roundtrip(tmp_path):
    case = _case(require_clarify=True, clarify_answer="a")
    path = tmp_path / "cases.jsonl"
    path.write_text(cases_to_jsonl([case]), "utf-8")
    assert load_cases(path) == [case]


def test_validate_catches_broken_cases():
    assert validate_case(_case(prompt="  "))  # 空 prompt
    assert validate_case(_case(criteria=()))  # 无判据
    assert validate_case(_case(require_clarify=True, clarify_answer=""))  # 等待无解
    bad_dim = _case(criteria=(Criterion(id="c", dimension="magic", text="t"),))
    assert validate_case(bad_dim)


# ---- drb adapter ----


@pytest.fixture
def fake_drb(tmp_path):
    root = tmp_path / "drb"
    (root / "data" / "prompt_data").mkdir(parents=True)
    (root / "data" / "criteria_data").mkdir()
    (root / "data" / "test_data" / "cleaned_data").mkdir(parents=True)
    (root / "data" / "prompt_data" / "query.jsonl").write_text(
        json.dumps(
            {"id": 1, "topic": "Finance & Business", "language": "zh", "prompt": "中产人数"},
            ensure_ascii=False,
        )
        + "\n"
        + json.dumps(
            {"id": 2, "topic": "Health", "language": "en", "prompt": "x"}, ensure_ascii=False
        )
        + "\n",
        "utf-8",
    )
    (root / "data" / "criteria_data" / "criteria.jsonl").write_text(
        json.dumps(
            {
                "id": 1,
                "prompt": "中产人数",
                "dimension_weight": {"comprehensiveness": 0.5, "insight": 0.5},
                "criterions": {
                    "comprehensiveness": [
                        {"criterion": "列出两口径", "explanation": "e", "weight": 0.6}
                    ],
                    "insight": [{"criterion": "解释差异", "explanation": "e", "weight": 0.4}],
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        "utf-8",
    )
    (root / "data" / "test_data" / "cleaned_data" / "reference.jsonl").write_text(
        json.dumps({"id": 1, "prompt": "中产人数", "article": "专家文章"}, ensure_ascii=False)
        + "\n",
        "utf-8",
    )
    return root


def test_drb_adapter_builds_official_criteria_without_copying(fake_drb):
    cases = drb.build_cases(fake_drb)
    assert len(cases) == 1  # 只收中文
    case = cases[0]
    assert case.track == "external"
    assert case.case_id == "drb-zh-001"
    assert case.dimension_weights == {"comprehensiveness": 0.5, "insight": 0.5}
    assert [c.scorer for c in case.criteria] == ["judge", "judge"]
    assert drb.load_references(fake_drb) == {1: "专家文章"}


def test_drb_split_is_seeded_stable_and_partitions(fake_drb, tmp_path):
    # 只有 1 个中文题，用 100 题语义验证：两次生成结果一致、dev+holdout 全划分
    out1 = tmp_path / "s1.json"
    out2 = tmp_path / "s2.json"
    s1 = drb.make_split(fake_drb, dev_size=1, out_path=out1)
    s2 = drb.make_split(fake_drb, dev_size=1, out_path=out2)
    assert s1 == s2
    held = drb.build_cases(fake_drb, split_file=out1, split_name="holdout")
    assert held == []  # id=1 落在 dev


# ---- deterministic scorer ----


def _artifact(**overrides) -> Artifact:
    detail = {
        "status": "completed",
        "report_markdown": "正文引用[来源1]与[来源2]。",
        "citations": [
            {"id": "e1", "url": "https://a", "title": "A", "quote": "q1"},
            {"id": "e2", "url": "https://b", "title": "B", "quote": "q2"},
        ],
    }
    events = [
        {"event_type": "run_status", "payload": {"status": "queued"}, "seq": 1},
        {"event_type": "run_status", "payload": {"status": "running"}, "seq": 2},
        {"event_type": "run_done", "payload": {"status": "completed"}, "seq": 3},
    ]
    base = dict(case_id="x-1", attempt=1, run_id="r", detail=detail, events=events)
    base.update(overrides)
    return Artifact(**base)


def _gate(results, name):
    return next(r for r in results if r.criterion_id == f"gate:{name}")


def test_citation_gate_pass_and_dangling():
    results = score_artifact(_artifact(), _case(criteria=()))
    assert _gate(results, "citation_integrity").verdict == "yes"
    bad = _artifact(
        detail={
            "status": "completed",
            "report_markdown": "引用[来源9]。",
            "citations": [{"id": "e1", "url": "u", "quote": "q"}],
        }
    )
    verdict = _gate(score_artifact(bad, _case(criteria=())), "citation_integrity")
    assert verdict.verdict == "no"
    pending = _artifact(detail={"status": "running", "report_markdown": "", "citations": []})
    assert (
        _gate(score_artifact(pending, _case(criteria=())), "citation_integrity").verdict
        == "unknown"
    )


def test_seq_and_done_gates():
    results = score_artifact(_artifact(), _case(criteria=()))
    assert _gate(results, "done_exactly_once").verdict == "yes"
    assert _gate(results, "seq_continuous").verdict == "yes"
    holed = _artifact(
        events=_artifact().events[:2] + [{"event_type": "run_done", "payload": {}, "seq": 9}]
    )
    assert _gate(score_artifact(holed, _case(criteria=())), "seq_continuous").verdict == "no"


def _clarify_events(extra_router_reruns: int = 0):
    events = [
        {"event_type": "run_status", "payload": {"status": "queued"}, "seq": 1},
        {"event_type": "run_status", "payload": {"status": "running"}, "seq": 2},
        {"event_type": "node_started", "node": "router", "seq": 3},
        {
            "event_type": "clarification_requested",
            "payload": {"question": "q", "options": ["a", "b", "c"]},
            "seq": 4,
        },
        {"event_type": "run_status", "payload": {"status": "awaiting_input"}, "seq": 5},
        {"event_type": "run_status", "payload": {"status": "queued"}, "seq": 6},
        {"event_type": "run_status", "payload": {"status": "running"}, "seq": 7},
        {"event_type": "run_done", "payload": {"status": "completed"}, "seq": 8},
    ]
    events += [
        {"event_type": "node_started", "node": "router", "seq": 9 + i}
        for i in range(extra_router_reruns)
    ]
    return events


def test_clarify_flow_checks_resume_path():
    case = _case(
        criteria=(
            Criterion(id="cf", dimension="behavior", text="t", scorer="deterministic:clarify_flow"),
        )
    )
    results = score_artifact(_artifact(events=_clarify_events()), case)
    cf = next(r for r in results if r.criterion_id == "cf")
    assert cf.verdict == "yes"
    assert _gate(results, "seq_continuous").verdict == "yes"  # 附挂门同为绿
    rerun = score_artifact(_artifact(events=_clarify_events(1)), case)
    cf = next(r for r in rerun if r.criterion_id == "cf")
    assert cf.verdict == "no" and "Router" in cf.reason  # 恢复重跑 Router → 行为违规


def test_lease_takeover_and_cache_reuse():
    case = _case(
        criteria=(
            Criterion(
                id="tk", dimension="behavior", text="t", scorer="deterministic:lease_takeover"
            ),
            Criterion(id="cr", dimension="behavior", text="t", scorer="deterministic:cache_reuse"),
        )
    )
    takeover = _artifact(
        run_row={"attempt": 2},
        detail={**_artifact().detail, "cache_hit_count": 3, "saved_external_request_count": 5},
    )
    results = score_artifact(takeover, case)
    assert next(r for r in results if r.criterion_id == "tk").verdict == "yes"
    assert next(r for r in results if r.criterion_id == "cr").verdict == "yes"
    cold = _artifact(run_row={"attempt": 1})
    results = score_artifact(cold, case)
    assert next(r for r in results if r.criterion_id == "tk").verdict == "unknown"
    assert next(r for r in results if r.criterion_id == "cr").verdict == "no"


# ---- judge ----


class ScriptedInvoker:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    async def complete(self, system: str, user: str) -> str:
        self.calls.append(user)
        return self.answers.pop(0)


@pytest.mark.asyncio
async def test_judge_one_call_per_criterion_with_unknown_exit():
    case = _case(
        track="external",
        criteria=(
            Criterion(id="a", dimension="insight", text="甲", weight=0.5),
            Criterion(id="b", dimension="insight", text="乙", weight=0.5),
        ),
        dimension_weights={"insight": 1.0},
    )
    invoker = ScriptedInvoker(["yes", "抱歉，无法判断"])
    results = await judge_case(case, report="R", attempt=1, invoker=invoker, reference="REF")
    assert [r.verdict for r in results] == ["yes", "unknown"]
    assert len(invoker.calls) == 2  # 逐条独立调用
    assert "REF" in invoker.calls[0] and "【待评报告】" in invoker.calls[0]


@pytest.mark.asyncio
async def test_judge_transport_failure_degrades_to_unknown():
    class Broken:
        async def complete(self, system, user):
            raise RuntimeError("429")

    case = _case(criteria=(Criterion(id="a", dimension="insight", text="甲"),))
    results = await judge_case(case, report="R", attempt=1, invoker=Broken())
    assert results[0].verdict == "unknown"
    assert "429" in results[0].reason


def test_parse_verdict_prefix_ordering():
    assert parse_verdict("unknown") == "unknown"
    assert parse_verdict("NO.") == "no"
    assert parse_verdict("yes, clearly") == "yes"


# ---- aggregate ----


def test_external_weighted_score_and_threshold():
    case = _case(
        track="external",
        dimension_weights={"comprehensiveness": 1.0},
        criteria=(
            Criterion(id="a", dimension="comprehensiveness", text="甲", weight=0.5),
            Criterion(id="b", dimension="comprehensiveness", text="乙", weight=0.5),
        ),
    )
    results = [
        CriterionResult("x-1", 1, "gate:citation_integrity", "grounding", "yes", "deterministic"),
        CriterionResult("x-1", 1, "gate:done_exactly_once", "grounding", "yes", "deterministic"),
        CriterionResult("x-1", 1, "gate:seq_continuous", "grounding", "yes", "deterministic"),
        CriterionResult("x-1", 1, "a", "comprehensiveness", "yes", "judge"),
        CriterionResult("x-1", 1, "b", "comprehensiveness", "no", "judge"),
    ]
    score = aggregate.score_case(case, 1, results)
    assert score.quality_score == 50.0 and not score.passed
    # unknown 出分母不算错：b 判 unknown → 仅 a 计分 → 100
    unknown_b = [
        dataclasses.replace(r, verdict="unknown") if r.criterion_id == "b" else r for r in results
    ]
    score = aggregate.score_case(case, 1, unknown_b)
    assert score.quality_score == 100.0 and score.passed


def test_behavior_judge_only_positive_no_fails_not_unknown():
    """条件化语义：unknown=判不准（非确定性/rubric 歧义），记入 unknown_rate 交人工，
    不制造假失败；只有确凿的 no 才 fail。"""
    case = _case(criteria=(Criterion(id="j", dimension="grounding", text="未编造", weight=1),))
    gates = [
        CriterionResult("x-1", 1, "gate:citation_integrity", "grounding", "yes", "deterministic"),
        CriterionResult("x-1", 1, "gate:done_exactly_once", "grounding", "yes", "deterministic"),
        CriterionResult("x-1", 1, "gate:seq_continuous", "grounding", "yes", "deterministic"),
    ]
    undecidable = [*gates, CriterionResult("x-1", 1, "j", "grounding", "unknown", "judge")]
    score = aggregate.score_case(case, 1, undecidable)
    assert score.passed and score.unknown_count == 1 and not score.judge_failures
    violated = [*gates, CriterionResult("x-1", 1, "j", "grounding", "no", "judge")]
    score = aggregate.score_case(case, 1, violated)
    assert not score.passed and score.judge_failures == ["j"]


def test_clarify_flow_is_vacuously_true_when_no_clarification():
    """未触发澄清 → 前件假 → 条件断言空真（yes），不因模型这次没问而假失败。"""
    from evals.deterministic import CHECKS

    art = _artifact()  # 状态只有 queued/running/done，无 awaiting_input
    result = CHECKS["clarify_flow"](art, Criterion(id="cf", dimension="behavior", text="t"))
    assert result.verdict == "yes" and "空真" in result.reason


def test_k_metrics_pass_any_vs_all():
    case = _case()
    rows = [
        CriterionResult("x-1", 1, "gate:citation_integrity", "grounding", "yes", "deterministic"),
        CriterionResult("x-1", 1, "gate:done_exactly_once", "grounding", "yes", "deterministic"),
        CriterionResult("x-1", 1, "gate:seq_continuous", "grounding", "yes", "deterministic"),
        CriterionResult("x-1", 2, "gate:citation_integrity", "grounding", "no", "deterministic"),
        CriterionResult("x-1", 2, "gate:done_exactly_once", "grounding", "yes", "deterministic"),
        CriterionResult("x-1", 2, "gate:seq_continuous", "grounding", "yes", "deterministic"),
    ]
    scores = [aggregate.score_case(case, attempt, rows) for attempt in (1, 2)]
    metrics = aggregate.k_metrics(scores)
    assert metrics["x-1"]["pass_any"] is True
    assert metrics["x-1"]["pass_all"] is False


def test_annotation_roundtrip_and_agreement():
    results = [
        CriterionResult("x-1", 1, "j", "grounding", "yes", "judge", "r"),
        CriterionResult("x-1", 1, "k", "grounding", "unknown", "judge", "u"),
    ]
    case = _case(criteria=(Criterion(id="j", dimension="grounding", text="乙"),))
    csv_text = aggregate.export_annotation_csv(results, {"x-1": case})
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    assert len(rows) == 2 and rows[0]["criterion_text"] == "乙"
    # 人工填：j=yes（一致），k=yes（judge unknown → 不进二值域但计 unknown 率）
    for row in rows:
        row["human_verdict"] = "yes"
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=aggregate.ANNOTATION_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    imported = aggregate.import_annotation_csv(buffer.getvalue())
    assert [r["agree"] for r in imported] == ["1", ""]
    report = aggregate.agreement_report(imported)
    assert report["labeled"] == 2 and report["both_binary"] == 1
    assert report["raw_agreement"] == 1.0
    assert report["judge_unknown_rate"] == 0.5


def test_cohen_kappa_basic():
    # 完全一致 → 1.0 之前的退化保护；随机对半 → 0
    assert aggregate._cohen_kappa(["yes", "no", "yes", "no"], ["yes", "no", "yes", "no"]) == 1.0
    assert (
        abs(aggregate._cohen_kappa(["yes", "no", "yes", "no"], ["no", "yes", "no", "yes"])) - 1.0
        < 1e-9
    )
