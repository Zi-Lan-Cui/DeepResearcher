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
from evals.judge import SYSTEM_PROMPT, judge_case, parse_verdict  # noqa: E402
from evals.schemas import (  # noqa: E402
    Criterion,
    CriterionResult,
    EvalCase,
    cases_to_jsonl,
    load_cases,
    load_results,
    merge_results,
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


def test_core30_profile_is_small_fixed_and_representative() -> None:
    profile = json.loads((REPO_ROOT / "evals" / "profiles" / "core30.json").read_text("utf-8"))
    external = profile["external_ids"]
    behavior = profile["behavior_ids"]

    assert len(external) == 20
    assert len(behavior) == 10
    assert len(set(external + behavior)) == 30
    assert {case_id.split("-")[1][0] for case_id in behavior} == {"b", "c", "d", "f"}


def test_cases_jsonl_roundtrip(tmp_path):
    case = _case(require_clarify=True, clarify_answer="a")
    path = tmp_path / "cases.jsonl"
    path.write_text(cases_to_jsonl([case]), "utf-8")
    assert load_cases(path) == [case]


def test_partial_result_merge_replaces_only_evaluated_rounds(tmp_path):
    path = tmp_path / "judge.jsonl"
    original = [
        CriterionResult("a", 1, "old-a", "insight", "yes", "judge"),
        CriterionResult("b", 1, "keep-b", "insight", "yes", "judge"),
    ]
    path.write_text("".join(json.dumps(dataclasses.asdict(r)) + "\n" for r in original), "utf-8")
    replacement = CriterionResult("a", 1, "new-a", "insight", "no", "judge")

    merge_results(path, [replacement], {("a", 1)})

    rows = load_results(path)
    assert [(row.case_id, row.criterion_id) for row in rows] == [
        ("a", "new-a"),
        ("b", "keep-b"),
    ]


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


def test_drb_root_prefers_explicit_path(fake_drb, monkeypatch):
    monkeypatch.setenv("DRB_ROOT", "/path/that/must/not/be/used")
    assert drb.drb_root(fake_drb) == fake_drb


def test_drb_root_reads_environment(fake_drb, monkeypatch):
    monkeypatch.setenv("DRB_ROOT", str(fake_drb))
    assert drb.drb_root() == fake_drb


def test_drb_root_requires_configuration(monkeypatch):
    monkeypatch.delenv("DRB_ROOT", raising=False)
    with pytest.raises(FileNotFoundError, match="--drb-root.*DRB_ROOT"):
        drb.drb_root()


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
    evidences = [
        {"evidence_id": "e1", "quote": "q1", "audit_chunk": "before q1 after"},
        {"evidence_id": "e2", "quote": "q2", "audit_chunk": "before q2 after"},
    ]
    base = dict(
        case_id="x-1",
        attempt=1,
        run_id="r",
        detail=detail,
        events=events,
        evidences=evidences,
    )
    base.update(overrides)
    return Artifact(**base)


def _gate(results, name):
    return next(r for r in results if r.criterion_id == f"gate:{name}")


def test_artifact_normalizes_span_failures_from_top_level_records():
    artifact = _artifact(
        events=[
            {
                "event_type": "span_failed",
                "trace_id": "trace-1",
                "span_id": "span-1",
                "parent_span_id": "span-0",
                "name": "evidence_extract",
                "kind": "llm",
                "node_id": "researcher",
                "duration_ms": 12.5,
                "error": "TimeoutError",
                "payload": None,
            }
        ]
    )

    assert artifact.span_failures() == [
        {
            "trace_id": "trace-1",
            "span_id": "span-1",
            "parent_span_id": "span-0",
            "name": "evidence_extract",
            "kind": "llm",
            "node_id": "researcher",
            "duration_ms": 12.5,
            "error": "TimeoutError",
        }
    ]


def test_process_metrics_carries_cache_and_token_fields():
    """效率块不能静默漏字段：cached/saved tokens 必须在 process_metrics 里（曾因缺
    字段令 report 显示 0，尽管 DB/detail 有 941k cached）。"""
    from evals.deterministic import process_metrics

    art = _artifact(
        detail={
            "status": "completed",
            "report_markdown": "",
            "citations": [],
            "input_tokens": 1000,
            "output_tokens": 100,
            "cached_input_tokens": 800,
            "saved_tokens": 5000,
            "saved_external_request_count": 7,
            "external_request_count": 59,
            "llm_call_count": 93,
            "cache_hit_count": 12,
            "estimated_cost_usd": 0.0,
            "elapsed_ms": 1234,
        }
    )
    metrics = process_metrics(art)
    assert metrics["cached_input_tokens"] == 800
    assert metrics["saved_tokens"] == 5000
    assert metrics["saved_external_request_count"] == 7
    agg = aggregate.summarize_metrics([metrics, metrics])
    assert agg["cached_input_tokens"] == 1600 and agg["llm_call_count"] == 186
    assert agg["cost_note"]  # token-only 口径显式标注


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


def test_evidence_audit_gate_recomputes_quote_presence() -> None:
    good = _gate(score_artifact(_artifact(), _case(criteria=())), "evidence_audit_integrity")
    assert good.verdict == "yes"

    broken = _artifact(
        evidences=[{"evidence_id": "e1", "quote": "missing", "audit_chunk": "other text"}]
    )
    result = _gate(score_artifact(broken, _case(criteria=())), "evidence_audit_integrity")
    assert result.verdict == "no"


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
    assert "报告没写到”一律判 no" in SYSTEM_PROMPT


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
    assert score.track == "external"
    # unknown 不是有效交付：保守按0分计，同时另报 unknown_rate。
    unknown_b = [
        dataclasses.replace(r, verdict="unknown") if r.criterion_id == "b" else r for r in results
    ]
    score = aggregate.score_case(case, 1, unknown_b)
    assert score.quality_score == 50.0 and not score.passed


def test_behavior_judge_requires_an_explicit_yes():
    """unknown 进健康度报表供人工复核，但不能把产品 case 判通过。"""
    case = _case(criteria=(Criterion(id="j", dimension="grounding", text="未编造", weight=1),))
    gates = [
        CriterionResult("x-1", 1, "gate:citation_integrity", "grounding", "yes", "deterministic"),
        CriterionResult("x-1", 1, "gate:done_exactly_once", "grounding", "yes", "deterministic"),
        CriterionResult("x-1", 1, "gate:seq_continuous", "grounding", "yes", "deterministic"),
    ]
    undecidable = [*gates, CriterionResult("x-1", 1, "j", "grounding", "unknown", "judge")]
    score = aggregate.score_case(case, 1, undecidable)
    assert not score.passed and score.unknown_count == 1 and score.judge_failures == ["j"]
    violated = [*gates, CriterionResult("x-1", 1, "j", "grounding", "no", "judge")]
    score = aggregate.score_case(case, 1, violated)
    assert not score.passed and score.judge_failures == ["j"]


def test_track_summary_never_mixes_behavior_judge_score_into_report_quality():
    external = aggregate.CaseScore("external", 1, "external", True, quality_score=20.0)
    behavior = aggregate.CaseScore("behavior", 1, "behavior", True, quality_score=100.0)
    metrics = {
        "external": {"pass_any": False, "pass_all": False},
        "behavior": {"pass_any": True, "pass_all": True},
    }

    summary = aggregate.track_summary([external, behavior], metrics)

    assert summary["report_quality"]["mean"] == 20.0
    assert summary["system_reliability"]["pass_all"] == 1
    per_case = aggregate.k_metrics([external, behavior])
    assert per_case["behavior"]["mean_quality"] is None


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
