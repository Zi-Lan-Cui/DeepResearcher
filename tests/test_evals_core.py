"""evals harness 的核心回归：契约校验、题集名单、确定性门、RACE judge、聚合。

evals/ 不在 ruff/pyright 门检范围内，它的正确性由本文件的断言覆盖。
"""

import dataclasses
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # evals 非安装包，按路径导入

from evals import aggregate, drb  # noqa: E402
from evals.deterministic import Artifact, score_artifact  # noqa: E402
from evals.judge import (  # noqa: E402
    judge_case_race,
    parse_race_scores,
)
from evals.schemas import (  # noqa: E402
    Criterion,
    CriterionResult,
    EvalCase,
    RaceCriterionScore,
    load_race_results,
    load_results,
    merge_results,
    validate_case,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

GATE_IDS = (
    "gate:citation_integrity",
    "gate:done_exactly_once",
    "gate:seq_continuous",
)


def _case(**overrides):
    base = dict(
        case_id="x-1",
        prompt="p",
        criteria=(Criterion(id="a", dimension="insight", text="甲", weight=1.0),),
        dimension_weights={"insight": 1.0},
    )
    base.update(overrides)
    return EvalCase(**base)


def _gates(attempt: int = 1, verdicts: dict[str, str] | None = None) -> list[CriterionResult]:
    failures = verdicts or {}
    return [
        CriterionResult(
            "x-1", attempt, gate_id, "grounding", failures.get(gate_id, "yes"), "deterministic"
        )
        for gate_id in GATE_IDS
    ]


def test_dev_profile_is_external_only() -> None:
    profile = json.loads((REPO_ROOT / "evals" / "profiles" / "dev.json").read_text("utf-8"))
    assert "behavior_ids" not in profile
    external = profile["external_ids"]
    assert len(external) == 20
    assert all(case_id.startswith("drb-zh-") for case_id in external)


def test_dev_test30_partition_zh50() -> None:
    profiles = REPO_ROOT / "evals" / "profiles"
    dev = json.loads((profiles / "dev.json").read_text("utf-8"))["external_ids"]
    test30 = json.loads((profiles / "test30.json").read_text("utf-8"))["external_ids"]
    zh50 = {f"drb-zh-{index:03d}" for index in range(1, 51)}
    assert len(dev) == 20 and len(test30) == 30
    # 开发集与实测集零交集，合起来恰好铺满 DRB 中文 50 题。
    assert set(dev) & set(test30) == set()
    assert set(dev) | set(test30) == zh50


def test_partial_result_merge_replaces_only_evaluated_rounds(tmp_path):
    path = tmp_path / "deterministic.jsonl"
    original = [
        CriterionResult("a", 1, "old-a", "grounding", "yes", "deterministic"),
        CriterionResult("b", 1, "keep-b", "grounding", "yes", "deterministic"),
    ]
    path.write_text("".join(json.dumps(dataclasses.asdict(r)) + "\n" for r in original), "utf-8")
    replacement = CriterionResult("a", 1, "new-a", "grounding", "no", "deterministic")

    merge_results(path, [replacement], {("a", 1)})

    rows = load_results(path)
    assert [(row.case_id, row.criterion_id) for row in rows] == [
        ("a", "new-a"),
        ("b", "keep-b"),
    ]


def test_validate_catches_broken_cases():
    assert validate_case(_case(prompt="  "))
    assert validate_case(_case(criteria=()))
    bad_dim = _case(criteria=(Criterion(id="c", dimension="magic", text="t"),))
    assert validate_case(bad_dim)
    assert validate_case(
        _case(criteria=(Criterion(id="c", dimension="insight", text="t", weight=0),))
    )


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
    assert case.case_id == "drb-zh-001"
    assert case.dimension_weights == {"comprehensiveness": 0.5, "insight": 0.5}
    assert [c.id for c in case.criteria] == ["comprehensiveness-1", "insight-1"]
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
    base = dict(
        case_id="x-1",
        attempt=1,
        run_id="r",
        detail=detail,
        events=events,
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
    """效率字段不能静默漏:cached 与 saved tokens 必须出现在 process_metrics 里。
    曾因漏字段使 report 显示 0,而底层数据其实存在。"""
    from evals.deterministic import process_metrics

    art = _artifact(
        detail={
            "status": "completed",
            "report_markdown": "",
            "citations": [],
            "input_tokens": 1000,
            "output_tokens": 100,
            "cached_input_tokens": 800,
            "saved_external_request_count": 7,
            "external_request_count": 59,
            "llm_call_count": 93,
            "cache_hit_count": 12,
            "estimated_cost_cny": 0.0,
            "elapsed_ms": 1234,
        }
    )
    metrics = process_metrics(art)
    assert metrics["cached_input_tokens"] == 800
    assert metrics["saved_external_request_count"] == 7
    agg = aggregate.summarize_metrics([metrics, metrics])
    assert agg["cached_input_tokens"] == 1600 and agg["llm_call_count"] == 186
    assert agg["cost_note"]  # token-only 统计方式显式标注


def test_citation_gate_pass_and_dangling():
    results = score_artifact(_artifact())
    assert _gate(results, "citation_integrity").verdict == "yes"
    bad = _artifact(
        detail={
            "status": "completed",
            "report_markdown": "引用[来源9]。",
            "citations": [{"id": "e1", "url": "u", "quote": "q"}],
        }
    )
    verdict = _gate(score_artifact(bad), "citation_integrity")
    assert verdict.verdict == "no"
    pending = _artifact(detail={"status": "running", "report_markdown": "", "citations": []})
    assert _gate(score_artifact(pending), "citation_integrity").verdict == "unknown"


def test_seq_and_done_gates():
    results = score_artifact(_artifact())
    assert _gate(results, "done_exactly_once").verdict == "yes"
    assert _gate(results, "seq_continuous").verdict == "yes"
    holed = _artifact(
        events=_artifact().events[:2] + [{"event_type": "run_done", "payload": {}, "seq": 9}]
    )
    assert _gate(score_artifact(holed), "seq_continuous").verdict == "no"


def test_score_artifact_emits_exactly_three_gates():
    results = score_artifact(_artifact())
    assert {r.criterion_id for r in results} == set(GATE_IDS)


@pytest.mark.asyncio
async def test_judge_case_race_scores_both_articles_and_degrades_on_garbage():
    class RaceInvoker:
        def __init__(self):
            self.calls = []

        async def complete(self, system: str, user: str, max_tokens: int) -> str:
            self.calls.append(user)
            if len(self.calls) == 1:
                return '{"analysis": "甲覆盖更广", "article_1_score": 7, "article_2_score": 9}'
            return "输出坏了"

    case = _case(
        criteria=(
            Criterion(id="a", dimension="insight", text="甲", weight=0.5),
            Criterion(id="b", dimension="insight", text="乙", weight=0.5),
        ),
    )
    invoker = RaceInvoker()
    rows = await judge_case_race(case, report="R", attempt=1, invoker=invoker, reference="REF")
    assert (rows[0].target_score, rows[0].reference_score) == (7.0, 9.0)
    assert (rows[1].target_score, rows[1].reference_score) == (None, None)
    assert "解析失败" in rows[1].analysis or "judge 调用或解析失败" in rows[1].analysis
    assert "REF" in invoker.calls[0] and "文章二" in invoker.calls[0]


@pytest.mark.asyncio
async def test_judge_case_race_swap_maps_scores_back_to_target_and_reference():
    class FixedInvoker:
        def __init__(self):
            self.calls = []

        async def complete(self, system: str, user: str, max_tokens: int) -> str:
            self.calls.append(user)
            # 文章一恒得 9 分、文章二恒得 3 分，不辨内容——专门验证映射方向。
            return '{"analysis": "x", "article_1_score": 9, "article_2_score": 3}'

    case = _case(criteria=(Criterion(id="a", dimension="insight", text="甲", weight=1.0),))
    invoker = FixedInvoker()
    normal = await judge_case_race(case, report="R", attempt=1, invoker=invoker, reference="REF")
    assert (normal[0].target_score, normal[0].reference_score) == (9.0, 3.0)
    swapped = await judge_case_race(
        case, report="R", attempt=1, invoker=invoker, reference="REF", swap_positions=True
    )
    assert (swapped[0].target_score, swapped[0].reference_score) == (3.0, 9.0)
    for prompt in invoker.calls:
        assert "专家参考文章" not in prompt and "待评估报告" not in prompt
        assert "**文章一**" in prompt and "**文章二**" in prompt
    assert "REF" in invoker.calls[-1].split("**文章一**")[1].split("**文章二**")[0]


def test_parse_race_scores_strict_lenient_and_clamp():
    assert parse_race_scores(
        '{"analysis": "对比", "article_1_score": 8, "article_2_score": 5}'
    ) == (
        8.0,
        5.0,
        "对比",
    )
    fenced = '```json\n{"analysis": "x", "article_1_score": 12.5, "article_2_score": -3}\n```'
    assert parse_race_scores(fenced)[:2] == (10.0, 0.0)
    truncated = '先写了一段分析…… "article_1_score": 6.5, "article_2_score": 4.0'
    assert parse_race_scores(truncated)[:2] == (6.5, 4.0)
    with pytest.raises(ValueError):
        parse_race_scores("两篇都不错，无法给分")


def test_score_case_race_handles_round_without_rows() -> None:
    case = _case(
        dimension_weights={"comprehensiveness": 1.0},
        criteria=(Criterion(id="c1", dimension="comprehensiveness", text="丙", weight=1.0),),
    )
    scored = aggregate.score_case_race(case, 7, [])
    assert scored.overall is None
    assert scored.dimension_ratios == {}


def test_score_case_race_official_weighting_and_ratio():
    case = _case(
        dimension_weights={"comprehensiveness": 0.6, "readability": 0.4},
        criteria=(
            Criterion(id="c1", dimension="comprehensiveness", text="丙", weight=2.0),
            Criterion(id="c2", dimension="comprehensiveness", text="丁", weight=1.0),
            Criterion(id="r1", dimension="readability", text="戊", weight=1.0),
        ),
    )
    rows = [
        RaceCriterionScore("x-1", 1, "c1", "comprehensiveness", 8.0, 4.0),
        RaceCriterionScore("x-1", 1, "c2", "comprehensiveness", 10.0, 10.0),
        RaceCriterionScore("x-1", 1, "r1", "readability", None, None, "judge 调用或解析失败"),
    ]
    scored = aggregate.score_case_race(case, 1, rows)
    # comprehensiveness 加权均分： target (8*2+10*1)/3≈8.667, reference (4*2+10*1)/3=6.0
    # readability 整维无有效分→均分 0(与官方置零一致，比值项被剔除)
    assert scored.target_total == round(26 / 3 * 0.6, 4)
    assert scored.reference_total == round(6.0 * 0.6, 4)
    assert scored.overall == round(
        scored.target_total / (scored.target_total + scored.reference_total), 4
    )
    assert set(scored.dimension_ratios) == {"comprehensiveness"}
    assert scored.missing_count == 1


def test_score_case_requires_both_orders_and_passing_gates():
    forward = [RaceCriterionScore("x-1", 1, "a", "insight", 8.0, 4.0)]
    swapped = [RaceCriterionScore("x-1", 1, "a", "insight", 7.0, 5.0)]
    score = aggregate.score_case(_case(), 1, _gates(), forward, swapped)
    assert score.gates_green and score.passed
    assert score.race_forward.overall == round(8 / 12, 4)
    assert score.race_swapped.overall == round(7 / 12, 4)
    assert score.race_mean == round((8 / 12 + 7 / 12) / 2, 4)

    # 单序缺失时双序均值不成立，该轮不判过。
    single = aggregate.score_case(_case(), 1, _gates(), forward, [])
    assert single.race_mean is None and not single.passed

    # 均值过线但某个门未通过，仍不过。
    failed_gate = _gates(verdicts={"gate:citation_integrity": "no"})
    assert not aggregate.score_case(_case(), 1, failed_gate, forward, swapped).passed

    # 双序均值 ≤ 0.5 即未超专家参考，不过。
    low_forward = [RaceCriterionScore("x-1", 1, "a", "insight", 4.0, 8.0)]
    low_swapped = [RaceCriterionScore("x-1", 1, "a", "insight", 3.0, 9.0)]
    assert not aggregate.score_case(_case(), 1, _gates(), low_forward, low_swapped).passed


def test_k_metrics_pass_any_vs_all():
    rows = [*_gates(1), *_gates(2, {"gate:citation_integrity": "no"})]
    forward = [
        RaceCriterionScore("x-1", 1, "a", "insight", 8.0, 4.0),
        RaceCriterionScore("x-1", 2, "a", "insight", 8.0, 4.0),
    ]
    swapped = [
        RaceCriterionScore("x-1", 1, "a", "insight", 7.0, 5.0),
        RaceCriterionScore("x-1", 2, "a", "insight", 7.0, 5.0),
    ]
    scores = [aggregate.score_case(_case(), attempt, rows, forward, swapped) for attempt in (1, 2)]
    metrics = aggregate.k_metrics(scores)
    assert metrics["x-1"]["pass_any"] is True
    assert metrics["x-1"]["pass_all"] is False
    assert metrics["x-1"]["mean_race"] == round((8 / 12 + 7 / 12) / 2, 4)
    assert metrics["x-1"]["missing_criterion_count"] == 0


# judge 身份约束：引擎模型给自己产出打分有偏好，自评需显式开启，异构模型不受限。


def _fake_llm_settings(monkeypatch, model: str = "mimo-v2.5"):
    from types import SimpleNamespace

    import deepresearcher.config as config_module
    from deepresearcher.config import LLMConfig

    settings = SimpleNamespace(llm=LLMConfig(api_key="k", base_url="https://gw/v1", model=model))
    monkeypatch.setattr(config_module, "get_settings", lambda: settings)


def test_judge_falling_back_to_engine_model_requires_explicit_optin(monkeypatch):
    from evals.judge import OpenAICompatJudge

    _fake_llm_settings(monkeypatch)
    monkeypatch.delenv("EVAL_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("EVAL_SELF_JUDGE", raising=False)
    with pytest.raises(RuntimeError, match="EVAL_JUDGE_MODEL"):
        OpenAICompatJudge()

    monkeypatch.setenv("EVAL_SELF_JUDGE", "1")
    judge = OpenAICompatJudge()
    assert judge.identity == {"model": "mimo-v2.5", "self_judging": True}


def test_judge_heterogeneous_model_needs_no_optin(monkeypatch):
    from evals.judge import OpenAICompatJudge

    _fake_llm_settings(monkeypatch)
    monkeypatch.delenv("EVAL_SELF_JUDGE", raising=False)
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "another-family-70b")
    judge = OpenAICompatJudge()
    assert judge.identity == {"model": "another-family-70b", "self_judging": False}


@pytest.mark.asyncio
async def test_cmd_judge_writes_both_orders(fake_drb, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from evals import cli as evals_cli

    results_dir = tmp_path / "results"
    (results_dir / "drb-zh-001").mkdir(parents=True)
    (results_dir / "drb-zh-001" / "round1.json").write_text(
        json.dumps(
            [
                {
                    "attempt": 1,
                    "run_id": "r1",
                    "detail": {"status": "completed", "report_markdown": "正文"},
                    "events": [],
                }
            ]
        ),
        "utf-8",
    )

    class DualInvoker:
        def __init__(self):
            self.calls: list[str] = []
            self.self_judging = False

        @property
        def identity(self):
            return {"model": "fake-judge", "self_judging": False}

        async def complete(self, system: str, user: str, max_tokens: int) -> str:
            self.calls.append(user)
            return '{"analysis": "x", "article_1_score": 9, "article_2_score": 3}'

    fake = DualInvoker()
    monkeypatch.setattr(evals_cli, "OpenAICompatJudge", lambda: fake)
    monkeypatch.setattr(evals_cli, "RESULTS_DIR", results_dir)
    args = SimpleNamespace(case=None, drb_root=str(fake_drb), attempt=None)
    await evals_cli.cmd_judge(args)

    assert len(fake.calls) == 4  # fake_drb 每题 2 条 criterion × 双序
    forward = load_race_results(results_dir / "judge_race.jsonl")
    swapped = load_race_results(results_dir / "judge_race_swapped.jsonl")
    assert len(forward) == 2 and len(swapped) == 2
    assert all(row.target_score == 9.0 for row in forward)
    assert all(row.target_score == 3.0 for row in swapped)


def test_round_persist_resume_roundtrip_is_atomic(tmp_path, monkeypatch):
    import asyncio

    from evals.runner import RunHarness

    monkeypatch.setenv("EVAL_PASSWORD", "pw")
    monkeypatch.setenv("SERVICE_DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    harness = RunHarness(out_dir=tmp_path)
    case = _case(case_id="rt-1")
    artifact = Artifact(case_id="rt-1", attempt=1, run_id="run-1", detail={"status": "completed"})
    harness._persist(case, 1, [artifact])
    loaded = harness._load_round(case, 1)
    assert loaded is not None
    assert loaded[0].run_id == "run-1" and loaded[0].case_id == "rt-1"
    assert not list((tmp_path / "rt-1").glob("*.tmp")), "原子替换后不留临时文件"
    assert harness._load_round(case, 2) is None
    (tmp_path / "rt-1" / "round1.json").write_text("{broken", encoding="utf-8")
    assert harness._load_round(case, 1) is None, "半截 JSON 按缺跑处理"
    asyncio.run(harness.close())


def test_round_load_ignores_retired_payload_keys(tmp_path, monkeypatch):
    """历史轮次产物带 run_row/control/evidences 键,--resume 读回时按白名单忽略。"""
    import asyncio

    from evals.runner import RunHarness

    monkeypatch.setenv("EVAL_PASSWORD", "pw")
    monkeypatch.setenv("SERVICE_DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    harness = RunHarness(out_dir=tmp_path)
    case = _case(case_id="old-1")
    (tmp_path / "old-1").mkdir()
    (tmp_path / "old-1" / "round1.json").write_text(
        json.dumps(
            [
                {
                    "run_id": "r",
                    "attempt": 1,
                    "detail": {"status": "completed"},
                    "run_row": {"fence_token": 2},
                    "control": {"auto_resumed": True},
                    "events": [],
                    "evidences": [{"evidence_id": "e1"}],
                    "metrics": {"status": "completed"},
                }
            ]
        ),
        "utf-8",
    )
    loaded = harness._load_round(case, 1)
    assert loaded is not None and loaded[0].run_id == "r"
    asyncio.run(harness.close())


def test_cmd_run_from_attempt_resume_and_concurrency(tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from evals import cli as evals_cli

    calls: list[tuple[str, int, bool]] = []
    inflight = 0
    peak = 0

    class FakeHarness:
        def __init__(self, *, out_dir):
            assert out_dir is not None

        async def warmup_auth(self):
            pass

        async def close(self):
            pass

        async def run_case(self, case, attempt, *, resume=False):
            nonlocal inflight, peak
            inflight += 1
            peak = max(peak, inflight)
            calls.append((case.case_id, attempt, resume))
            await asyncio.sleep(0.01)
            inflight -= 1
            if case.case_id == "a" and attempt == 2:
                raise RuntimeError("模拟额度耗尽")

    monkeypatch.setattr(evals_cli, "RunHarness", FakeHarness)
    monkeypatch.setattr(
        evals_cli,
        "collect_cases",
        lambda args: [_case(case_id="a", repeat=3), _case(case_id="b", repeat=3)],
    )
    args = SimpleNamespace(
        dry_run=False,
        attempts=0,
        from_attempt=2,
        resume=True,
        concurrency=3,
    )
    asyncio.run(evals_cli.cmd_run(args))
    # from-attempt=2：每题只跑 2、3 两个 attempt；单题异常不拖垮整批。
    assert sorted((cid, at) for cid, at, _ in calls) == [("a", 2), ("a", 3), ("b", 2), ("b", 3)]
    assert all(resume for _, _, resume in calls)
    assert peak >= 2, "concurrency=3 应出现并发窗口"

    calls.clear()
    peak = 0
    args.concurrency = 1
    args.resume = False
    asyncio.run(evals_cli.cmd_run(args))
    assert [cid for cid, _, _ in calls].count("a") == 2 and peak == 1, "默认串行语义不变"


def test_await_state_execution_budget_starts_after_queue(tmp_path, monkeypatch):
    """排队时间不计入执行超时;并发批次中排队不是 (case,attempt) 失败。"""
    import asyncio as loop_asyncio

    from evals import runner as evals_runner

    monkeypatch.setenv("EVAL_PASSWORD", "pw")
    monkeypatch.setenv("SERVICE_DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    monkeypatch.setattr(evals_runner, "POLL_SECONDS", 0.02)

    class _Detail:
        def __init__(self, status):
            self._status = status

        def raise_for_status(self):
            pass

        def json(self):
            return {"status": self._status}

    class _Client:
        def __init__(self, statuses, default_status="completed"):
            self.statuses = list(statuses)
            self.default_status = default_status

        async def get(self, *_args, **_kwargs):
            status = self.statuses.pop(0) if self.statuses else self.default_status
            return _Detail(status)

    harness = evals_runner.RunHarness(out_dir=tmp_path, timeout_seconds=0.05)

    async def scenario():
        # 8 轮 queued ≈160ms，超出 50ms 执行预算：排队时段不计入执行预算。
        client = _Client(["queued"] * 8)
        return await harness._await_state(client, {}, "run-x")

    detail = loop_asyncio.run(scenario())
    assert detail["status"] == "completed"

    async def running_forever():
        client = _Client(["running"], default_status="running")
        return await harness._await_state(client, {}, "run-y")

    with pytest.raises(TimeoutError, match="执行超时"):
        loop_asyncio.run(running_forever())
    loop_asyncio.run(harness.close())
