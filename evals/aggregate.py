"""聚合与报表：criterion 级结果 → 分维得分、case 判定、pass@k/pass^3、一致率。

分数呈现纪律：五个分块各自呈现,不合并为单一总分。
  DeepResearchBench Quality   xx.x     ← external 轨四维加权,用官方权重与官方 criterion
  Evidence Groundedness       xx.x     ← 确定性门通过率,即引用完整性等
  Behavior Pass Rate          xx.x     ← behavior 轨行为断言
  Engineering pass@3 / pass^3          ← behavior 轨整体稳定性表述
  Unknown Rate / 人机一致率 / Cohen κ  ← judge 可信度证明
"""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from evals.schemas import CriterionResult, EvalCase, RaceCriterionScore

VERDICT_VALUE = {"yes": 1.0, "no": 0.0, "unknown": 0.0}


def criterion_weight(case: EvalCase, criterion_id: str) -> float:
    for c in case.criteria:
        if c.id == criterion_id:
            return c.weight * (case.dimension_weights.get(c.dimension, 1.0))
    return 0.0  # gate:* 门控项不参与题分加权


@dataclass
class CaseScore:
    case_id: str
    attempt: int
    track: str
    gates_green: bool
    gate_failures: list[str] = field(default_factory=list)
    behavior_pass: bool = True
    behavior_failures: list[str] = field(default_factory=list)
    quality_score: float | None = None  # external 轨 judge 维度的加权分
    judge_failures: list[str] = field(default_factory=list)  # behavior 轨 judge 断言的非 yes 项
    unknown_count: int = 0
    judge_count: int = 0
    passed: bool = False


def score_case(case: EvalCase, attempt: int, results: Iterable[CriterionResult]) -> CaseScore:
    rows = [r for r in results if r.case_id == case.case_id and r.attempt == attempt]
    gates = [r for r in rows if r.criterion_id.startswith("gate:")]
    deterministic = [
        r for r in rows if r.source == "deterministic" and not r.criterion_id.startswith("gate:")
    ]
    judged = [r for r in rows if r.source == "judge"]

    score = CaseScore(
        case_id=case.case_id,
        attempt=attempt,
        track=case.track,
        gates_green=all(r.verdict == "yes" for r in gates),
        gate_failures=[r.criterion_id for r in gates if r.verdict != "yes"],
    )
    score.behavior_pass = all(r.verdict == "yes" for r in deterministic)
    score.behavior_failures = [r.criterion_id for r in deterministic if r.verdict != "yes"]
    # unknown 表示 judge 未给出有效判定,不能算产品通过;同时计入
    # unknown_rate,用于区分系统未达标与 judge/rubric 需要校准两种情况。
    score.judge_failures = [r.criterion_id for r in judged if r.verdict != "yes"]
    score.unknown_count = sum(1 for r in judged if r.verdict == "unknown")
    score.judge_count = len(judged)

    if judged:
        dim_totals: defaultdict[str, float] = defaultdict(float)
        dim_hits: defaultdict[str, float] = defaultdict(float)
        for r in judged:
            weight = criterion_weight(case, r.criterion_id)
            if weight <= 0:
                continue
            dim_totals[r.dimension] += weight
            dim_hits[r.dimension] += weight * VERDICT_VALUE[r.verdict]
        if dim_totals:
            per_dim = {d: dim_hits[d] / dim_totals[d] for d in dim_totals}
            total_weight = sum(case.dimension_weights.get(d, 1.0) for d in per_dim) or 1.0
            score.quality_score = (
                sum(per_dim[d] * case.dimension_weights.get(d, 1.0) for d in per_dim) / total_weight
            ) * 100
    if case.track == "external":
        score.passed = (
            score.gates_green
            and score.quality_score is not None
            and score.quality_score >= EXTERNAL_PASS_THRESHOLD
        )
    else:  # behavior 轨：门控、deterministic 断言、judge 断言任一不过即失败
        score.passed = score.gates_green and score.behavior_pass and not score.judge_failures
    return score


EXTERNAL_PASS_THRESHOLD = 75.0


@dataclass
class RaceCaseScore:
    """RACE 官方口径的单轮得分：target/(target+reference) 归一，0.5 为与参考文章打平。"""

    case_id: str
    attempt: int
    target_total: float
    reference_total: float
    overall: float | None
    dimension_ratios: dict[str, float] = field(default_factory=dict)
    scored_count: int = 0
    missing_count: int = 0


def score_case_race(
    case: EvalCase, attempt: int, rows: Iterable[RaceCriterionScore]
) -> RaceCaseScore:
    """复刻官方聚合：维度内 criterion 加权均分 → 维度权重求和 → 与参考文章做比值。

    参数 rows 为全部 RACE 分数记录。双 None 的 criterion（judge 失败）不计入加权
    分母；整维无有效分时该维均分按 0 计，与官方 score_calculator 的置零行为一致。
    """
    own = [
        row
        for row in rows
        if row.case_id == case.case_id
        and row.attempt == attempt
        and row.dimension in set(criterion.dimension for criterion in case.criteria)
    ]
    criterion_weights = {criterion.id: criterion.weight for criterion in case.criteria}
    dimension_weights = case.dimension_weights or {}
    per_dimension_target: dict[str, float] = {}
    per_dimension_reference: dict[str, float] = {}
    scored_count = 0
    missing_count = 0
    for dimension in dict.fromkeys(row.dimension for row in own):
        numerator_target = 0.0
        numerator_reference = 0.0
        denominator_weight = 0.0
        for row in (item for item in own if item.dimension == dimension):
            weight = criterion_weights.get(row.criterion_id, 1.0)
            if row.target_score is None or row.reference_score is None:
                missing_count += 1
                continue
            scored_count += 1
            numerator_target += row.target_score * weight
            numerator_reference += row.reference_score * weight
            denominator_weight += weight
        per_dimension_target[dimension] = (
            numerator_target / denominator_weight if denominator_weight else 0.0
        )
        per_dimension_reference[dimension] = (
            numerator_reference / denominator_weight if denominator_weight else 0.0
        )

    target_total = sum(
        per_dimension_target.get(dimension, 0.0) * dimension_weights.get(dimension, 0.0)
        for dimension in dimension_weights
    )
    reference_total = sum(
        per_dimension_reference.get(dimension, 0.0) * dimension_weights.get(dimension, 0.0)
        for dimension in dimension_weights
    )
    dimension_ratios = {
        dimension: (
            target_value / (target_value + reference_value)
            if target_value + reference_value > 0
            else None
        )
        for dimension in dimension_weights
        for target_value in (per_dimension_target.get(dimension, 0.0),)
        for reference_value in (per_dimension_reference.get(dimension, 0.0),)
    }
    overall = (
        target_total / (target_total + reference_total)
        if target_total + reference_total > 0
        else None
    )
    return RaceCaseScore(
        case_id=case.case_id,
        attempt=attempt,
        target_total=round(target_total, 4),
        reference_total=round(reference_total, 4),
        overall=round(overall, 4) if overall is not None else None,
        dimension_ratios={
            key: round(value, 4) for key, value in dimension_ratios.items() if value is not None
        },
        scored_count=scored_count,
        missing_count=missing_count,
    )


def k_metrics(scores: list[CaseScore]) -> dict[str, dict[str, Any]]:
    """按题聚合 pass@k（至少一次）与 pass^k（次次成功）。"""
    by_case: defaultdict[str, list[CaseScore]] = defaultdict(list)
    for score in scores:
        by_case[score.case_id].append(score)
    out: dict[str, dict[str, Any]] = {}
    for case_id, runs in by_case.items():
        runs.sort(key=lambda s: s.attempt)
        passed = [s.passed for s in runs]
        out[case_id] = {
            "attempts": len(runs),
            "pass_any": any(passed),
            "pass_all": all(passed),
            "mean_quality": (
                round(
                    sum(s.quality_score for s in runs if s.quality_score is not None)
                    / max(1, sum(1 for s in runs if s.quality_score is not None)),
                    1,
                )
                if runs[0].track == "external" and any(s.quality_score is not None for s in runs)
                else None
            ),
            "unknown_rate": round(
                sum(s.unknown_count for s in runs) / max(1, sum(s.judge_count for s in runs)), 3
            ),
        }
    return out


def track_summary(
    scores: list[CaseScore], metrics: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """分轨汇总：外轨只算报告质量，内轨只算系统可靠性。"""
    external = [score for score in scores if score.track == "external"]
    behavior = [score for score in scores if score.track == "behavior"]
    external_ids = {score.case_id for score in external}
    behavior_ids = {score.case_id for score in behavior}
    quality = [score.quality_score for score in external if score.quality_score is not None]
    return {
        "report_quality": {
            "cases": len(external_ids),
            "rounds": len(external),
            "mean": round(sum(quality) / len(quality), 2) if quality else None,
        },
        "system_reliability": {
            "cases": len(behavior_ids),
            "rounds": len(behavior),
            "pass_any": sum(1 for case_id in behavior_ids if metrics[case_id]["pass_any"]),
            "pass_all": sum(1 for case_id in behavior_ids if metrics[case_id]["pass_all"]),
        },
    }


ANNOTATION_COLUMNS = [
    "case_id",
    "attempt",
    "criterion_id",
    "dimension",
    "criterion_text",
    "judge_verdict",
    "judge_reason",
    "human_verdict",  # 人工填 yes/no/unknown,留空视为未标注
    "agree",  # 重新导入时在 import_annotation_csv 里回算
]


def export_annotation_csv(results: Iterable[CriterionResult], cases: dict[str, EvalCase]) -> str:
    """judge 结果 → 待人工标注 CSV。人工只需填 human_verdict 一列。"""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=ANNOTATION_COLUMNS)
    writer.writeheader()
    for r in sorted(results, key=lambda x: (x.case_id, x.attempt, x.criterion_id)):
        if r.source != "judge":
            continue
        text = ""
        case = cases.get(r.case_id)
        if case is not None:
            text = next((c.text for c in case.criteria if c.id == r.criterion_id), "")
        writer.writerow(
            {
                "case_id": r.case_id,
                "attempt": r.attempt,
                "criterion_id": r.criterion_id,
                "dimension": r.dimension,
                "criterion_text": text,
                "judge_verdict": r.verdict,
                "judge_reason": r.reason,
                "human_verdict": "",
                "agree": "",
            }
        )
    return buffer.getvalue()


def import_annotation_csv(content: str) -> list[dict[str, str]]:
    """读入人工列并回算 agree:双方判定相同算一致,含都判 unknown;单方
    unknown 的行不计一致也不计分歧,rubric 歧义信号由 agreement_report 另算。"""
    rows = list(csv.DictReader(io.StringIO(content)))
    for row in rows:
        human = (row.get("human_verdict") or "").strip().lower()
        judge = (row.get("judge_verdict") or "").strip().lower()
        if human not in {"yes", "no", "unknown"}:
            row["agree"] = ""  # 未标注
        elif human == judge:
            row["agree"] = "1"  # 双方一致判 unknown 也算一致
        elif "unknown" in (human, judge):
            row["agree"] = ""  # 单方 unknown,不参与一致与否的判定
        else:
            row["agree"] = "0"
    return rows


def agreement_report(rows: list[dict[str, str]]) -> dict[str, Any]:
    """人机一致率 + Cohen's kappa。kappa 只在 yes/no 二值域上计算,
    unknown 不计入,单独以双方歧义率呈现。"""
    judge: list[str] = []
    human: list[str] = []
    for row in rows:
        h = (row.get("human_verdict") or "").strip().lower()
        j = (row.get("judge_verdict") or "").strip().lower()
        if h not in {"yes", "no", "unknown"} or j not in {"yes", "no", "unknown"}:
            continue
        judge.append(j)
        human.append(h)
    if not judge:
        return {"labeled": 0}
    both_binary = [(j, h) for j, h in zip(judge, human) if j != "unknown" and h != "unknown"]
    kappa = _cohen_kappa([j for j, _ in both_binary], [h for _, h in both_binary])
    return {
        "labeled": len(judge),
        "both_binary": len(both_binary),
        "raw_agreement": (
            round(sum(1 for j, h in both_binary if j == h) / len(both_binary), 4)
            if both_binary
            else None
        ),
        "cohens_kappa": None if kappa is None else round(kappa, 4),
        "judge_unknown_rate": round(judge.count("unknown") / len(judge), 4),
        "human_unknown_rate": round(human.count("unknown") / len(human), 4),
    }


def _cohen_kappa(a: list[str], b: list[str]) -> float | None:
    if not a or len(a) != len(b):
        return None
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum(
        (a.count(label) / n) * (b.count(label) / n)  # pyright: ignore[reportUnknownVariableType]
        for label in labels
    )
    if pe >= 1.0:
        return None  # 单类别退化，kappa 无定义
    return (po - pe) / (1 - pe)


def summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """把各 artifact 的 process_metrics 汇成效率块。

    成本口径:token 已从网关实采,usage_estimated=false;estimated_cost_usd
    在未配 LLM 单价时恒 0,这是当前只记录 token 的刻意口径,不是数据缺失。
    """
    if not rows:
        return {"runs": 0}
    total_in = sum(int(r.get("input_tokens") or 0) for r in rows)
    total_out = sum(int(r.get("output_tokens") or 0) for r in rows)
    return {
        "runs": len(rows),
        "input_tokens": total_in,
        "output_tokens": total_out,
        "total_tokens": total_in + total_out,
        "cached_input_tokens": sum(int(r.get("cached_input_tokens") or 0) for r in rows),
        "llm_call_count": sum(int(r.get("llm_call_count") or 0) for r in rows),
        "external_request_count": sum(int(r.get("external_request_count") or 0) for r in rows),
        "cache_hit_count": sum(int(r.get("cache_hit_count") or 0) for r in rows),
        "mean_total_tokens_per_run": round((total_in + total_out) / len(rows)),
        "mean_elapsed_ms": round(sum(int(r.get("elapsed_ms") or 0) for r in rows) / len(rows)),
        "estimated_cost_usd": round(sum(float(r.get("estimated_cost_usd") or 0) for r in rows), 6),
        "cost_note": "tokens 实采；$ 需配 LLM_*_USD_PER_MILLION 单价才计",
    }
