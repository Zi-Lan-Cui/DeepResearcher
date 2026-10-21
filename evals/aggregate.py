"""聚合与报表：criterion 级结果 → RACE 双序得分、case 判定、pass@k/pass^k、效率汇总。

分数呈现纪律：分块各自呈现,不合并为单一总分。
  RACE Quality      正序与换序 target/(target+reference) 的均值,0.5=与专家参考打平
  Gates             确定性三门(引用完整性/run_done/seq),不产质量分
  Engineering pass@k / pass^k          ← 通过=三门均通过且双序均值>0.5
  Efficiency        token/耗时/缓存,只记录,不参与判定
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from evals.schemas import CriterionResult, EvalCase, RaceCriterionScore

# 与专家参考打平记不过:通过要求双序均值严格高于 0.5。
RACE_PASS_RATIO = 0.5


@dataclass
class CaseScore:
    case_id: str
    attempt: int
    gates_green: bool
    gate_failures: list[str] = field(default_factory=list)
    race_forward: RaceCaseScore | None = None
    race_swapped: RaceCaseScore | None = None
    race_mean: float | None = None
    passed: bool = False


def score_case(
    case: EvalCase,
    attempt: int,
    deterministic_rows: Iterable[CriterionResult],
    race_forward_rows: Iterable[RaceCriterionScore],
    race_swapped_rows: Iterable[RaceCriterionScore],
) -> CaseScore:
    """门 + RACE 双序合成单轮判定：通过 = 三门均通过 ∧ (正序+换序) overall 均值 > 0.5。

    任一序未判或无有效分则 race_mean 为 None,该轮不判过;两个方向的缺分
    criterion 数由 race 聚合对象各自记录。
    """
    rows = [r for r in deterministic_rows if r.case_id == case.case_id and r.attempt == attempt]
    gates = [r for r in rows if r.criterion_id.startswith("gate:")]
    gates_green = all(r.verdict == "yes" for r in gates)
    forward = score_case_race(case, attempt, race_forward_rows)
    swapped = score_case_race(case, attempt, race_swapped_rows)
    forward_overall = forward.overall
    swapped_overall = swapped.overall
    race_mean = (
        round((forward_overall + swapped_overall) / 2, 4)
        if forward_overall is not None and swapped_overall is not None
        else None
    )
    return CaseScore(
        case_id=case.case_id,
        attempt=attempt,
        gates_green=gates_green,
        gate_failures=[r.criterion_id for r in gates if r.verdict != "yes"],
        race_forward=forward,
        race_swapped=swapped,
        race_mean=race_mean,
        passed=gates_green and race_mean is not None and race_mean > RACE_PASS_RATIO,
    )


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
    """按题聚合 pass@k（至少一次）与 pass^k（次次成功），附 RACE 双序均值。"""
    by_case: defaultdict[str, list[CaseScore]] = defaultdict(list)
    for score in scores:
        by_case[score.case_id].append(score)
    out: dict[str, dict[str, Any]] = {}
    for case_id, runs in by_case.items():
        runs.sort(key=lambda s: s.attempt)
        passed = [s.passed for s in runs]
        means = [s.race_mean for s in runs if s.race_mean is not None]
        out[case_id] = {
            "attempts": len(runs),
            "pass_any": any(passed),
            "pass_all": all(passed),
            "mean_race": round(sum(means) / len(means), 4) if means else None,
            "missing_criterion_count": sum(
                (s.race_forward.missing_count if s.race_forward else 0)
                + (s.race_swapped.missing_count if s.race_swapped else 0)
                for s in runs
            ),
        }
    return out


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
