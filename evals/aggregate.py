"""聚合与报表：criterion 级结果 → 分维得分、case 判定、pass@k/pass^3、一致率。

分数呈现纪律（已定决策，不合并单一总分）：
  DeepResearchBench Quality   xx.x     ← 外轨四维加权（官方权重，官方 criterion）
  Evidence Groundedness       xx.x     ← 确定性门通过率（引用完整性等）
  Behavior Pass Rate          xx.x     ← 内轨行为断言
  Engineering pass@3 / pass^3          ← 内轨整体稳定性表述
  Unknown Rate / 人机一致率 / Cohen κ  ← judge 可信度证明
"""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from evals.schemas import CriterionResult, EvalCase

VERDICT_VALUE = {"yes": 1.0, "no": 0.0}  # unknown 不进分子分母


def criterion_weight(case: EvalCase, criterion_id: str) -> float:
    for c in case.criteria:
        if c.id == criterion_id:
            return c.weight * (case.dimension_weights.get(c.dimension, 1.0))
    return 0.0  # gate:* 等附挂项不占题分


@dataclass
class CaseScore:
    case_id: str
    attempt: int
    gates_green: bool
    gate_failures: list[str] = field(default_factory=list)
    behavior_pass: bool = True
    behavior_failures: list[str] = field(default_factory=list)
    quality_score: float | None = None  # 外轨加权分（judge 维度）
    judge_failures: list[str] = field(default_factory=list)  # 内轨 judge 断言的 no 项
    unknown_count: int = 0
    judge_count: int = 0
    passed: bool = False


def score_case(
    case: EvalCase, attempt: int, results: Iterable[CriterionResult]
) -> CaseScore:
    rows = [r for r in results if r.case_id == case.case_id and r.attempt == attempt]
    gates = [r for r in rows if r.criterion_id.startswith("gate:")]
    deterministic = [
        r for r in rows if r.source == "deterministic" and not r.criterion_id.startswith("gate:")
    ]
    judged = [r for r in rows if r.source == "judge"]

    score = CaseScore(
        case_id=case.case_id,
        attempt=attempt,
        gates_green=all(r.verdict == "yes" for r in gates),
        gate_failures=[r.criterion_id for r in gates if r.verdict != "yes"],
    )
    score.behavior_pass = all(r.verdict == "yes" for r in deterministic)
    score.behavior_failures = [r.criterion_id for r in deterministic if r.verdict != "yes"]
    # 内轨 judge 断言是布尔（保守交付题的"未编造数字"等）：非 yes 即失败；
    # 外轨 judge 是加权质量分，unknown 只出分母、不算失败。
    if case.track == "behavior":
        score.judge_failures = [r.criterion_id for r in judged if r.verdict != "yes"]
    else:
        score.judge_failures = [r.criterion_id for r in judged if r.verdict == "no"]
    score.unknown_count = sum(1 for r in judged if r.verdict == "unknown")
    score.judge_count = len(judged)

    if judged:
        weighted: defaultdict[str, list[float]] = defaultdict(list)
        dim_totals: defaultdict[str, float] = defaultdict(float)
        dim_hits: defaultdict[str, float] = defaultdict(float)
        for r in judged:
            weight = criterion_weight(case, r.criterion_id)
            if weight <= 0 or r.verdict == "unknown":
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
    else:  # 行为题：门 + 全部 deterministic 断言 + 全部 judge 断言，一票不过
        score.passed = (
            score.gates_green and score.behavior_pass and not score.judge_failures
        )
    return score


EXTERNAL_PASS_THRESHOLD = 75.0


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
                if any(s.quality_score is not None for s in runs)
                else None
            ),
            "unknown_rate": round(
                sum(s.unknown_count for s in runs) / max(1, sum(s.judge_count for s in runs)), 3
            ),
        }
    return out


# ---- 人工标注同步表 ----

ANNOTATION_COLUMNS = [
    "case_id",
    "attempt",
    "criterion_id",
    "dimension",
    "criterion_text",
    "judge_verdict",
    "judge_reason",
    "human_verdict",  # 人工填 yes/no/unknown（留空=未标）
    "agree",  # import 时回算
]


def export_annotation_csv(
    results: Iterable[CriterionResult], cases: dict[str, EvalCase]
) -> str:
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
    """回导人工列并回算 agree；human 为 unknown 的行不计一致（rubric 歧义信号另算）。"""
    rows = list(csv.DictReader(io.StringIO(content)))
    for row in rows:
        human = (row.get("human_verdict") or "").strip().lower()
        judge = (row.get("judge_verdict") or "").strip().lower()
        if human not in {"yes", "no", "unknown"}:
            row["agree"] = ""  # 未标注
        elif human == judge:
            row["agree"] = "1"  # 含双方都判 unknown 的"一致地认为歧义"
        elif "unknown" in (human, judge):
            row["agree"] = ""  # 一方 unknown：二值域外，不记分歧也不记一致
        else:
            row["agree"] = "0"
    return rows


def agreement_report(rows: list[dict[str, str]]) -> dict[str, Any]:
    """人机一致率 + Cohen's kappa（yes/no 二值域上算，unknown 单独计歧义率）。"""
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
