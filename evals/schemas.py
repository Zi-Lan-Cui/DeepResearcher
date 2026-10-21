"""评测数据契约：DeepResearch Bench 的 case 与逐条判定记录。

题集全部来自 DRB 中文题池,criterion 与维度权重取官方 rubric;判分走 RACE 口径
(被测与专家参考文章各打 0-10 分,见 judge/aggregate)。确定性门对每份产物自动
附加,判定写入 ``CriterionResult``,与 pass^k 同批记录计算,不另设第二套数据。

设计约束:
- 打分单位是 criterion,judge 每条独立调用,不给整篇报告打印象分;
- 门行不写进题集,由 ``deterministic.score_artifact`` 注入,gate 前缀见该模块。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

# 维度名沿用 DRB 官方四维;门行自带 grounding 维,不在题集 rubric 范围内。
DIMENSIONS = (
    "comprehensiveness",
    "insight",
    "instruction_following",
    "readability",
)

Verdict = Literal["yes", "no", "unknown"]


@dataclass(frozen=True)
class Criterion:
    id: str
    dimension: str
    text: str
    explanation: str = ""
    weight: float = 1.0


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    prompt: str
    criteria: tuple[Criterion, ...]
    language: str = "zh"
    topic: str = ""
    repeat: int = 1
    dimension_weights: dict[str, float] = field(default_factory=dict)  # DRB 官方维度权重


@dataclass(frozen=True)
class CriterionResult:
    case_id: str
    attempt: int
    criterion_id: str
    dimension: str
    verdict: Verdict
    source: Literal["deterministic"]
    reason: str = ""


def results_to_jsonl(results: list[CriterionResult]) -> str:
    return "".join(json.dumps(asdict(r), ensure_ascii=False) + "\n" for r in results)


def load_results(path: Path) -> list[CriterionResult]:
    return [
        CriterionResult(**json.loads(line))
        for line in path.read_text("utf-8").splitlines()
        if line.strip()
    ]


def merge_results(
    path: Path,
    replacements: list[CriterionResult],
    evaluated: set[tuple[str, int]],
) -> list[CriterionResult]:
    """原子替换本次已评估的 case/attempt，保留其他轮次。

    ``--case`` 是局部重评工具，不应把全局结果文件截成当前一题。
    已评估但本次没有产生结果的轮次也会清除旧值，避免残留过期判定。
    """
    existing = load_results(path) if path.is_file() else []
    merged = [row for row in existing if (row.case_id, row.attempt) not in evaluated]
    merged.extend(replacements)
    merged.sort(key=lambda row: (row.case_id, row.attempt, row.criterion_id, row.source))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(results_to_jsonl(merged), "utf-8")
    temporary.replace(path)
    return merged


@dataclass(frozen=True)
class RaceCriterionScore:
    """RACE 对齐口径的单条判定：同一调用内给被测与参考两篇文章各打 0-10 分。"""

    case_id: str
    attempt: int
    criterion_id: str
    dimension: str
    target_score: float | None
    reference_score: float | None
    analysis: str = ""


def race_results_to_jsonl(rows: list[RaceCriterionScore]) -> str:
    return "".join(json.dumps(asdict(r), ensure_ascii=False) + "\n" for r in rows)


def load_race_results(path: Path) -> list[RaceCriterionScore]:
    return [
        RaceCriterionScore(**json.loads(line))
        for line in path.read_text("utf-8").splitlines()
        if line.strip()
    ]


def merge_race_results(
    path: Path,
    replacements: list[RaceCriterionScore],
    evaluated: set[tuple[str, int]],
) -> list[RaceCriterionScore]:
    """与 merge_results 同一套原子替换语义，作用于 RACE 分数文件。"""
    existing = load_race_results(path) if path.is_file() else []
    merged = [row for row in existing if (row.case_id, row.attempt) not in evaluated]
    merged.extend(replacements)
    merged.sort(key=lambda row: (row.case_id, row.attempt, row.criterion_id))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(race_results_to_jsonl(merged), "utf-8")
    temporary.replace(path)
    return merged


def validate_case(case: EvalCase) -> list[str]:
    problems: list[str] = []
    if not case.prompt.strip():
        problems.append("prompt 为空")
    if not case.criteria:
        problems.append("没有任何 criterion，跑完无法判")
    seen: set[str] = set()
    for c in case.criteria:
        if c.id in seen:
            problems.append(f"criterion id 重复: {c.id}")
        seen.add(c.id)
        if c.dimension not in DIMENSIONS:
            problems.append(f"未知维度: {c.dimension}")
        if c.weight <= 0:
            problems.append(f"权重非正: {c.id}")
    if case.dimension_weights:
        total = sum(case.dimension_weights.values())
        if abs(total - 1.0) > 0.02:
            problems.append(f"维度权重和={total:.3f}，应为 1")
    return problems
