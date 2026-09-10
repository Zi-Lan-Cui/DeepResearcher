"""评测数据契约：case / criterion / criterion 级结果的双轨统一形态。

外轨（DeepResearch Bench 专家二元 criterion + 四维官方权重）与内轨（自研行为
断言）都归一到 ``EvalCase``；三方评判（确定性 scorer、LLM judge、人工标注）
全部落到同一批 ``CriterionResult`` 记录上——人机一致率、unknown 率、pass^k
都在这一张表上计算，不存在第二套事实。

设计纪律（对齐评测方法论）：
- criterion 一律二元可判（yes/no），并保留 unknown 出口——某条 criterion 的
  unknown 率异常即 rubric 定义歧义的反查信号；
- 打分单位是 criterion，不是整篇报告印象分：judge 每条独立调用；
- 工程行为断言走 ``deterministic:<check>``，由代码判，绝不劳驾 LLM。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

# 外轨四维沿用 DRB 官方维度名；behavior/grounding 为内轨扩展。
DIMENSIONS = (
    "comprehensiveness",
    "insight",
    "instruction_following",
    "readability",
    "behavior",
    "grounding",
)

Verdict = Literal["yes", "no", "unknown"]
JUDGE_SCORER = "judge"


@dataclass(frozen=True)
class Criterion:
    id: str
    dimension: str
    text: str
    explanation: str = ""
    weight: float = 1.0
    # "judge" 或 "deterministic:<check>"（check 注册表见 deterministic.py）
    scorer: str = JUDGE_SCORER


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    track: Literal["external", "behavior"]
    prompt: str
    criteria: tuple[Criterion, ...]
    language: str = "zh"
    topic: str = ""
    group: str = ""  # 内轨子类：B 澄清 / C 口径冲突 / D 保守交付 / F 恢复缓存
    repeat: int = 1
    require_clarify: bool = False
    clarify_answer: str = ""
    duplicate_answer: bool = False  # 回答后再提交一次，期待 409（payload 单次消费）
    fault: str = "none"  # none|rerun_cache|kill|stop（后两者需 harness 掌控 worker 进程）
    phase: int = 1  # 2 = 骨架已备、故障编排未接线的题，CLI 默认跳过并明示
    dimension_weights: dict[str, float] = field(default_factory=dict)  # 外轨官方权重

    def judge_criteria(self) -> list[Criterion]:
        return [c for c in self.criteria if c.scorer == JUDGE_SCORER]

    def deterministic_criteria(self) -> list[Criterion]:
        return [c for c in self.criteria if c.scorer.startswith("deterministic:")]

    def check_of(self, criterion: Criterion) -> str:
        return criterion.scorer.split(":", 1)[1]


@dataclass(frozen=True)
class CriterionResult:
    case_id: str
    attempt: int
    criterion_id: str
    dimension: str
    verdict: Verdict
    source: Literal["deterministic", "judge", "human"]
    reason: str = ""


# ---- JSONL 往返 ----


def cases_to_jsonl(cases: list[EvalCase]) -> str:
    lines = []
    for case in cases:
        row = asdict(case)
        row["criteria"] = [asdict(c) for c in case.criteria]
        lines.append(json.dumps(row, ensure_ascii=False))
    return "\n".join(lines) + "\n"


def case_from_row(row: dict[str, Any]) -> EvalCase:
    criteria = tuple(Criterion(**c) for c in row.get("criteria", []))
    return EvalCase(**{**row, "criteria": criteria})


def load_cases(path: Path) -> list[EvalCase]:
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    return [case_from_row(row) for row in rows]


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


# ---- 校验 ----


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
        if c.scorer != JUDGE_SCORER and not c.scorer.startswith("deterministic:"):
            problems.append(f"未知 scorer: {c.scorer}")
        if c.weight <= 0:
            problems.append(f"权重非正: {c.id}")
    if case.require_clarify and not case.clarify_answer:
        problems.append("require_clarify 但没有澄清回答，runner 将永远等不到终态")
    if case.dimension_weights:
        total = sum(case.dimension_weights.values())
        if abs(total - 1.0) > 0.02:
            problems.append(f"维度权重和={total:.3f}，应为 1")
    return problems
