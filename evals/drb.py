"""DeepResearch Bench 适配器：只引用、绝不拷贝。

DRB 代码是 MIT，但题目/criterion/专家参考文章是**数据集许可**——本模块
不在本仓库落任何一行 DRB 内容，只在运行时从 ``$DRB_ROOT`` 解析其 jsonl，
拼成我们的 :class:`~evals.schemas.EvalCase`。评测产物（分数、报告）是我们
自己的输出，可入库展示；原始数据集请使用者自行 clone（README 有指引）。

题集治理（防 rubric 过拟合）：中文 50 题按固定种子切 dev/holdout 两层——
dev 允许翻失败轨迹、归因、调提示词；holdout 只看分数。切分结果落
split_drb_zh.json 冻结，全组人共用同一份，杜绝"悄悄把某题挪回 dev"。
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

from evals.schemas import Criterion, EvalCase

DRB_ROOT_ENV = "DRB_ROOT"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def drb_root(override: str | Path | None = None) -> Path:
    """解析外部 DRB 数据集目录，不推测使用者的本地文件布局。

    命令行传入的 ``override`` 优先于 ``DRB_ROOT`` 环境变量。数据集不属于
    本仓库，因此二者均未配置时直接给出可操作的错误，而不是绑定开发者目录。
    """
    configured = override or os.environ.get(DRB_ROOT_ENV)
    if not configured:
        raise FileNotFoundError(
            "未配置 DRB 数据集目录。请传入 --drb-root，或设置 DRB_ROOT 环境变量。"
        )
    root = Path(configured).expanduser()
    if not (root / "data" / "prompt_data" / "query.jsonl").is_file():
        raise FileNotFoundError(f"DRB 根目录无效: {root}。请检查 --drb-root 或 DRB_ROOT 配置。")
    return root


def load_queries(root: Path) -> list[dict[str, Any]]:
    return _read_jsonl(root / "data" / "prompt_data" / "query.jsonl")


def load_criteria(root: Path) -> dict[int, dict[str, Any]]:
    rows = _read_jsonl(root / "data" / "criteria_data" / "criteria.jsonl")
    return {int(row["id"]): row for row in rows}


def load_references(root: Path) -> dict[int, str]:
    """专家参考文章（RACE 对照基准；judge 上下文用，永不提交进本仓库）。"""
    path = root / "data" / "test_data" / "cleaned_data" / "reference.jsonl"
    if not path.is_file():
        return {}
    return {int(row["id"]): str(row.get("article", "")) for row in _read_jsonl(path)}


def build_cases(
    root: Path | None = None,
    *,
    language: str = "zh",
    split_file: Path | None = None,
    split_name: str | None = None,  # dev | holdout | None=全部
) -> list[EvalCase]:
    root = root or drb_root()
    criteria_by_id = load_criteria(root)
    membership = (
        set(json.loads(split_file.read_text("utf-8"))[split_name])
        if split_file and split_name
        else None
    )
    cases: list[EvalCase] = []
    for row in load_queries(root):
        if row.get("language") != language:
            continue
        drb_id = int(row["id"])
        case_id = f"drb-{language}-{drb_id:03d}"
        if membership is not None and drb_id not in membership:
            continue
        rubric = criteria_by_id.get(drb_id)
        if rubric is None:
            raise KeyError(f"DRB 题 {drb_id} 缺 criteria.jsonl 条目")
        criteria: list[Criterion] = []
        for dimension, items in (rubric.get("criterions") or {}).items():
            for index, item in enumerate(items, start=1):
                criteria.append(
                    Criterion(
                        id=f"{dimension}-{index}",
                        dimension=dimension,
                        text=str(item.get("criterion", "")),
                        explanation=str(item.get("explanation", "")),
                        weight=float(item.get("weight", 1.0)),
                    )
                )
        if not criteria:
            raise KeyError(f"DRB 题 {drb_id} 的 criterion 列表为空")
        cases.append(
            EvalCase(
                case_id=case_id,
                track="external",
                prompt=str(row["prompt"]),
                language=language,
                topic=str(row.get("topic", "")),
                # 官方纪律：全量 100 题一次跑不动；首轮 dev 每题 ×1 校准，
                # 里程碑轮次再升 ×3 计 pass^3。
                repeat=1,
                dimension_weights=dict(rubric.get("dimension_weight") or {}),
                criteria=tuple(criteria),
            )
        )
    return cases


def make_split(
    root: Path,
    *,
    language: str = "zh",
    dev_size: int = 20,
    seed: int = 20260908,
    out_path: Path,
) -> dict[str, list[int]]:
    """固定种子分层切分：先按 topic 轮转再抽样，dev/holdout 领域分布一致。"""
    queries = [q for q in load_queries(root) if q.get("language") == language]
    by_topic: dict[str, list[int]] = {}
    for row in queries:
        by_topic.setdefault(str(row.get("topic", "")), []).append(int(row["id"]))
    rng = random.Random(seed)
    ordered: list[int] = []
    queues = {topic: sorted(ids) for topic, ids in sorted(by_topic.items())}
    while any(queues.values()):
        for topic in list(queues):
            if queues[topic]:
                ordered.append(queues[topic].pop(0))
    shuffled = ordered[:]
    rng.shuffle(shuffled)
    split = {"dev": sorted(shuffled[:dev_size]), "holdout": sorted(shuffled[dev_size:])}
    out_path.write_text(json.dumps(split, ensure_ascii=False, indent=1) + "\n", "utf-8")
    return split
