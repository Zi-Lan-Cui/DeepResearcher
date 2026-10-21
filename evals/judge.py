"""rubric LLM judge：RACE 对齐 DeepResearchBench 官方的 0-10 双文对比分。

- 判定单位是逐条 criterion,每条独立调用:一次长上下文里连判多条会互相干扰;
- 同一调用内被测与专家参考文章各打 0-10 分,聚合走官方加权与
  target/(target+reference) 归一(见 aggregate);
- 两篇只称文章一/文章二,judge 不知道身份,正序与换序各判一次取均值消位次偏置;
- 单条调用或解析失败记双 None,聚合端按官方口径把无分 criterion 排除出加权分母。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Protocol

from evals.schemas import Criterion, EvalCase, RaceCriterionScore

MAX_REPORT_CHARS = 60_000
MAX_REFERENCE_CHARS = 30_000


RACE_SYSTEM_PROMPT = """你是严格、细致、客观的调研文章评估专家，擅长针对具体评审标准对比两篇同一任务下的调研文章，并分别打分。

打分规则：对两篇文章各自按 0-10 连续分数打分，分数高低反映该文章在此标准上的表现：
* 0-2分：表现很差，几乎完全不符合标准要求。
* 2-4分：表现较差，少量符合标准要求，但有明显不足。
* 4-6分：表现中等，基本符合标准要求。
* 6-8分：表现较好，大部分符合标准要求，有可取之处。
* 8-10分：表现出色，完全或超预期符合标准要求。

只输出一个 JSON 对象，不要包含任何其他文字、引言或代码围栏：
{"analysis": "逐点对比两篇文章在此标准上的表现", "article_1_score": 0, "article_2_score": 0}"""

RACE_MAX_OUTPUT_TOKENS_DEFAULT = 2_048


class JudgeInvoker(Protocol):
    async def complete(self, system: str, user: str, max_tokens: int) -> str: ...


def build_race_user_prompt(
    case: EvalCase,
    criterion: Criterion,
    report: str,
    reference: str,
    *,
    swap_positions: bool = False,
) -> str:
    # 两篇只称文章一/文章二：judge 不应知道哪篇是被测、哪篇是参考（与官方一致），
    # 身份披露会引入"专家"标签效应，位次校准见 swap_positions。
    parts = [
        f"**任务背景**\n{case.prompt}",
        f"**评审标准**\n{criterion.text}",
    ]
    if criterion.explanation:
        parts.append(f"**标准说明**\n{criterion.explanation}")
    clipped_report = report[:MAX_REPORT_CHARS]
    clipped_reference = reference[:MAX_REFERENCE_CHARS]
    first_text, second_text = (
        (clipped_reference, clipped_report)
        if swap_positions
        else (clipped_report, clipped_reference)
    )
    parts.append(f"**文章一**\n{first_text}")
    parts.append(f"**文章二**\n{second_text}")
    return "\n\n".join(parts)


def parse_race_scores(raw: str) -> tuple[float, float, str]:
    """从 judge 输出提取双文分数；解析失败由调用方记 None,不在此兜底均值。"""
    text = raw.strip().strip("`")
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(text[start : end + 1])
            target = float(payload["article_1_score"])
            reference = float(payload["article_2_score"])
            analysis = str(payload.get("analysis", ""))[:800]
            return _clamp_score(target), _clamp_score(reference), analysis
        except (KeyError, TypeError, ValueError):
            pass
    target_match = re.search(r'"article_1_score"\s*:\s*([0-9.]+)', raw)
    reference_match = re.search(r'"article_2_score"\s*:\s*([0-9.]+)', raw)
    if target_match and reference_match:
        return (
            _clamp_score(float(target_match.group(1))),
            _clamp_score(float(reference_match.group(1))),
            "宽松解析",
        )
    raise ValueError(f"无法解析双文分数: {raw[:120]}")


def _clamp_score(value: float) -> float:
    return max(0.0, min(10.0, value))


async def judge_case_race(
    case: EvalCase,
    *,
    report: str,
    attempt: int,
    invoker: JudgeInvoker,
    reference: str,
    swap_positions: bool = False,
) -> list[RaceCriterionScore]:
    """RACE 口径：每条 criterion 一次调用，被测与参考文章同台各打 0-10 分。

    参数 reference 必须来自 DRB 的专家参考文章；swap_positions 为 True 时参考文章
    放文章一位（位次校准），分数映射回 target/reference。标准流程正序、换序各调
    一次本函数。单条失败或解析失败记双 None，聚合端按官方口径把无分 criterion
    排除出加权分母。
    """
    concurrency = max(1, int(os.environ.get("EVAL_JUDGE_CONCURRENCY", "5")))
    total_timeout = float(os.environ.get("EVAL_JUDGE_TOTAL_TIMEOUT_SECONDS", "180"))
    max_tokens = int(
        os.environ.get("EVAL_JUDGE_RACE_MAX_TOKENS", str(RACE_MAX_OUTPUT_TOKENS_DEFAULT))
    )
    gate = asyncio.Semaphore(concurrency)

    async def score_criterion(criterion: Criterion) -> RaceCriterionScore:
        user = build_race_user_prompt(
            case, criterion, report, reference, swap_positions=swap_positions
        )
        target_score: float | None = None
        reference_score: float | None = None
        analysis = ""
        try:
            async with gate:
                raw = await asyncio.wait_for(
                    invoker.complete(RACE_SYSTEM_PROMPT, user, max_tokens=max_tokens),
                    timeout=total_timeout,
                )
            first_score, second_score, analysis = parse_race_scores(raw)
            if swap_positions:
                target_score, reference_score = second_score, first_score
            else:
                target_score, reference_score = first_score, second_score
        except Exception as exc:  # noqa: BLE001 - 单条失败记 None,不拖垮整题
            analysis = f"judge 调用或解析失败: {type(exc).__name__}: {str(exc)[:120]}"
        return RaceCriterionScore(
            case_id=case.case_id,
            attempt=attempt,
            criterion_id=criterion.id,
            dimension=criterion.dimension,
            target_score=target_score,
            reference_score=reference_score,
            analysis=analysis,
        )

    return list(await asyncio.gather(*(score_criterion(item) for item in case.criteria)))


class OpenAICompatJudge:
    """生产 invoker：judge 身份取自评测侧 env,不回写也不依赖引擎配置。

    judge 与被测同模型时 self-preference 会污染分数,默认报错而不是静默回
    落:显式设 EVAL_SELF_JUDGE=1 才允许自评,此时自评状态写入 judge_meta、
    报表带警示,分数只用于调试。独立 judge 用 EVAL_JUDGE_MODEL 指向另一模型
    家族,可选配 EVAL_JUDGE_BASE_URL 与 EVAL_JUDGE_API_KEY。
    """

    def __init__(self) -> None:
        from openai import AsyncOpenAI

        from deepresearcher.config import get_settings

        llm = get_settings().llm
        model = os.environ.get("EVAL_JUDGE_MODEL", "").strip() or llm.model
        base_url = os.environ.get("EVAL_JUDGE_BASE_URL", "").strip() or llm.base_url
        api_key = os.environ.get("EVAL_JUDGE_API_KEY", "").strip() or llm.api_key
        if not (api_key and base_url and model):
            raise RuntimeError("judge 未配置：设 EVAL_JUDGE_* 或配好引擎 LLM_* 供回落")
        self._model = model
        self.self_judging = model == llm.model
        if self.self_judging and os.environ.get("EVAL_SELF_JUDGE", "").strip() != "1":
            raise RuntimeError(
                f"judge 与被测同模型({model})：self-preference 会虚高 Report Quality。"
                "换 EVAL_JUDGE_MODEL 指向异构家族；确要自评调试再设 EVAL_SELF_JUDGE=1。"
            )
        timeout_seconds = float(os.environ.get("EVAL_JUDGE_TIMEOUT_SECONDS", "120"))
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=1,
        )

    @property
    def identity(self) -> dict[str, object]:
        """写入 judge_meta.json 的评审身份,供报表标注是否自评。"""
        return {"model": self._model, "self_judging": self.self_judging}

    async def complete(self, system: str, user: str, max_tokens: int) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=max_tokens,
            # 网关模型是思考型:不禁用思考,max_tokens 预算会全花在隐藏思考上,
            # content 返回空串、被解析成 unknown,整批评测判定失效。
            extra_body={"thinking": {"type": "disabled"}},
        )
        return response.choices[0].message.content or ""
