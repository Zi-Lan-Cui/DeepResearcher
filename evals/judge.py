"""二元 rubric LLM judge：每条 criterion 独立调用，允许 unknown。

方法论对齐（美团的二元化改造 + Anthropic 的 unknown 出口建议）：
- 打分单位是"是/否断言"，不是 0-10 印象分——整体打分的人机一致率基线太低
  （DRB 官方自证人工互评一致率仅 ~69%），逐条二元判 + 人工同表标注才有
  可计算的一致率；
- 每条 criterion 独立调用：一次长上下文里顺判 12 条会互相污染；
- unknown 只表示 criterion 本身歧义、材料损坏或评审器无法工作；
  报告未覆盖 criterion 是明确的 no，不能用 unknown 逃避失分；
- 外轨带专家参考文章做对照（RACE 的 reference-based 语义），内轨行为题
  不走 judge（全部 deterministic）。
"""

from __future__ import annotations

import asyncio
import os
from typing import Protocol

from evals.schemas import Criterion, CriterionResult, EvalCase

SYSTEM_PROMPT = """你是严格、保守的研究报告评审员。你将看到一份研究报告、一条二元评审标准，\
可能还有一份专家参考文章（仅作覆盖面与深度的对照，不是唯一正确答案来源）。

只输出一个词：yes / no / unknown
- yes：报告确凿满足该标准；
- no：报告确凿不满足，包括缺失 criterion 要求的内容、内容错误或违背要求；
- unknown：只在 criterion 本身有歧义、输入材料损坏，或该 criterion 明确要求报告与参考文章之外的事实才能裁决时使用。
“报告没写到”一律判 no，不是 unknown。不要输出任何其他文字。"""

MAX_REPORT_CHARS = 60_000
MAX_REFERENCE_CHARS = 30_000


class JudgeInvoker(Protocol):
    async def complete(self, system: str, user: str) -> str: ...


def build_user_prompt(criterion: Criterion, report: str, reference: str | None) -> str:
    parts = [
        f"【评审标准】{criterion.text}",
    ]
    if criterion.explanation:
        parts.append(f"【标准说明】{criterion.explanation}")
    if reference:
        parts.append(f"【专家参考文章（对照用）】\n{reference[:MAX_REFERENCE_CHARS]}")
    parts.append(f"【待评报告】\n{report[:MAX_REPORT_CHARS]}")
    return "\n\n".join(parts)


def parse_verdict(raw: str) -> str:
    text = raw.strip().lower()
    for verdict in ("yes", "no", "unknown"):  # 顺序有意：unknown 含 "no" 前先匹配词首
        if text.startswith(verdict):
            return verdict
    return "unknown"


async def judge_case(
    case: EvalCase,
    *,
    report: str,
    attempt: int,
    invoker: JudgeInvoker,
    reference: str | None = None,
) -> list[CriterionResult]:
    """逐条独立判定；调用方负责把结果与 human/deterministic 记录合并。"""
    concurrency = max(1, int(os.environ.get("EVAL_JUDGE_CONCURRENCY", "5")))
    total_timeout = float(os.environ.get("EVAL_JUDGE_TOTAL_TIMEOUT_SECONDS", "120"))
    gate = asyncio.Semaphore(concurrency)

    async def judge_criterion(criterion: Criterion) -> CriterionResult:
        user = build_user_prompt(criterion, report, reference)
        try:
            async with gate:
                raw = await asyncio.wait_for(
                    invoker.complete(SYSTEM_PROMPT, user), timeout=total_timeout
                )
            verdict, reason = parse_verdict(raw), ""
        except Exception as exc:  # noqa: BLE001 - 单条 judge 失败降级 unknown，不废整题
            verdict, reason = "unknown", f"judge 调用失败: {type(exc).__name__}: {str(exc)[:120]}"
        return CriterionResult(
            case_id=case.case_id,
            attempt=attempt,
            criterion_id=criterion.id,
            dimension=criterion.dimension,
            verdict=verdict,  # type: ignore[arg-type]
            source="judge",
            reason=reason,
        )

    return list(await asyncio.gather(*(judge_criterion(item) for item in case.judge_criteria())))


class OpenAICompatJudge:
    """生产 invoker：judge 身份取自评测侧 env，不回写也不依赖引擎配置。

    与被测同模型自评为已知测量污染(self-preference)，因此回落同模型不再无声
    生效：必须 EVAL_SELF_JUDGE=1 显式认领，且 judge_meta 落盘、报表打水印，
    自评分数只能用于调试。异构 judge 用 EVAL_JUDGE_MODEL(+可选
    EVAL_JUDGE_BASE_URL/EVAL_JUDGE_API_KEY) 指向另一家族即可。
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
        self._max_tokens = int(os.environ.get("EVAL_JUDGE_MAX_TOKENS", "16"))

    @property
    def identity(self) -> dict[str, object]:
        """落进 judge_meta.json 的评审身份——报表水印的数据源。"""
        return {"model": self._model, "self_judging": self.self_judging}

    async def complete(self, system: str, user: str) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=self._max_tokens,
            # 该网关是思考型模型（引擎各处亦如此关闭）：不禁用则 token 预算全花在
            # 隐藏思考上、content 返回空串 → 解析成 unknown，把整批评测器打成 0 命中。
            extra_body={"thinking": {"type": "disabled"}},
        )
        return response.choices[0].message.content or ""
