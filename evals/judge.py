"""二元 rubric LLM judge：每条 criterion 独立调用，允许 unknown。

方法论对齐（美团的二元化改造 + Anthropic 的 unknown 出口建议）：
- 打分单位是"是/否断言"，不是 0-10 印象分——整体打分的人机一致率基线太低
  （DRB 官方自证人工互评一致率仅 ~69%），逐条二元判 + 人工同表标注才有
  可计算的一致率；
- 每条 criterion 独立调用：一次长上下文里顺判 12 条会互相污染；
- unknown 是一等公民：报告没写到、或 criterion 本身歧义时**必须** unknown，
  禁止猜测——unknown 率就是 rubric 质量的体检指标；
- 外轨带专家参考文章做对照（RACE 的 reference-based 语义），内轨行为题
  不走 judge（全部 deterministic）。
"""

from __future__ import annotations

from typing import Protocol

from evals.schemas import Criterion, CriterionResult, EvalCase

SYSTEM_PROMPT = """你是严格、保守的研究报告评审员。你将看到一份研究报告、一条二元评审标准，\
可能还有一份专家参考文章（仅作覆盖面与深度的对照，不是唯一正确答案来源）。

只输出一个词：yes / no / unknown
- yes：报告确凿满足该标准；
- no：报告确凿不满足（缺失、错误或违背）；
- unknown：依据所给材料无法裁决（报告未涉及该角度、标准表述歧义、或需要外部信息）。
宁可 unknown，绝不猜测。不要输出任何其他文字。"""

MAX_REPORT_CHARS = 60_000
MAX_REFERENCE_CHARS = 30_000


class JudgeInvoker(Protocol):
    async def complete(self, system: str, user: str) -> str: ...


def build_user_prompt(
    criterion: Criterion, report: str, reference: str | None
) -> str:
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
    results: list[CriterionResult] = []
    for criterion in case.judge_criteria():
        user = build_user_prompt(criterion, report, reference)
        try:
            raw = await invoker.complete(SYSTEM_PROMPT, user)
            verdict, reason = parse_verdict(raw), ""
        except Exception as exc:  # noqa: BLE001 - 单条 judge 失败降级 unknown，不废整题
            verdict, reason = "unknown", f"judge 调用失败: {type(exc).__name__}: {str(exc)[:120]}"
        results.append(
            CriterionResult(
                case_id=case.case_id,
                attempt=attempt,
                criterion_id=criterion.id,
                dimension=criterion.dimension,
                verdict=verdict,  # type: ignore[arg-type]
                source="judge",
                reason=reason,
            )
        )
    return results


class OpenAICompatJudge:
    """生产用 invoker：复用引擎同一 LLM 网关配置，可用 EVAL_JUDGE_* 覆盖。"""

    def __init__(self) -> None:
        import os

        from openai import AsyncOpenAI

        from deepsearch_agent.config import get_settings

        llm = get_settings().llm
        if not llm.configured:
            raise RuntimeError("LLM_API_KEY/LLM_BASE_URL/LLM_MODEL_ID 未配置，无法构造 judge")
        self._client = AsyncOpenAI(api_key=llm.api_key, base_url=llm.base_url)
        self._model = os.environ.get("EVAL_JUDGE_MODEL", "").strip() or llm.model
        self._max_tokens = int(os.environ.get("EVAL_JUDGE_MAX_TOKENS", "8"))

    async def complete(self, system: str, user: str) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=self._max_tokens,
        )
        return response.choices[0].message.content or ""
