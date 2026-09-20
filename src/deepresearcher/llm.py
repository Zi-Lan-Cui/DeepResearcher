"""模型装配与单次调用的唯一入口：build 出裸 ChatOpenAI，节点在此做文本/结构化调用。

分层原则——谁在哪一层重试：
- 传输级(超时/断连)：openai SDK 内建 max_retries 指数退避，覆盖一切经 ChatOpenAI
  发出的请求,agent 循环与节点单次调用同蒙其荫;
- 循环级：ModelRetryMiddleware(agents/middleware/retry.py)把最终失败的模型调用
  软化为指引文本,agent 得以继续;
- 内容级：本文件的 repair 循环——JSON 不合规时回炉,langchain 不提供。
曾有第四层"runnable 传输重试"(llm/retry.py),其谓词是内置 TimeoutError/ConnectionError,
而 langchain-openai 真实抛出的是 openai.*Error(不 subclass 之),实测永不命中,已删;
见 git blame。因此本模块不包装模型,create_agent 与中间件拿到的都是同一个裸实例。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, TypeVar, cast

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr, ValidationError

from deepresearcher.config import Settings, get_settings
from deepresearcher.errors import AgentError
from deepresearcher.observability.usage_runtime import enforce_usage_budget

SchemaT = TypeVar("SchemaT", bound=BaseModel)
STRUCTURED_ERRORS = (OutputParserException, ValidationError, ValueError)


class LLMError(AgentError):
    code = "llm_error"


class LLMConfigurationError(LLMError):
    """研究链所需的模型能力未在应用装配期提供。"""

    code = "llm_configuration"


class LLMUnavailableError(LLMError):
    """LLM 网关**账户级不可用**：key 无效 / 无余额 / 硬限流。

    连模型都调不动 → 无法"提醒 agent 收尾"，必须快速失败并把 user_code 透到前端。
    """

    code = "llm_unavailable"
    retryable = False

    def __init__(self, user_code: str, message: str):
        super().__init__(message)
        self.user_code = user_code  # invalid_key / forbidden / insufficient_credit / rate_limited


# openai 等 SDK 的 APIStatusError 带 .status_code；映射为稳定 user_code（不可自愈的那类）。
_LLM_FATAL_STATUS = {401: "invalid_key", 402: "insufficient_credit", 403: "forbidden"}


def classify_llm_error(exc: BaseException) -> str | None:
    """把传输层抛来的原始异常归类为 llm 不可用 user_code；非致命返回 None。"""
    status = getattr(exc, "status_code", None)
    if status in _LLM_FATAL_STATUS:
        return _LLM_FATAL_STATUS[status]
    if status == 429:
        # RateLimit 通常已被 SDK 内建重试消化；冒泡到这里=重试后仍不可用。
        return "rate_limited"
    return None


def build_llm(settings: Settings | None = None) -> ChatOpenAI:
    """构造研究链所需的裸模型；配置缺失应在应用装配期失败。"""
    config = (settings or get_settings()).llm
    if not config.configured:
        raise LLMConfigurationError(
            "缺少 LLM_API_KEY、LLM_BASE_URL 或 LLM_MODEL_ID，无法构造研究应用。"
        )
    return ChatOpenAI(
        model=config.model,
        api_key=SecretStr(config.api_key),
        base_url=config.base_url,
        temperature=config.temperature,
        timeout=config.timeout,
    )


async def ainvoke_text(
    llm: BaseChatModel,
    messages: Sequence[BaseMessage],
    *,
    request_kwargs: dict[str, Any] | None = None,
) -> Any:
    """节点的单发文本调用；用量闸口在调用边界强制。"""
    await enforce_usage_budget()
    runnable = llm.bind(**request_kwargs) if request_kwargs else llm
    return await runnable.ainvoke(list(messages))


async def ainvoke_structured(
    llm: BaseChatModel,
    schema: type[SchemaT],
    messages: Sequence[BaseMessage],
    *,
    request_kwargs: dict[str, Any] | None = None,
    repair_attempts: int | None = None,
) -> SchemaT:
    """只在已收到不合规 JSON 时 repair；传输抖动由 openai SDK 内建重试消化。

    repair_attempts 缺省读配置(LLM_STRUCTURED_REPAIR_ATTEMPTS)；保留为显式参数
    是节点测试的注入缝——同 ainvoke_structured 一直是模块函数而非方法的原因。
    """
    if repair_attempts is None:
        repair_attempts = get_settings().llm.structured_repair_attempts
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    contract = SystemMessage(
        content=(
            "必须只返回一个合法 JSON object，不要输出 Markdown、代码围栏或解释。"
            "JSON 必须符合以下 Schema；字段名、嵌套结构、枚举值不得自行改动：\n"
            f"{schema_json}"
        )
    )
    runnable = llm.with_structured_output(schema, method="json_mode")
    if request_kwargs:
        runnable = runnable.bind(**request_kwargs)
    structured_messages = [contract, *messages]
    current_messages = list(structured_messages)
    for attempt in range(repair_attempts + 1):
        try:
            await enforce_usage_budget()
            # with_structured_output 的运行期产物即 SchemaT;类型系统只看到 Runnable 宽签名。
            return cast(SchemaT, await runnable.ainvoke(current_messages))
        except STRUCTURED_ERRORS as exc:
            if attempt >= repair_attempts:
                raise
            current_messages = [
                *structured_messages,
                HumanMessage(
                    content=(
                        f"上一次 JSON 输出无法通过结构校验：{exc}。"
                        "请只重新输出符合 JSON Schema 的 JSON object，不要添加解释。"
                    )
                ),
            ]
    raise RuntimeError("structured output failed without a result")
