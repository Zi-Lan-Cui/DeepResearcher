"""模型装配与单次调用的统一入口"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar, cast

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr, ValidationError

from deepresearcher.config import Settings, get_settings
from deepresearcher.errors import AgentError
from deepresearcher.observability.logging_config import get_logger
from deepresearcher.usage_runtime import enforce_usage_budget

SchemaT = TypeVar("SchemaT", bound=BaseModel)
STRUCTURED_ERRORS = (OutputParserException, ValidationError, ValueError)

_logger = get_logger("deepresearcher.llm")


class LLMError(AgentError):
    code = "llm_error"


class LLMConfigurationError(LLMError):
    """研究链所需的模型能力未在应用装配期提供。"""

    code = "llm_configuration"


_LLM_FATAL_STATUS = {401: "invalid_key", 402: "insufficient_credit", 403: "forbidden"}


def classify_llm_error(exc: BaseException) -> str | None:
    """把模型调用抛出的原始异常归类为 llm 不可用的稳定 user_code。

    参数：
        exc: SDK 或传输层抛出的异常。
    返回：
        命中账户级不可用时返回对应 user_code，其余情况返回 None。
    """
    status = getattr(exc, "status_code", None)
    if status in _LLM_FATAL_STATUS:
        return _LLM_FATAL_STATUS[status]
    if status == 429:
        # 429 通常已被 SDK 内建重试处理；到这里说明重试之后仍不可用。
        return "rate_limited"
    return None


def build_llm(
    settings: Settings | None = None,
    *,
    request_overrides: Mapping[str, Any] | None = None,
) -> BaseChatModel:
    """构造研究链所需的裸模型；配置缺失应在应用装配期失败。

    参数：
        settings: 省略时取 get_settings()。
        request_overrides: 该实例的请求参数覆盖，语义见 derive_llm。
    """
    config = (settings or get_settings()).llm
    if not config.configured:
        raise LLMConfigurationError(
            "缺少 LLM_API_KEY、LLM_BASE_URL 或 LLM_MODEL_ID，无法构造LLM客户端。"
        )
    llm = ChatOpenAI(
        model=config.model,
        api_key=SecretStr(config.api_key),
        base_url=config.base_url,
        temperature=config.temperature,
        timeout=config.timeout,
        max_completion_tokens=config.max_completion_tokens,
    )
    return derive_llm(llm, request_overrides)


def derive_llm(
    llm: BaseChatModel,
    request_overrides: Mapping[str, Any] | None = None,
) -> BaseChatModel:
    """派生一份带请求参数覆盖的模型实例；没有覆盖项时原样返回。"""
    if not request_overrides:
        return llm
    updates = dict(request_overrides)
    fields = getattr(type(llm), "model_fields", None)
    if fields is not None:
        unknown = sorted(set(updates) - set(fields))
        if unknown:
            raise LLMConfigurationError(
                f"请求参数 {', '.join(unknown)} 不是 {type(llm).__name__} 的字段；"
                "供应商私有的思考开关请放进 extra_body。"
            )
    extra = updates.get("extra_body")
    if isinstance(extra, Mapping):
        # 与构造期已有的 extra_body 合并，覆盖项优先：派生不能把基线已有的键抹掉。
        base_extra = getattr(llm, "extra_body", None) or {}
        updates["extra_body"] = {**dict(base_extra), **dict(extra)}
    copier = getattr(llm, "model_copy", None)
    if copier is None:
        # 有覆盖项却派生不出来，必须报错：静默返回基线等于开关没生效。
        raise LLMConfigurationError(
            f"{type(llm).__name__} 不能按请求参数派生；覆盖项只支持 pydantic 模型实例。"
        )
    return copier(update=updates)


async def ainvoke_text(
    llm: BaseChatModel,
    messages: Sequence[BaseMessage],
    *,
    request_kwargs: dict[str, Any] | None = None,
) -> Any:
    """节点的单发文本调用；用量检查在调用边界强制执行。"""
    await enforce_usage_budget()
    runnable = llm.bind(**request_kwargs) if request_kwargs else llm
    return await runnable.ainvoke(list(messages))


async def ainvoke_structured(
    llm: BaseChatModel,
    schema: type[SchemaT],
    messages: Sequence[BaseMessage],
    *,
    request_kwargs: dict[str, Any] | None = None,
) -> SchemaT:
    """按 Schema 取结构化输出；输出不合规时自动重发。"""
    max_attempts = get_settings().llm.structured_max_attempts
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
    for attempt in range(max_attempts):
        try:
            await enforce_usage_budget()
            # with_structured_output 的运行期产物即 SchemaT；类型系统只看到 Runnable 宽签名。
            return cast(SchemaT, await runnable.ainvoke(current_messages))
        except STRUCTURED_ERRORS as exc:
            if attempt + 1 >= max_attempts:
                raise
            # 重发发生在调用方视野之外，只在日志留痕，调用方看到的是最终结果。
            _logger.warning(
                "structured_output_repair attempt=%d/%d schema=%s error=%s",
                attempt + 1,
                max_attempts,
                schema.__name__,
                type(exc).__name__,
            )
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
