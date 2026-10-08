"""模型装配与单次调用的唯一入口：build 出未经包装的 ChatOpenAI，节点在此做文本与结构化调用。

分层原则——谁在哪一层重试：
- 传输级，超时与断连：openai SDK 内建 max_retries 指数退避，所有经 ChatOpenAI
  发出的请求都归它，agent 循环与节点单次调用同样经过；
- 循环级：ModelRetryMiddleware（agents/middleware/retry.py）把最终失败的模型调用
  改写成指引文本，agent 得以继续；预算耗尽与账户级不可用是例外，
  retry_on 对它们豁免，让其冒泡到既有 fail-fast 收口；
- 内容级：本文件的 repair 循环，JSON 不合规时重新提交给模型修复，langchain
  没有提供这一层;
- 压缩级，属例外：SummarizationMiddleware 在 before_model 节点内直调模型做摘要，
  库内没有注入口，这条路径不经过预算检查与循环级重试。它只在上下文越限后发生，
  每次压缩至多一笔有界调用，是本分层原则明确承认的第四条模型调用路径。
本模块不包装模型:create_agent 与中间件拿到的都是同一个未包装的 ChatOpenAI,
网络抖动的重试归 SDK,不在 runnable 层再建一层重试。
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
from deepresearcher.usage_runtime import enforce_usage_budget

SchemaT = TypeVar("SchemaT", bound=BaseModel)
STRUCTURED_ERRORS = (OutputParserException, ValidationError, ValueError)


class LLMError(AgentError):
    code = "llm_error"


class LLMConfigurationError(LLMError):
    """研究链所需的模型能力未在应用装配期提供。"""

    code = "llm_configuration"


# 账户级不可用——key 无效、无余额、硬限流——刻意不包装成自定义异常：
# 模型调用失败时抛出的是 SDK 原始异常，executor 用本函数将其归类为
# invalid_key、forbidden、insufficient_credit、rate_limited 四种稳定 user_code 再交给前端。
# openai 等 SDK 的 APIStatusError 带 .status_code。
_LLM_FATAL_STATUS = {401: "invalid_key", 402: "insufficient_credit", 403: "forbidden"}


def classify_llm_error(exc: BaseException) -> str | None:
    """把模型调用抛出的原始异常归类为 llm 不可用的稳定 user_code。

    参数:
        exc: SDK 或传输层抛出的异常。
    返回:
        命中账户级不可用时返回对应 user_code，其余情况返回 None。
    """
    status = getattr(exc, "status_code", None)
    if status in _LLM_FATAL_STATUS:
        return _LLM_FATAL_STATUS[status]
    if status == 429:
        # 429 通常已被 SDK 内建重试处理；到这里说明重试之后仍不可用。
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
    """只在已收到不合规 JSON 时 repair；传输级重试由 openai SDK 内建机制处理。

    修复预算只从配置 LLM_STRUCTURED_REPAIR_ATTEMPTS 读取；测试的注入点是
    各节点的 ``invoke_structured=`` 关键字，本函数不提供第二个注入点。

    抛出:
        OutputParserException / ValidationError / ValueError:
            repair 预算耗尽仍未通过校验时，抛最后一次异常。
    """
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
