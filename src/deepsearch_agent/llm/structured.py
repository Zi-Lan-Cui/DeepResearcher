"""配置化的 LLM 调用：传输重试与结构化修复保持分离。"""

import json
from collections.abc import Sequence
from typing import Any, TypeVar

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from deepsearch_agent.config import LLMRetryConfig
from deepsearch_agent.llm.errors import LLMConfigurationError
from deepsearch_agent.llm.retry import with_transport_retry

SchemaT = TypeVar("SchemaT", bound=BaseModel)
STRUCTURED_ERRORS = (OutputParserException, ValidationError, ValueError)


class LLMInvoker:
    """应用装配期创建的模型调用器，所有调用共享同一明确策略。"""

    def __init__(self, model: BaseChatModel, retry: LLMRetryConfig):
        if model is None:
            raise LLMConfigurationError("LLMInvoker 需要已配置的聊天模型，不能传入 None。")
        self._model = model
        self._retry = retry

    def bind_tools(self, tools: list[type[BaseModel]], **kwargs: Any):
        """绑定工具并统一接入 LLM 的传输重试策略。

        Agent 不应访问 ``_model``；底层模型替换、重试和后续观测都由此
        公开边界集中处理。
        """
        runnable = self._model.bind_tools(tools, **kwargs)
        return with_transport_retry(runnable, self._retry)

    async def ainvoke_text(
        self,
        messages: Sequence[BaseMessage],
        *,
        request_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        runnable = self._model.bind(**request_kwargs) if request_kwargs else self._model
        return await with_transport_retry(runnable, self._retry).ainvoke(list(messages))

    async def ainvoke_structured(
        self,
        schema: type[SchemaT],
        messages: Sequence[BaseMessage],
        *,
        request_kwargs: dict[str, Any] | None = None,
    ) -> SchemaT:
        """只在已收到不合规 JSON 时 repair；传输失败由 transport retry 处理。"""
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        contract = SystemMessage(
            content=(
                "必须只返回一个合法 JSON object，不要输出 Markdown、代码围栏或解释。"
                "JSON 必须符合以下 Schema；字段名、嵌套结构和枚举值不得自行改动：\n"
                f"{schema_json}"
            )
        )
        runnable = self._model.with_structured_output(schema, method="json_mode")
        if request_kwargs:
            runnable = runnable.bind(**request_kwargs)
        runnable = with_transport_retry(runnable, self._retry)
        structured_messages = [contract, *messages]
        current_messages = list(structured_messages)
        for attempt in range(self._retry.structured_repair_attempts + 1):
            try:
                return await runnable.ainvoke(current_messages)
            except STRUCTURED_ERRORS as exc:
                if attempt >= self._retry.structured_repair_attempts:
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


async def ainvoke_structured(
    llm: LLMInvoker,
    schema: type[SchemaT],
    messages: Sequence[BaseMessage],
    *,
    request_kwargs: dict[str, Any] | None = None,
) -> SchemaT:
    """保留单一适配函数，便于节点测试注入；策略由 LLMInvoker 持有。"""
    if llm is None:
        raise LLMConfigurationError("结构化调用需要已装配的 LLMInvoker。")
    return await llm.ainvoke_structured(schema, messages, request_kwargs=request_kwargs)
