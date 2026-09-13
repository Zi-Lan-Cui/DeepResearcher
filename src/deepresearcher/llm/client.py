"""模型客户端创建，不包含业务节点逻辑。"""

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from deepresearcher.config import Settings, get_settings
from deepresearcher.llm.errors import LLMConfigurationError
from deepresearcher.llm.structured import LLMInvoker


def build_llm(settings: Settings | None = None) -> LLMInvoker:
    """构造研究链所需的 LLM；配置缺失应在应用装配期失败。"""
    config = (settings or get_settings()).llm
    if not config.configured:
        raise LLMConfigurationError(
            "缺少 LLM_API_KEY、LLM_BASE_URL 或 LLM_MODEL_ID，无法构造研究应用。"
        )
    model = ChatOpenAI(
        model=config.model,
        api_key=SecretStr(config.api_key),
        base_url=config.base_url,
        temperature=config.temperature,
        timeout=config.timeout,
    )
    return LLMInvoker(model, config.retry)
