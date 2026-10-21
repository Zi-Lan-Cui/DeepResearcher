"""搜索 Provider 的最小能力契约。"""

from typing import Protocol

from deepresearcher.tools.web.search.models import SearchResult


class SearchProvider(Protocol):
    """一个搜索端点的适配层；SearchService 负责扇出、限流与熔断。

    异常类型决定熔断语义，新 provider 必须原样遵守：
    账户级不可用（鉴权失败、额度耗尽、key 无效）抛 ProviderExhaustedError，
    retryable=False，记为跨 worker 共享的熔断状态，不得压成通用请求错误；
    瞬时网络故障、超时或可重试 5xx 抛 retryable=True 的 ToolRequestError，
    传输层限流时另附 rate_limit_reset_ts；响应结构畸形抛 ToolParseError。
    """

    async def asearch(self, query: str, limit: int) -> list[SearchResult]: ...
