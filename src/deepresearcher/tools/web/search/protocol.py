"""搜索 Provider 的最小能力契约。"""

from typing import Protocol

from deepresearcher.tools.web.search.models import SearchResult


class SearchProvider(Protocol):
    """一个搜索端点的适配层；SearchService 负责扇出、限流与熔断。

    失败语义是本契约的一部分：SearchService 的 except 分支按异常类型分流，新 provider
    必须原样遵守，否则跨 worker 的熔断一致性对它失效时不会有任何报错：
      - 账户级不可用，即鉴权失败、额度耗尽或 key 无效：抛 ProviderExhaustedError，
        retryable=False，由 SearchService 记为熔断状态并跨 worker 共享；不得压成通用请求错误。
      - 瞬时网络故障、超时或可重试的 5xx：抛 retryable=True 的 ToolRequestError；
        传输层限流时该错误另附 rate_limit_reset_ts。
      - 响应结构畸形、无法解析为 SearchResult：抛 ToolParseError。
    """

    async def asearch(self, query: str, limit: int) -> list[SearchResult]: ...
