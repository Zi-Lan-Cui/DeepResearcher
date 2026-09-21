"""搜索 Provider 的最小能力契约。"""

from typing import Protocol

from deepresearcher.tools.web.search.models import SearchResult


class SearchProvider(Protocol):
    """一个搜索端点的适配层；SearchService 负责扇出、限流与熔断。

    失败语义是本契约的一部分——SearchService 的 except 分支按异常类型分流,新
    provider 必须原样遵守,否则"一处发现、全体停手"的熔断对它静默失效:
      - 账户级不可用(鉴权失败/额度耗尽/key 无效)→ 抛 ProviderExhaustedError
        (retryable=False),触发跨 worker 健康位;绝不许压成通用请求错误。
      - 瞬时网络/超时/可重试的 5xx → 抛 retryable=True 的 ToolRequestError
        (传输层限流时另附 rate_limit_reset_ts)。
      - 响应结构畸形、无法解析为 SearchResult → 抛 ToolParseError。
    """

    async def asearch(self, query: str, limit: int) -> list[SearchResult]: ...
