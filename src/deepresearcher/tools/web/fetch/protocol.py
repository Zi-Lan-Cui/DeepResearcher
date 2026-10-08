"""网页抓取 Provider 的最小能力契约。"""

from typing import Protocol

from deepresearcher.tools.web.fetch.models import SourceDocument


class FetchProvider(Protocol):
    """取得并标准化一个 URL；不负责 Evidence 抽取。

    失败分类与 SearchProvider 相同：账户级不可用抛 ProviderExhaustedError，
    瞬时故障抛 retryable 的 ToolRequestError，响应畸形抛 ToolParseError。
    另有一类抓取特有：来源可达但取不到可验证正文，例如验证码页、登录墙或
    动态空壳，抛 SourceUnavailableError；它只对该来源成立，不代表提供方整体不可用。
    """

    name: str

    async def afetch(
        self,
        url: str,
        *,
        fetch_timeout: float | None = None,
        parse_timeout: float | None = None,
    ) -> SourceDocument: ...
