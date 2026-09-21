"""网页抓取 Provider 的最小能力契约。"""

from typing import Protocol

from deepresearcher.tools.web.fetch.models import SourceDocument


class FetchProvider(Protocol):
    """取得并标准化一个 URL；不负责 Evidence 抽取。

    失败分类与 SearchProvider 同规(账户级→ProviderExhaustedError、瞬时→
    retryable ToolRequestError、畸形→ToolParseError);另有一类抓取特有:
    来源可达但取不到可验证正文(验证码页/登录墙/空壳)→ SourceUnavailableError,
    它是 per-source 结论,不代表 provider 整体挂掉。
    """

    name: str

    async def afetch(
        self,
        url: str,
        *,
        fetch_timeout: float | None = None,
        parse_timeout: float | None = None,
    ) -> SourceDocument: ...
