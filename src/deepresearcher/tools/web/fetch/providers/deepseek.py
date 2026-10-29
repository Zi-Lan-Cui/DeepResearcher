"""DeepSeek 网页抓取 FetchProvider：由服务端取回 Markdown 正文。"""

import asyncio
import hashlib
from pathlib import PurePosixPath

from deepresearcher.config import SearchConfig
from deepresearcher.tools.errors import (
    ProviderExhaustedError,
    SourceUnavailableError,
    ToolParseError,
    ToolRequestError,
)
from deepresearcher.tools.transport.http_client import HttpClient
from deepresearcher.tools.web.fetch.models import SourceDocument
from deepresearcher.tools.web.parsing import DocumentModality, parse_markdown
from deepresearcher.vocab import RETRIEVAL_DEEPSEEK_WEB_FETCH

# 401/402 是账号级不可用，同一个 key 重试无解；403/404 只针对这一次请求的来源。
_ACCOUNT_LEVEL = {
    401: ("invalid_key", "DeepSeek 凭据无效或已失效。"),
    402: ("insufficient_credit", "DeepSeek 账户余额不足。"),
}
# 接口把「取不到可用正文」压成一个 404，安全策略拒绝是 403：对单个来源都是终局，
# 不该重试，也不该算供应商整体不可用。
_SOURCE_REJECTED = {
    403: ("access_challenge", "DeepSeek 抓取被其安全策略拒绝。"),
    404: ("empty_content", "DeepSeek 未取到可提取的网页正文。"),
}


class DeepSeekFetchProvider:
    name = "deepseek"

    def __init__(self, config: SearchConfig, http_client: HttpClient | None = None):
        self._config = config
        self._http = http_client or HttpClient(config)

    async def afetch(
        self,
        url: str,
        *,
        fetch_timeout: float | None = None,
        parse_timeout: float | None = None,
    ) -> SourceDocument:
        started = asyncio.get_running_loop().time()
        try:
            response = await asyncio.wait_for(
                self._http.arequest(
                    "POST",
                    f"{self._config.deepseek_base_url}/v1/tools/fetch",
                    headers={"Authorization": f"Bearer {self._config.deepseek_api_key}"},
                    json={"url": url},
                    timeout=fetch_timeout or self._config.http_timeout,
                    request_kind="fetch",
                ),
                timeout=fetch_timeout or self._config.http_timeout,
            )
        except asyncio.TimeoutError as exc:
            raise ToolRequestError("DeepSeek 抓取请求超时。") from exc
        except ToolRequestError as exc:
            translated = _translate_rejection(exc)
            if translated is None:
                raise
            raise translated from exc
        fetch_duration_ms = (asyncio.get_running_loop().time() - started) * 1000
        try:
            body = response.json()
            content = str(body.get("content") or "")
            title = str(body.get("title") or "")
        except (AttributeError, TypeError, ValueError) as exc:
            raise ToolParseError(f"DeepSeek 抓取响应格式异常：{exc}") from exc
        if not content.strip():
            # 接口对无正文的页面可能仍回 200，判据不能只看状态码。
            raise SourceUnavailableError("empty_content", "DeepSeek 未返回可读取正文。")
        data = content.encode("utf-8")
        if len(data) > self._config.max_fetch_bytes:
            raise SourceUnavailableError(
                "content_too_large",
                f"来源正文 {len(data)} 字节，超过上限 {self._config.max_fetch_bytes} 字节。",
            )
        parse_started = asyncio.get_running_loop().time()
        try:
            parsed_title, text, blocks = await asyncio.wait_for(
                asyncio.to_thread(parse_markdown, data),
                timeout=parse_timeout or self._config.http_timeout,
            )
        except (TypeError, ValueError) as exc:
            raise ToolParseError(f"DeepSeek 抓取正文格式异常：{exc}") from exc
        except asyncio.TimeoutError as exc:
            raise ToolParseError("DeepSeek 抓取正文解析超时。") from exc
        parse_duration_ms = (asyncio.get_running_loop().time() - parse_started) * 1000
        return {
            "source_url": url,
            "final_url": url,
            "status": "completed",
            "name": PurePosixPath(url.split("?", 1)[0]).name or "document",
            "ext": "md",
            "content_type": "text/markdown",
            "modality": DocumentModality.TEXT.value,
            "title": title or parsed_title,
            "text": text,
            "blocks": blocks,
            "raw_bytes": len(data),
            "status_code": response.status_code,
            "content_hash": hashlib.sha256(data).hexdigest(),
            "retrieval_method": RETRIEVAL_DEEPSEEK_WEB_FETCH,
            "fetch_duration_ms": fetch_duration_ms,
            "parse_duration_ms": parse_duration_ms,
        }


def _translate_rejection(error: ToolRequestError) -> Exception | None:
    """按状态码区分账号级与来源级失败；返回 None 表示原样上抛，交给上层重试或降级。"""
    status = error.status_code or 0
    account = _ACCOUNT_LEVEL.get(status)
    if account is not None:
        user_code, message = account
        return ProviderExhaustedError(user_code, message)
    rejected = _SOURCE_REJECTED.get(status)
    if rejected is None:
        return None
    reason_code, message = rejected
    return SourceUnavailableError(reason_code, message)
