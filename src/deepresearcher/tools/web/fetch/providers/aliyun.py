"""阿里云 DTS AI WebFetch Provider。"""

import asyncio
import hashlib
from pathlib import PurePosixPath

from deepresearcher.config import SearchConfig
from deepresearcher.tools.errors import SourceUnavailableError, ToolParseError, ToolRequestError
from deepresearcher.tools.web.aliyun import AliyunDtsApi
from deepresearcher.tools.web.fetch.models import SourceDocument
from deepresearcher.tools.web.parsing import DocumentModality, parse_html_blocks, parse_markdown

_CHALLENGE_TITLE_MARKERS = (
    "请求已被拦截",
    "安全验证",
    "访问验证",
    "验证码",
    "just a moment",
    "security check",
    "access denied",
)
_CHALLENGE_BODY_MARKERS = (
    "请求已被站点的安全策略拦截",
    "本次访问已被限制",
    "由 tencent cloud edgeone 提供防护",
    "verify you are human",
    "checking your browser before accessing",
)


class AliyunFetchProvider:
    name = "aliyun"

    def __init__(self, config: SearchConfig, client: AliyunDtsApi):
        self._config = config
        self._client = client

    async def afetch(
        self,
        url: str,
        *,
        fetch_timeout: float | None = None,
        parse_timeout: float | None = None,
    ) -> SourceDocument:
        started = asyncio.get_running_loop().time()
        try:
            body = await asyncio.wait_for(
                self._client.web_fetch(url, self._config.aliyun_fetch_output_format),
                timeout=fetch_timeout or self._config.timeout,
            )
        except asyncio.TimeoutError as exc:
            raise ToolRequestError("阿里云 WebFetch 请求超时。") from exc
        fetch_duration_ms = (asyncio.get_running_loop().time() - started) * 1000
        if not body or not bool(getattr(body, "success", False)):
            message = getattr(body, "error_message", "") if body else "空响应"
            raise ToolRequestError(f"阿里云 WebFetch 返回失败：{message or '未知错误'}")

        content = str(getattr(body, "content", "") or "")
        if not content.strip():
            raise ToolRequestError("阿里云 WebFetch 未返回可读取正文。", retryable=False)
        title = str(getattr(body, "title", "") or "")
        self._ensure_readable(title, content)
        content_format = str(getattr(body, "content_format", "") or "").lower()
        data = content.encode("utf-8")
        parse_started = asyncio.get_running_loop().time()
        try:
            parser = parse_html_blocks if content_format == "html" else parse_markdown
            parsed_title, text, blocks = await asyncio.wait_for(
                asyncio.to_thread(parser, data),
                timeout=parse_timeout or self._config.timeout,
            )
        except (TypeError, ValueError) as exc:
            raise ToolParseError(f"阿里云 WebFetch 正文格式异常：{exc}") from exc
        except asyncio.TimeoutError as exc:
            raise ToolParseError("阿里云 WebFetch 正文解析超时。") from exc
        parse_duration_ms = (asyncio.get_running_loop().time() - parse_started) * 1000
        final_url = str(getattr(body, "url", "") or url)
        title = title or parsed_title
        return {
            "source_url": url,
            "final_url": final_url,
            "status": "completed",
            "name": PurePosixPath(final_url.split("?", 1)[0]).name or "document",
            "ext": "html" if content_format == "html" else "md",
            "content_type": "text/html" if content_format == "html" else "text/markdown",
            "modality": DocumentModality.TEXT.value,
            "title": title,
            "text": text,
            "blocks": blocks,
            "raw_bytes": len(data),
            "status_code": int(getattr(body, "http_status_code", 200) or 200),
            "content_hash": hashlib.sha256(data).hexdigest(),
            "retrieval_method": "aliyun_web_fetch",
            "provider_request_id": str(getattr(body, "request_id", "") or ""),
            "url_type": str(getattr(body, "url_type", "") or ""),
            "fetch_duration_ms": fetch_duration_ms,
            "parse_duration_ms": parse_duration_ms,
        }

    @staticmethod
    def _ensure_readable(title: str, content: str) -> None:
        """拦截云端抓取成功但正文实为安全挑战页的假阳性。"""
        normalized_title = title.casefold().strip()
        normalized_content = content.casefold()
        if any(marker in normalized_title for marker in _CHALLENGE_TITLE_MARKERS) or any(
            marker in normalized_content for marker in _CHALLENGE_BODY_MARKERS
        ):
            raise SourceUnavailableError(
                "access_challenge", "阿里云 WebFetch 返回了站点安全验证页，未取得可验证正文。"
            )
