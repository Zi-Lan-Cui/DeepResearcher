"""异步 Fetcher：下载内容并将阻塞解析委托给线程。"""

import asyncio
import hashlib
from pathlib import PurePosixPath
from typing import cast

from bs4 import BeautifulSoup

from deepsearch_agent.config import SearchConfig
from deepsearch_agent.parsers import (
    DocumentBlock,
    DocumentModality,
    parse_docx,
    parse_html_blocks,
    parse_pdf,
    parse_text,
)
from deepsearch_agent.tools.cache import CacheValue, NoOpToolCache, ToolCache
from deepsearch_agent.tools.cache_keys import canonical_url, semantic_cache_key
from deepsearch_agent.tools.errors import SourceUnavailableError
from deepsearch_agent.tools.sources.models import SourceDocument
from deepsearch_agent.tools.transport.http_client import HttpClient

_CHALLENGE_TITLE_MARKERS = ("验证码", "安全验证", "访问验证", "just a moment", "security check")
_LOGIN_TITLE_MARKERS = ("登录", "sign in", "log in")
_CHALLENGE_BODY_MARKERS = (
    "captcha",
    "geetest",
    "recaptcha",
    "challenge",
    "security verification",
    "访问验证",
)


class WebFetcher:
    def __init__(
        self,
        config: SearchConfig,
        http_client: HttpClient | None = None,
        *,
        tool_cache: ToolCache | None = None,
        cache_ttl_seconds: int = 0,
        fetch_policy_version: str = "public-fetch-v1",
        parser_version: str = "parser-v1",
    ):
        self.timeout = config.timeout
        self.http = http_client or HttpClient(config)
        self.tool_cache = tool_cache or NoOpToolCache()
        self.cache_ttl_seconds = cache_ttl_seconds
        self.fetch_policy_version = fetch_policy_version
        self.parser_version = parser_version

    async def afetch(
        self,
        url: str,
        *,
        fetch_timeout: float | None = None,
        parse_timeout: float | None = None,
    ) -> SourceDocument:
        normalized_url = canonical_url(url)
        if not normalized_url or not normalized_url.startswith(("http://", "https://")):
            return await self._afetch_uncached(
                url, fetch_timeout=fetch_timeout, parse_timeout=parse_timeout
            )

        async def compute() -> CacheValue:
            document = await self._afetch_uncached(
                url, fetch_timeout=fetch_timeout, parse_timeout=parse_timeout
            )
            return CacheValue(
                value=dict(document),
                content_hash=document.get("content_hash"),
                metrics={"saved_external_requests": 1},
                cacheable=document.get("status") == "completed" and not document.get("error"),
            )

        cached = await self.tool_cache.get_or_compute(
            "fetch",
            semantic_cache_key(normalized_url, self.fetch_policy_version, self.parser_version),
            ttl_seconds=self.cache_ttl_seconds,
            schema_version=self.parser_version,
            compute=compute,
        )
        document = cast(SourceDocument, dict(cached.value))
        document["source_url"] = url
        document["cache_hit"] = cached.hit
        if cached.hit:
            document["fetch_duration_ms"] = 0
            document["parse_duration_ms"] = 0
        return document

    async def _afetch_uncached(
        self,
        url: str,
        *,
        fetch_timeout: float | None = None,
        parse_timeout: float | None = None,
    ) -> SourceDocument:
        try:
            fetch_started = asyncio.get_running_loop().time()
            try:
                response = await asyncio.wait_for(
                    self.http.arequest(
                        "GET",
                        url,
                        timeout=fetch_timeout or self.timeout,
                        request_kind="fetch",
                    ),
                    timeout=fetch_timeout or self.timeout,
                )
            except asyncio.TimeoutError as exc:
                raise TimeoutError("source_fetch_timeout") from exc
            fetch_duration_ms = (asyncio.get_running_loop().time() - fetch_started) * 1000
            data = response.content
            content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
            ext = self._extension(str(response.url), content_type)
            try:
                parse_started = asyncio.get_running_loop().time()
                title, text, blocks = await asyncio.wait_for(
                    asyncio.to_thread(self._parse, data, content_type, ext),
                    timeout=parse_timeout or self.timeout,
                )
                parse_duration_ms = (asyncio.get_running_loop().time() - parse_started) * 1000
            except asyncio.TimeoutError as exc:
                raise TimeoutError("source_parse_timeout") from exc
            self._ensure_readable(title, text, data, content_type)
            return {
                "source_url": url,
                "final_url": str(response.url),
                "status": "completed",
                "name": PurePosixPath(str(response.url).split("?", 1)[0]).name or "document",
                "ext": ext,
                "content_type": content_type,
                "modality": self._modality(content_type, ext).value,
                "title": title,
                "text": text,
                "blocks": blocks,
                "raw_bytes": len(data),
                "status_code": response.status_code,
                "content_hash": hashlib.sha256(data).hexdigest(),
                "fetch_duration_ms": fetch_duration_ms,
                "parse_duration_ms": parse_duration_ms,
            }
        except (ValueError, KeyError) as exc:
            return {
                "source_url": url,
                "final_url": url,
                "name": PurePosixPath(url.split("?", 1)[0]).name or "document",
                "ext": "",
                "content_type": "",
                "modality": DocumentModality.BINARY.value,
                "title": "",
                "text": "",
                "blocks": [],
                "raw_bytes": 0,
                "status_code": 0,
                "status": "failed",
                "error_code": "fetch_parse_error",
                "error": str(exc)[:500],
            }

    @staticmethod
    def _ensure_readable(title: str, text: str, data: bytes, content_type: str) -> None:
        """在正文进入 LLM 前拦截验证码页、登录墙与动态空壳。"""
        is_html = content_type in {"text/html", "application/xhtml+xml"}
        if not is_html:
            if not text.strip():
                raise SourceUnavailableError(
                    "empty_content", "文档未解析出文本，可能是扫描件或不支持的格式。"
                )
            return

        normalized_title = title.casefold().strip()
        visible_html_text = BeautifulSoup(data, "html.parser").get_text(" ", strip=True).casefold()
        if any(marker in normalized_title for marker in _CHALLENGE_TITLE_MARKERS):
            raise SourceUnavailableError(
                "access_challenge", "来源返回验证码或安全验证页，无法取得可验证正文。"
            )
        if (
            any(marker in normalized_title for marker in _LOGIN_TITLE_MARKERS)
            and len(text.strip()) < 300
        ):
            raise SourceUnavailableError("login_required", "来源要求登录后才能读取正文。")
        if not text.strip():
            if any(marker in visible_html_text for marker in _CHALLENGE_BODY_MARKERS):
                raise SourceUnavailableError(
                    "access_challenge", "来源返回验证码或安全验证页，无法取得可验证正文。"
                )
            raise SourceUnavailableError(
                "empty_content", "来源未提供可读取正文，可能是动态渲染页面或空页面。"
            )

    def _parse(
        self, data: bytes, content_type: str, ext: str
    ) -> tuple[str, str, list[DocumentBlock]]:
        if content_type in {"text/html", "application/xhtml+xml"} or ext in {"html", "htm"}:
            return parse_html_blocks(data)
        if content_type == "application/pdf" or ext == "pdf":
            title, text = parse_pdf(data)
            return title, text, self._text_blocks(text)
        if (
            content_type
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            or ext == "docx"
        ):
            title, text = parse_docx(data)
            return title, text, self._text_blocks(text)
        if content_type.startswith("text/") or ext in {"txt", "md", "csv", "json", "xml"}:
            title, text = parse_text(data)
            return title, text, self._text_blocks(text)
        raise ValueError(f"暂不支持解析的文档类型：{content_type or ext}")

    @staticmethod
    def _text_blocks(text: str) -> list[DocumentBlock]:
        return [
            {
                "block_id": f"b-{index:04d}",
                "block_type": "paragraph",
                "text": line.strip(),
                "heading_path": [],
                "order": index,
            }
            for index, line in enumerate(text.splitlines())
            if line.strip()
        ]

    @staticmethod
    def _extension(url: str, content_type: str) -> str:
        suffix = PurePosixPath(url.split("?", 1)[0]).suffix.lower().lstrip(".")
        return suffix or {"text/html": "html", "application/pdf": "pdf", "text/plain": "txt"}.get(
            content_type, ""
        )

    @staticmethod
    def _modality(content_type: str, ext: str) -> DocumentModality:
        if content_type.startswith("text/") or ext in {
            "html",
            "htm",
            "txt",
            "md",
            "csv",
            "json",
            "xml",
        }:
            return DocumentModality.TEXT
        if ext in {"pdf", "docx"} or content_type in {
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        }:
            return DocumentModality.RICH
        return DocumentModality.BINARY
