"""跨工具统一的语义键规范化。"""

import hashlib
import json
import unicodedata
from typing import Any
from urllib.parse import urldefrag, urlsplit, urlunsplit


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).strip().casefold().split())


def canonical_url(value: str, *, strip_trailing_slash: bool = False) -> str:
    """URL 归一的唯一实现。返回空串表示该 URL 含凭证，禁止进入共享缓存与去重身份。

    strip_trailing_slash 只在搜索去重时折叠 /page 与 /page/，让两者算作同一条；
    缓存键保持缺省。含凭证时返回空串并跳过缓存的语义，两侧调用方共用。
    """
    value, _fragment = urldefrag(value.strip())
    parts = urlsplit(value)
    host = (parts.hostname or "").lower()
    if parts.port:
        host = f"{host}:{parts.port}"
    if parts.username or parts.password:
        # 身份凭证 URL 不应进入全局公开缓存；调用方据此 bypass。
        return ""
    path = parts.path or "/"
    if strip_trailing_slash:
        path = path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), host, path, parts.query, ""))


def semantic_cache_key(*parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
