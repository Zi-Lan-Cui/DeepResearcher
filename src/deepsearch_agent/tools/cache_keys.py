"""跨工具统一的语义键规范化。"""

import hashlib
import json
import unicodedata
from typing import Any
from urllib.parse import urldefrag, urlsplit, urlunsplit


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).strip().casefold().split())


def canonical_url(value: str) -> str:
    value, _fragment = urldefrag(value.strip())
    parts = urlsplit(value)
    host = (parts.hostname or "").lower()
    if parts.port:
        host = f"{host}:{parts.port}"
    if parts.username or parts.password:
        # 身份凭证 URL 不应进入全局公开缓存；调用方据此 bypass。
        return ""
    return urlunsplit((parts.scheme.lower(), host, parts.path or "/", parts.query, ""))


def semantic_cache_key(*parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
