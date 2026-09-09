"""Public-web URL policy used by source fetching.

The guard resolves a hostname before the request and returns libcurl RESOLVE
entries.  The caller must use those entries for the actual connection; merely
checking DNS and resolving it again inside the HTTP stack leaves a DNS-rebinding
window.
"""

import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import SplitResult, urlsplit, urlunsplit

from deepsearch_agent.tools.errors import UnsafeUrlError


@dataclass(frozen=True)
class ResolvedPublicUrl:
    """A normalized public URL and the addresses approved for this request."""

    url: str
    hostname: str
    port: int
    addresses: tuple[str, ...]

    @property
    def curl_resolve(self) -> list[str]:
        # CURLOPT_RESOLVE accepts comma-separated addresses. IPv6 literals need
        # brackets so their colons are not confused with host/port separators.
        pinned = ",".join(f"[{item}]" if ":" in item else item for item in self.addresses)
        return [f"{self.hostname}:{self.port}:{pinned}"]


class PublicUrlGuard:
    """Resolve and admit only ordinary public HTTP(S) destinations."""

    async def resolve(self, url: str) -> ResolvedPublicUrl:
        parsed = self._parse(url)
        hostname = parsed.hostname
        assert hostname is not None
        try:
            ascii_hostname = hostname.encode("idna").decode("ascii")
        except UnicodeError as exc:
            raise UnsafeUrlError("URL 主机名不是有效的 IDNA 名称。") from exc

        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        addresses = await self._resolve_addresses(ascii_hostname, port)
        if not addresses:
            raise UnsafeUrlError("URL 主机名没有可连接的地址。")
        for address in addresses:
            self.ensure_public_ip(address)

        # Normalize the host used by URL and CURLOPT_RESOLVE to the same ASCII
        # spelling. Preserve path/query/fragment; fragments are not sent on wire.
        host_for_url = f"[{ascii_hostname}]" if ":" in ascii_hostname else ascii_hostname
        if parsed.port is not None:
            host_for_url = f"{host_for_url}:{parsed.port}"
        normalized = urlunsplit(
            SplitResult(
                parsed.scheme, host_for_url, parsed.path or "/", parsed.query, parsed.fragment
            )
        )
        return ResolvedPublicUrl(normalized, ascii_hostname, port, addresses)

    async def _resolve_addresses(self, hostname: str, port: int) -> tuple[str, ...]:
        try:
            rows = await asyncio.get_running_loop().getaddrinfo(
                hostname,
                port,
                family=socket.AF_UNSPEC,
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            raise UnsafeUrlError("URL 主机名无法解析。") from exc
        return tuple(dict.fromkeys(str(row[4][0]) for row in rows))

    @staticmethod
    def ensure_public_ip(address: str) -> None:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError as exc:
            raise UnsafeUrlError("目标地址不是有效的 IP 地址。") from exc
        if not ip.is_global:
            raise UnsafeUrlError("出于安全原因，不能访问本机、私网或保留地址。")

    @staticmethod
    def _parse(url: str) -> SplitResult:
        try:
            parsed = urlsplit(str(url).strip())
            # Accessing port performs its validation (invalid/out-of-range ports).
            parsed.port
        except ValueError as exc:
            raise UnsafeUrlError("URL 格式无效。") from exc
        if parsed.scheme not in {"http", "https"}:
            raise UnsafeUrlError("只允许访问 HTTP 或 HTTPS 来源。")
        if not parsed.hostname:
            raise UnsafeUrlError("URL 缺少主机名。")
        if parsed.username is not None or parsed.password is not None:
            raise UnsafeUrlError("来源 URL 不允许包含用户名或密码。")
        return parsed
