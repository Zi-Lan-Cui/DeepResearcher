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

from deepresearcher.tools.errors import UnsafeUrlError


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
        resolved = await self._resolve_addresses(ascii_hostname, port)
        if not resolved:
            raise UnsafeUrlError("URL 主机名没有可连接的地址。")
        # 双栈主机会同时返回 A 与 AAAA；某些公开站点的一条记录（或本机 DNS 的
        # 一条）可能落在非全局段。旧实现"任一地址非全局即整源拒绝"会误杀大量合法
        # 来源（量子网络评测里一个方向被拦 50+ 次，Writer 无源可写→吐 0 引用残卷）。
        # 正确策略：只要存在一个全局地址就放行，并把连接固定到全局地址集合；仅当
        # 全部地址都非全局（真 SSRF）才拒绝。
        addresses = tuple(a for a in resolved if self._is_global(a))
        if not addresses:
            raise UnsafeUrlError("出于安全原因，不能访问本机、私网或保留地址。")

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
    def _is_global(address: str) -> bool:
        try:
            return ipaddress.ip_address(address).is_global
        except ValueError:
            return False

    @staticmethod
    def ensure_public_ip(address: str) -> None:
        # 用于重定向逐跳校验：单个目标地址非全局即拒绝。
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
