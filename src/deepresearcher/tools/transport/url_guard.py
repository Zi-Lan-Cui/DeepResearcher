"""来源抓取使用的公网 URL 策略。

本模块在发出请求前解析主机名并返回 libcurl RESOLVE 条目。
调用方必须用这些条目建立实际连接；只查 DNS、再在 HTTP 栈内
重新解析，会留下 DNS rebinding 窗口。
"""

import asyncio
import ipaddress
import socket
from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import SplitResult, urlsplit, urlunsplit

from deepresearcher.tools.errors import UnsafeUrlError


@dataclass(frozen=True)
class ResolvedPublicUrl:
    """规范化后的公网 URL 与本次请求获准使用的地址集合。"""

    url: str
    hostname: str
    port: int
    addresses: tuple[str, ...]

    @property
    def curl_resolve(self) -> list[str]:
        # CURLOPT_RESOLVE 接受逗号分隔的地址。IPv6 字面量需要方括号，
        # 否则冒号会与 host/port 分隔符混淆。
        pinned = ",".join(f"[{item}]" if ":" in item else item for item in self.addresses)
        return [f"{self.hostname}:{self.port}:{pinned}"]


class PublicUrlGuard:
    """按拦截清单判定目标地址可连接性的 HTTP(S) 目标守卫。

    清单唯一来源是部署方的 SSRF_BLOCK_CIDRS 配置：不配置=不拦截
    （URL 语法与凭据检查仍在）。判定只有清单成员资格一条规则，
    代码里不留任何内置默认——配置里看到的清单就是实际执行的清单。
    """

    def __init__(self, block_cidrs: Sequence[str] = ()):
        self._block: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = tuple(
            ipaddress.ip_network(item.strip(), strict=False) for item in block_cidrs if item.strip()
        )

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
        # 双栈主机同时返回 A 与 AAAA，个别记录可能落在拦截段：存在可连接
        # 地址即放行并把连接固定到这些地址；全部被拦（真 SSRF）才拒绝。
        addresses = tuple(a for a in resolved if self._is_connectable(a))
        if not addresses:
            raise UnsafeUrlError("出于安全原因，不能访问本机、私网或保留地址。")

        # 把 URL 与 CURLOPT_RESOLVE 使用的主机名归一到同一 ASCII 写法。
        # 保留 path/query/fragment；fragment 不会随线上请求发送。
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

    def _is_connectable(self, address: str) -> bool:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            return False
        return not self._in_block_list(ip)

    def _in_block_list(self, ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
        return any(ip.version == network.version and ip in network for network in self._block)

    def ensure_public_ip(self, address: str) -> None:
        # 用于重定向逐跳校验：命中拦截清单即拒绝。
        try:
            ip = ipaddress.ip_address(address)
        except ValueError as exc:
            raise UnsafeUrlError("目标地址不是有效的 IP 地址。") from exc
        if self._in_block_list(ip):
            raise UnsafeUrlError("出于安全原因，不能访问本机、私网或保留地址。")

    @staticmethod
    def _parse(url: str) -> SplitResult:
        try:
            parsed = urlsplit(str(url).strip())
            # 访问 .port 同时完成校验，拒绝非法与越界端口。
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
