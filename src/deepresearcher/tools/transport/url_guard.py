"""来源抓取使用的公网 URL 策略。"""

import asyncio
import ipaddress
import socket
from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import SplitResult, urlsplit, urlunsplit

from deepresearcher.tools.errors import UnsafeUrlError

# IPv6 里承载 IPv4 的几种前缀：mapped、6to4、teredo 由 ipaddress 属性暴露，
# 剩下的三种只能按末 32 位取。
_CARRIED_IPV4_PREFIXES = (
    ipaddress.ip_network("::/96"),  # IPv4-compatible，已废弃
    ipaddress.ip_network("::ffff:0:0:0/96"),  # IPv4-translated，已废弃
    ipaddress.ip_network("64:ff9b::/96"),  # NAT64
)

# DNS 查询失败重试一次：本机代理的假 DNS 会间歇超时，单次失败不代表域名不可达。
_RESOLVE_ATTEMPTS = 2
_RESOLVE_RETRY_SECONDS = 0.3


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
    """按可达性判定目标地址的 HTTP(S) 目标守卫。

    缺省拒绝：只接受全球可路由的单播地址，私网、回环、链路本地、保留段与组播
    一律不可连接，不需要任何配置。SSRF_BLOCK_CIDRS 是在此之上追加的封堵清单。
    allow_private 供 fake-IP 代理这类把公网域名解析到私网段的环境关掉基线。
    """

    def __init__(self, block_cidrs: Sequence[str] = (), *, allow_private: bool = False):
        self._block: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = tuple(
            ipaddress.ip_network(item.strip(), strict=False) for item in block_cidrs if item.strip()
        )
        self._allow_private = allow_private

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
            raise UnsafeUrlError("URL 主机名没有可连接的地址。", "resolve_failed")
        # 双栈主机同时返回 A 与 AAAA，个别记录可能落在拦截段：存在可连接
        # 地址即允许通过并把连接固定到这些地址；全部被拦（真 SSRF）才拒绝。
        addresses = tuple(a for a in resolved if self._is_connectable(a))
        if not addresses:
            raise UnsafeUrlError("出于安全原因，不能访问本机、私网或保留地址。", "ssrf_block")

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
        """解析主机名；DNS 抖动时重试一次，耗尽后按 resolve_failed 抛出。"""
        last_error: socket.gaierror | None = None
        for attempt in range(_RESOLVE_ATTEMPTS):
            try:
                rows = await asyncio.get_running_loop().getaddrinfo(
                    hostname,
                    port,
                    family=socket.AF_UNSPEC,
                    type=socket.SOCK_STREAM,
                )
            except socket.gaierror as exc:
                last_error = exc
                if attempt + 1 < _RESOLVE_ATTEMPTS:
                    await asyncio.sleep(_RESOLVE_RETRY_SECONDS)
                continue
            return tuple(dict.fromkeys(str(row[4][0]) for row in rows))
        raise UnsafeUrlError(
            f"URL 主机名无法解析（DNS 查询失败：{last_error}）。", "resolve_failed"
        ) from last_error

    def _is_connectable(self, address: str) -> bool:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            return False
        return not self._is_blocked(ip)

    def _is_blocked(self, ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
        for candidate in (ip, *_carried_ipv4(ip)):
            # 基线：全球可路由的单播才算公网。is_global 对组播返回真，得单独排除。
            if not self._allow_private and not (candidate.is_global and not candidate.is_multicast):
                return True
            if any(candidate in net for net in self._block if candidate.version == net.version):
                return True
        return False

    def ensure_public_ip(self, address: str) -> None:
        # 用于重定向逐跳校验：命中可达性判据或追加清单即拒绝。
        try:
            ip = ipaddress.ip_address(address)
        except ValueError as exc:
            raise UnsafeUrlError("目标地址不是有效的 IP 地址。") from exc
        if self._is_blocked(ip):
            raise UnsafeUrlError("出于安全原因，不能访问本机、私网或保留地址。", "ssrf_block")

    @staticmethod
    def _parse(url: str) -> SplitResult:
        try:
            parsed = urlsplit(str(url).strip())
            # 访问 .port 同时完成校验，拒绝非法与越界端口；取值本身不用。
            _ = parsed.port
        except ValueError as exc:
            raise UnsafeUrlError("URL 格式无效。") from exc
        if parsed.scheme not in {"http", "https"}:
            raise UnsafeUrlError("只允许访问 HTTP 或 HTTPS 来源。")
        if not parsed.hostname:
            raise UnsafeUrlError("URL 缺少主机名。")
        if parsed.username is not None or parsed.password is not None:
            raise UnsafeUrlError("来源 URL 不允许包含用户名或密码。")
        return parsed


def _carried_ipv4(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> tuple[ipaddress.IPv4Address, ...]:
    """取 IPv6 地址里内嵌的 IPv4 目标，没有就返回空。"""
    if ip.version != 6:
        return ()
    assert isinstance(ip, ipaddress.IPv6Address)
    found: list[ipaddress.IPv4Address] = []
    if ip.ipv4_mapped is not None:
        found.append(ip.ipv4_mapped)
    if ip.sixtofour is not None:
        found.append(ip.sixtofour)
    if ip.teredo is not None:
        # teredo 的 server 与 client 两个端点都得判。
        found.extend(ip.teredo)
    if ip.ipv4_mapped is None and any(ip in net for net in _CARRIED_IPV4_PREFIXES):
        found.append(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF))
    return tuple(found)
