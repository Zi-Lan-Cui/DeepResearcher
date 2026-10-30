import asyncio
import socket

import httpx
import pytest
from curl_cffi import CurlOpt

from deepresearcher.config import SearchConfig
from deepresearcher.tools.errors import UnsafeUrlError
from deepresearcher.tools.transport import HttpClient, PublicUrlGuard, ResolvedPublicUrl, url_guard

# 拦截清单唯一来源是配置；测试自带一份标准清单，验证清单语义本身。
BLOCK = (
    "0.0.0.0/8",
    "10.0.0.0/8",
    "100.64.0.0/10",
    "127.0.0.0/8",
    "169.254.0.0/16",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "198.18.0.0/15",
    "::1/128",
    "fc00::/7",
    "fe80::/10",
)


def test_public_url_guard_accepts_public_dns_and_builds_curl_pin(monkeypatch):
    guard = PublicUrlGuard(BLOCK)

    async def resolve(_hostname: str, _port: int) -> tuple[str, ...]:
        return ("93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946")

    monkeypatch.setattr(guard, "_resolve_addresses", resolve)
    result = asyncio.run(guard.resolve("https://例子.测试:8443/a?q=1"))

    assert result.hostname == "xn--fsqu00a.xn--0zwm56d"
    assert result.port == 8443
    assert result.url == "https://xn--fsqu00a.xn--0zwm56d:8443/a?q=1"
    assert result.curl_resolve == [
        "xn--fsqu00a.xn--0zwm56d:8443:93.184.216.34,[2606:2800:220:1:248:1893:25c8:1946]"
    ]


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/file",
        "https://user:secret@example.com/",
        "https:///missing-host",
        "https://example.com:99999/",
    ],
)
def test_public_url_guard_rejects_invalid_or_credentialed_urls(url):
    # 地址清单为空也不放行畸形 URL：语法与凭据检查独立于拦截清单。
    with pytest.raises(UnsafeUrlError):
        asyncio.run(PublicUrlGuard().resolve(url))


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "169.254.169.254",
        "172.16.0.1",
        "192.168.1.1",
        "198.18.0.28",
        "::1",
        "fc00::1",
        "fe80::1",
    ],
)
def test_public_url_guard_rejects_address_in_block_list(monkeypatch, address):
    guard = PublicUrlGuard(BLOCK)

    async def resolve(_hostname: str, _port: int) -> tuple[str, ...]:
        return (address,)

    monkeypatch.setattr(guard, "_resolve_addresses", resolve)
    with pytest.raises(UnsafeUrlError, match="私网"):
        asyncio.run(guard.resolve("https://attacker.example/resource"))


def test_unconfigured_guard_still_blocks_private_by_baseline(monkeypatch):
    """清单为空也有基线：私网照样拦，公网照样过，不需要任何配置。"""
    guard = PublicUrlGuard()

    async def private(_hostname: str, _port: int) -> tuple[str, ...]:
        return ("10.0.0.1",)

    monkeypatch.setattr(guard, "_resolve_addresses", private)
    with pytest.raises(UnsafeUrlError, match="私网"):
        asyncio.run(guard.resolve("https://internal.example/resource"))

    async def public(_hostname: str, _port: int) -> tuple[str, ...]:
        return ("93.184.216.34",)

    monkeypatch.setattr(guard, "_resolve_addresses", public)
    result = asyncio.run(guard.resolve("https://public.example/resource"))
    assert result.addresses == ("93.184.216.34",)


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "169.254.169.254",
        "100.64.0.1",
        "192.0.2.1",
        "198.18.0.28",
        "240.0.0.1",
        "224.0.0.1",
        "::1",
        "fe80::1",
        "fc00::1",
        "2001:db8::1",
        "ff02::1",
    ],
)
def test_baseline_rejects_non_public_addresses_without_configuration(monkeypatch, address):
    """基线覆盖私网、保留、链路本地与组播：清单没列也要拦。"""
    guard = PublicUrlGuard()

    async def resolve(_hostname: str, _port: int) -> tuple[str, ...]:
        return (address,)

    monkeypatch.setattr(guard, "_resolve_addresses", resolve)
    with pytest.raises(UnsafeUrlError, match="私网"):
        asyncio.run(guard.resolve("https://attacker.example/resource"))


@pytest.mark.parametrize(
    "address",
    [
        "::ffff:169.254.169.254",
        "::ffff:127.0.0.1",
        "::ffff:0:169.254.169.254",
        "::169.254.169.254",
        "::127.0.0.1",
        "2002:a9fe:a9fe::1",
        "64:ff9b::a9fe:a9fe",
        "2001:0:4136:e378:8000:63bf:3fff:fdd2",
    ],
)
def test_guard_unwraps_ipv4_carried_inside_ipv6(monkeypatch, address):
    """IPv6 字面量里内嵌的 IPv4 目标按 IPv4 判据判，不因地址族不同而放行。"""
    guard = PublicUrlGuard()

    async def resolve(_hostname: str, _port: int) -> tuple[str, ...]:
        return (address,)

    monkeypatch.setattr(guard, "_resolve_addresses", resolve)
    with pytest.raises(UnsafeUrlError, match="私网"):
        asyncio.run(guard.resolve("https://attacker.example/resource"))


@pytest.mark.parametrize("address", ["::ffff:8.8.8.8", "64:ff9b::808:808", "2001:4860::1"])
def test_guard_admits_public_ipv4_carried_inside_ipv6(monkeypatch, address):
    """内嵌的是公网 IPv4 时照常放行：展开只影响判据，不误伤合法地址。"""
    guard = PublicUrlGuard()

    async def resolve(_hostname: str, _port: int) -> tuple[str, ...]:
        return (address,)

    monkeypatch.setattr(guard, "_resolve_addresses", resolve)
    result = asyncio.run(guard.resolve("https://public.example/resource"))
    assert result.addresses == (address,)


def test_public_url_guard_admits_dual_stack_with_one_allowed_record(monkeypatch):
    """双栈主机同时返回被拦与放行记录时,只筛掉被拦地址并仅钉放行地址,
    不得整源拒绝。"""
    guard = PublicUrlGuard(BLOCK)

    async def resolve(_hostname: str, _port: int) -> tuple[str, ...]:
        return ("127.0.0.1", "93.184.216.34")

    monkeypatch.setattr(guard, "_resolve_addresses", resolve)
    result = asyncio.run(guard.resolve("https://dual-stack.example/paper"))

    assert result.addresses == ("93.184.216.34",)
    assert result.curl_resolve == ["dual-stack.example:443:93.184.216.34"]


def test_allow_private_admits_fake_ip_proxy_ranges(monkeypatch):
    """fake-IP 代理环境：关掉基线后，删掉清单里的 198.18.0.0/15 即可放行。"""
    relaxed = [item for item in BLOCK if item != "198.18.0.0/15"]
    guard = PublicUrlGuard(relaxed, allow_private=True)

    async def resolve(_hostname: str, _port: int) -> tuple[str, ...]:
        return ("198.18.0.28",)

    monkeypatch.setattr(guard, "_resolve_addresses", resolve)
    result = asyncio.run(guard.resolve("https://en.wikipedia.org/wiki/X"))
    assert result.addresses == ("198.18.0.28",)
    guard.ensure_public_ip("198.18.0.99")
    with pytest.raises(UnsafeUrlError):
        guard.ensure_public_ip("169.254.169.254")  # 未删的条目依旧拦


def test_invalid_block_cidr_fails_fast():
    with pytest.raises(ValueError):
        PublicUrlGuard(["not-a-cidr"])


def test_cidr_list_env_accepts_multiline_and_comma_forms(monkeypatch):
    from deepresearcher.config import _cidr_list_env

    monkeypatch.setenv("SSRF_BLOCK_CIDRS", "\n10.0.0.0/8\n127.0.0.0/8\n")
    assert _cidr_list_env("SSRF_BLOCK_CIDRS") == ("10.0.0.0/8", "127.0.0.0/8")
    monkeypatch.setenv("SSRF_BLOCK_CIDRS", "10.0.0.0/8, 127.0.0.0/8 ")
    assert _cidr_list_env("SSRF_BLOCK_CIDRS") == ("10.0.0.0/8", "127.0.0.0/8")
    monkeypatch.delenv("SSRF_BLOCK_CIDRS")
    assert _cidr_list_env("SSRF_BLOCK_CIDRS") == ()


def test_http_client_validates_every_redirect_before_following():
    class Guard:
        def __init__(self):
            self.urls: list[str] = []

        async def resolve(self, url: str) -> ResolvedPublicUrl:
            self.urls.append(url)
            if "127.0.0.1" in url:
                raise UnsafeUrlError("blocked redirect")
            return ResolvedPublicUrl(url, "public.example", 443, ("93.184.216.34",))

        @staticmethod
        def ensure_public_ip(_address: str) -> None:
            return None

    class RedirectingClient:
        def __init__(self):
            self.calls = 0

        async def request(self, method, url, **kwargs):
            self.calls += 1
            assert kwargs["allow_redirects"] is False
            return httpx.Response(
                302,
                headers={"Location": "http://127.0.0.1/admin"},
                request=httpx.Request(method, url),
            )

    raw = RedirectingClient()
    guard = Guard()
    client = HttpClient(
        SearchConfig(max_attempts=1),
        raw,  # type: ignore[arg-type]
        url_guard=guard,  # type: ignore[arg-type]
    )

    with pytest.raises(UnsafeUrlError, match="blocked redirect"):
        asyncio.run(client.arequest("GET", "https://public.example/start", request_kind="fetch"))

    assert raw.calls == 1
    assert guard.urls == ["https://public.example/start", "http://127.0.0.1/admin"]


def test_http_client_reuses_one_impersonate_profile_for_search_requests():
    class RecordingClient:
        def __init__(self):
            self.impersonates: list[str] = []

        async def request(self, method, url, **kwargs):
            self.impersonates.append(kwargs["impersonate"])
            if len(self.impersonates) == 1:
                return httpx.Response(429, request=httpx.Request(method, url))
            return httpx.Response(200, request=httpx.Request(method, url))

    raw = RecordingClient()
    client = HttpClient(
        SearchConfig(max_attempts=2, retry_initial_seconds=0.0, retry_max_seconds=0.0),
        raw,  # type: ignore[arg-type]
    )
    asyncio.run(client.arequest("GET", "https://api.example/search", request_kind="search"))

    # 共享连接池上重试必须沿用同一指纹，不跨 TLS 上下文复用同一条连接。
    assert raw.impersonates == [client._pinned_impersonate, client._pinned_impersonate]


def test_http_client_rejects_peer_outside_pinned_dns_set():
    class Guard:
        async def resolve(self, url: str) -> ResolvedPublicUrl:
            return ResolvedPublicUrl(url, "public.example", 443, ("93.184.216.34",))

        @staticmethod
        def ensure_public_ip(_address: str) -> None:
            return None

    class ReboundClient:
        async def request(self, method, url, **_kwargs):
            response = httpx.Response(200, request=httpx.Request(method, url))
            response.primary_ip = "93.184.216.35"
            return response

    client = HttpClient(
        SearchConfig(max_attempts=1),
        ReboundClient(),  # type: ignore[arg-type]
        url_guard=Guard(),  # type: ignore[arg-type]
    )
    with pytest.raises(Exception, match="DNS 地址不一致"):
        asyncio.run(client.arequest("GET", "https://public.example/", request_kind="fetch"))


def test_fetch_curl_options_disable_ambient_proxy_and_keep_dns_pin():
    client = HttpClient(SearchConfig(max_attempts=1))
    resolved = ResolvedPublicUrl(
        "https://public.example/", "public.example", 443, ("93.184.216.34",)
    )

    direct = client._fetch_curl_options(resolved, "")
    # 空串才是禁用：不设置 CURLOPT_PROXY 时 libcurl 自己会读 HTTPS_PROXY。
    assert direct[CurlOpt.PROXY] == ""
    assert direct[CurlOpt.RESOLVE] == ["public.example:443:93.184.216.34"]

    proxied = client._fetch_curl_options(resolved, "http://127.0.0.1:7890")
    assert proxied[CurlOpt.PROXY] == "http://127.0.0.1:7890"
    # 代理按主机名自行解析，钉住地址只会让 URL 与实际连接脱节。
    assert CurlOpt.RESOLVE not in proxied


def test_http_client_still_blocks_loopback_peer_when_no_proxy_configured():
    class Guard:
        async def resolve(self, url: str) -> ResolvedPublicUrl:
            return ResolvedPublicUrl(url, "public.example", 443, ("93.184.216.34",))

        @staticmethod
        def ensure_public_ip(address: str) -> None:
            raise UnsafeUrlError(f"拒绝 {address}")

    class LoopbackPeerClient:
        async def request(self, method, url, **_kwargs):
            response = httpx.Response(200, request=httpx.Request(method, url))
            response.primary_ip = "127.0.0.1"
            return response

    client = HttpClient(
        SearchConfig(max_attempts=1),
        LoopbackPeerClient(),  # type: ignore[arg-type]
        url_guard=Guard(),  # type: ignore[arg-type]
    )
    with pytest.raises(UnsafeUrlError, match="127.0.0.1"):
        asyncio.run(client.arequest("GET", "https://public.example/", request_kind="fetch"))


def test_http_client_skips_peer_pin_when_proxy_is_configured():
    class Guard:
        async def resolve(self, url: str) -> ResolvedPublicUrl:
            return ResolvedPublicUrl(url, "public.example", 443, ("93.184.216.34",))

        @staticmethod
        def ensure_public_ip(address: str) -> None:
            raise AssertionError(f"走代理时对端必然是代理地址，不应校验，实际校验了 {address}")

    class ProxiedClient:
        async def request(self, method, url, **_kwargs):
            response = httpx.Response(200, request=httpx.Request(method, url))
            response.primary_ip = "127.0.0.1"
            return response

    client = HttpClient(
        SearchConfig(max_attempts=1, fetch_proxy="http://127.0.0.1:7890"),
        ProxiedClient(),  # type: ignore[arg-type]
        url_guard=Guard(),  # type: ignore[arg-type]
    )
    response = asyncio.run(client.arequest("GET", "https://public.example/", request_kind="fetch"))
    assert response.status_code == 200


def test_guard_retries_dns_once_before_declaring_resolve_failed(monkeypatch):
    calls: list[str] = []

    async def fail(host: str, _port: int) -> tuple[str, ...]:
        calls.append(host)
        raise socket.gaierror(11001, "getaddrinfo failed")

    # 替换单次解析而非 socket.getaddrinfo：uvloop 经 libuv 解析，打桩 socket 层无效。
    monkeypatch.setattr(url_guard, "_resolve_once", fail)
    monkeypatch.setattr(url_guard, "_RESOLVE_RETRY_SECONDS", 0)
    guard = PublicUrlGuard(BLOCK)

    with pytest.raises(UnsafeUrlError) as excinfo:
        asyncio.run(guard.resolve("https://flaky.example/"))

    assert len(calls) == 2  # 抖动重试一次，第二次仍失败才判不可达
    assert excinfo.value.reason_code == "resolve_failed"


def test_guard_recovers_when_second_dns_attempt_succeeds(monkeypatch):
    calls: list[str] = []

    async def flaky(host: str, _port: int) -> tuple[str, ...]:
        calls.append(host)
        if len(calls) == 1:
            raise socket.gaierror(11001, "getaddrinfo failed")
        return ("93.184.216.34",)

    monkeypatch.setattr(url_guard, "_resolve_once", flaky)
    monkeypatch.setattr(url_guard, "_RESOLVE_RETRY_SECONDS", 0)
    guard = PublicUrlGuard(BLOCK)

    result = asyncio.run(guard.resolve("https://flaky.example/"))

    assert len(calls) == 2
    assert result.addresses == ("93.184.216.34",)


def test_guard_marks_real_block_as_ssrf_not_resolve_failure(monkeypatch):
    guard = PublicUrlGuard(BLOCK)

    async def resolve(_hostname: str, _port: int) -> tuple[str, ...]:
        return ("127.0.0.1",)

    monkeypatch.setattr(guard, "_resolve_addresses", resolve)

    with pytest.raises(UnsafeUrlError) as excinfo:
        asyncio.run(guard.resolve("https://internal.example/"))

    assert excinfo.value.reason_code == "ssrf_block"
