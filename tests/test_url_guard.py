import asyncio

import httpx
import pytest

from deepresearcher.config import SearchConfig
from deepresearcher.tools.errors import UnsafeUrlError
from deepresearcher.tools.transport import HttpClient, PublicUrlGuard, ResolvedPublicUrl

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


def test_unconfigured_guard_blocks_no_addresses(monkeypatch):
    """不配置 = 不拦截是明确语义：10.0.0.1 在无清单守卫下可通过（URL 检查仍在）。"""
    guard = PublicUrlGuard()

    async def resolve(_hostname: str, _port: int) -> tuple[str, ...]:
        return ("10.0.0.1",)

    monkeypatch.setattr(guard, "_resolve_addresses", resolve)
    result = asyncio.run(guard.resolve("https://internal.example/resource"))
    assert result.addresses == ("10.0.0.1",)


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


def test_edited_block_list_admits_fake_ip_proxy_ranges(monkeypatch):
    """清单删一行即放行该段：fake-IP 代理环境删 198.18.0.0/15，其余照拦。"""
    relaxed = [item for item in BLOCK if item != "198.18.0.0/15"]
    guard = PublicUrlGuard(relaxed)

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
        SearchConfig(retry_attempts=1),
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
        SearchConfig(retry_attempts=2, retry_initial_seconds=0.0, retry_max_seconds=0.0),
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
        SearchConfig(retry_attempts=1),
        ReboundClient(),  # type: ignore[arg-type]
        url_guard=Guard(),  # type: ignore[arg-type]
    )
    with pytest.raises(Exception, match="DNS 地址不一致"):
        asyncio.run(client.arequest("GET", "https://public.example/", request_kind="fetch"))
