import asyncio

import httpx
import pytest

from deepresearcher.config import SearchConfig
from deepresearcher.tools.errors import UnsafeUrlError
from deepresearcher.tools.transport import HttpClient, PublicUrlGuard, ResolvedPublicUrl


def test_public_url_guard_accepts_public_dns_and_builds_curl_pin(monkeypatch):
    guard = PublicUrlGuard()

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
        "::1",
        "fc00::1",
        "fe80::1",
    ],
)
def test_public_url_guard_rejects_non_public_dns_answers(monkeypatch, address):
    guard = PublicUrlGuard()

    async def resolve(_hostname: str, _port: int) -> tuple[str, ...]:
        return (address,)

    monkeypatch.setattr(guard, "_resolve_addresses", resolve)
    with pytest.raises(UnsafeUrlError, match="私网"):
        asyncio.run(guard.resolve("https://attacker.example/resource"))


def test_public_url_guard_admits_dual_stack_with_one_global_record(monkeypatch):
    """双栈主机同时返回一条全局 + 一条非全局记录时，旧实现整源拒绝，误杀大量
    合法来源（评测里一个研究方向被拦 50+ 次）。修正后放行并只钉全局地址。"""
    guard = PublicUrlGuard()

    async def resolve(_hostname: str, _port: int) -> tuple[str, ...]:
        return ("127.0.0.1", "93.184.216.34")

    monkeypatch.setattr(guard, "_resolve_addresses", resolve)
    result = asyncio.run(guard.resolve("https://dual-stack.example/paper"))

    assert result.addresses == ("93.184.216.34",)
    assert result.curl_resolve == ["dual-stack.example:443:93.184.216.34"]


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
