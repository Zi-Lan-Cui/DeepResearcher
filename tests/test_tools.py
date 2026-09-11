import asyncio
import json

import httpx
import pytest
from curl_cffi.requests.exceptions import Timeout

from deepsearch_agent.config import SearchConfig
from deepsearch_agent.tools.errors import (
    SourceUnavailableError,
    ToolConfigurationError,
    ToolParseError,
    ToolRequestError,
)
from deepsearch_agent.tools.search import SearchClient
from deepsearch_agent.tools.sources import WebFetcher
from deepsearch_agent.tools.transport import HttpClient


@pytest.fixture(autouse=True)
def _run_parser_inline(monkeypatch):
    """当前受限测试容器中 BeautifulSoup 在线程池会阻塞；不改变生产线程隔离。"""

    async def run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("deepsearch_agent.tools.sources.fetcher.asyncio.to_thread", run_inline)


class FakeHttpClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.last_kwargs = {}

    async def arequest(self, *args, **kwargs):
        self.last_kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


def response(payload, *, url="https://example.com", content=b"", content_type="application/json"):
    body = content or json.dumps(payload).encode()
    return httpx.Response(
        200, content=body, headers={"content-type": content_type}, request=httpx.Request("GET", url)
    )


def test_classify_source_flags_primary_domains_conservatively():
    from deepsearch_agent.tools.search.models import classify_source, describe_source

    assert classify_source("https://arxiv.org/pdf/1706.03762") == "primary"
    assert classify_source("https://www.stats.gov.cn/x") == "primary"
    assert classify_source("https://cs.stanford.edu/~person") == "primary"
    assert classify_source("https://nsfc.gov.cn/p1") == "primary"
    # 商业门户/学生报纸/会议营销站 → general（不冒充一手权威）
    assert classify_source("https://www.thecrimson.com/article") == "general"
    assert classify_source("https://uppersideconferences.com/2024") == "general"
    assert classify_source("https://m.36kr.com/p/1") == "general"
    assert describe_source("https://arxiv.org/abs/1706.03762").model_dump() == {
        "source_type": "academic",
        "authority_tier": "primary",
        "publication_status": "preprint",
        "primary_source": True,
    }
    assert describe_source("https://www.stats.gov.cn/x").model_dump() == {
        "source_type": "government",
        "authority_tier": "primary",
        "publication_status": "official",
        "primary_source": True,
    }
    assert describe_source("https://m.36kr.com/p/1").authority_tier == "secondary"


@pytest.mark.parametrize(
    ("url", "source_type", "tier"),
    [
        ("https://space.bilibili.com/123/video", "media", "secondary"),
        ("https://www.zhihu.com/question/1", "blog", "secondary"),
        ("https://author.medium.com/post", "blog", "secondary"),
        ("https://cs.stanford.edu/paper", "academic", "primary"),
        ("https://news.tsinghua.edu.cn/info", "academic", "primary"),
        ("https://datatracker.ietf.org/doc/rfc9000", "standard", "primary"),
        ("https://platform.openai.com/docs", "company", "primary"),
    ],
)
def test_source_profile_rules_cover_subdomains(url, source_type, tier):
    from deepsearch_agent.tools.search.models import describe_source

    profile = describe_source(url)
    assert profile.source_type == source_type
    assert profile.authority_tier == tier


def test_source_profile_domain_rules_resist_suffix_spoofing():
    from deepsearch_agent.tools.search.models import describe_source

    profile = describe_source("https://bilibili.com.evil.example/video")
    assert profile.source_type == "general"
    assert profile.authority_tier == "secondary"


def test_search_parses_tavily_response_without_network():
    config = SearchConfig(tavily_api_key="test", max_results=2)
    client = SearchClient(
        config,
        FakeHttpClient(
            response(
                {
                    "results": [
                        {
                            "title": "A",
                            "url": "https://a.test",
                            "content": "text",
                            "score": 0.8,
                            "published_date": "2026-07-04T00:00:00Z",
                        }
                    ]
                }
            )
        ),
    )
    assert asyncio.run(client.asearch("question")) == [
        {
            "title": "A",
            "url": "https://a.test",
            "snippet": "text",
            "raw_content": "",
            "content_provider": "tavily",
            "score": 0.8,
            "published_at": "2026-07-04T00:00:00Z",
        }
    ]


def test_tavily_null_raw_content_still_validates_as_result():
    """回归守卫：Tavily advanced depth 对无法抽取正文的页返回 JSON null（非缺键）。
    `.get(k,"")` 只在缺键时兜底，null 会漏成 None，进而令 SearchToolResult 的
    Pydantic 校验整批失败 → researcher 反复重试空转（真实评测被卡 684 事件零产出）。
    必须：null 归一为 ""，且能过 model_validate。"""
    from deepsearch_agent.tools.search.models import SearchToolResult

    config = SearchConfig(tavily_api_key="test", max_results=5)
    client = SearchClient(
        config,
        FakeHttpClient(
            response(
                {
                    "results": [
                        {
                            "title": None,
                            "url": "https://a.test",
                            "content": None,
                            "raw_content": None,  # 付费墙/JS 页：显式 null
                            "score": None,
                            "published_date": None,
                        },
                        {
                            "title": "OK",
                            "url": "https://b.test",
                            "content": "有摘要",
                            "raw_content": "有正文",
                            "score": 0.4,
                            "published_date": "2026-01-01",
                        },
                    ]
                }
            )
        ),
    )
    results = asyncio.run(client.asearch("中国一线城市 租金回报率"))
    assert [type(r["raw_content"]).__name__ for r in results] == ["str", "str"]
    assert results[0]["raw_content"] == "" and results[0]["title"] == ""
    assert results[0]["published_at"] == ""  # str(None) 会得到 "None" 字符串污染
    # 关键：整批能被真实消费方（researcher）的 Pydantic 校验接受
    SearchToolResult.model_validate({"task_id": "t", "status": "completed", "results": results})


def test_search_parses_baidu_references_through_common_result_contract():
    config = SearchConfig(provider="baidu", baidu_api_key="test", max_results=2)
    client = SearchClient(
        config,
        FakeHttpClient(
            response(
                {
                    "request_id": "req-1",
                    "references": [
                        {
                            "title": "天气页面",
                            "url": "[天气](https://weather.test/page)",
                            "content": "今天晴。",
                            "date": "2025-05-23 00:00:00",
                            "type": "web",
                        },
                        {"title": "图片", "url": "https://image.test/1", "type": "image"},
                    ],
                }
            )
        ),
    )

    assert asyncio.run(client.asearch("天气")) == [
        {
            "title": "天气页面",
            "url": "https://weather.test/page",
            "snippet": "今天晴。",
            "content_provider": "baidu",
            "score": 1.0,
            "published_at": "2025-05-23 00:00:00",
        }
    ]


def test_search_reports_baidu_api_error_instead_of_empty_results():
    client = SearchClient(
        SearchConfig(provider="baidu", baidu_api_key="test"),
        FakeHttpClient(response({"code": "401", "message": "invalid key"})),
    )

    with pytest.raises(ToolRequestError, match="invalid key"):
        asyncio.run(client.asearch("question"))


def test_search_rejects_missing_configuration():
    with pytest.raises(ToolConfigurationError):
        asyncio.run(SearchClient(SearchConfig()).asearch("question"))


def test_search_rejects_malformed_response():
    config = SearchConfig(tavily_api_key="test")
    client = SearchClient(
        config,
        FakeHttpClient(
            response(
                {
                    "results": [
                        {"title": "bad score", "url": "https://a.test", "score": "not-a-number"}
                    ]
                }
            )
        ),
    )
    with pytest.raises(ToolParseError):
        asyncio.run(client.asearch("question"))


def test_fetch_parses_html_without_network():
    html = b"<html><head><title>Example</title></head><body><p>Hello world</p></body></html>"
    fake = FakeHttpClient(
        response({}, url="https://example.com/page", content=html, content_type="text/html")
    )
    document = asyncio.run(WebFetcher(SearchConfig(), fake).afetch("https://example.com/page"))
    assert document["title"] == "Example"
    assert "Hello world" in document["text"]
    assert document["content_hash"]
    assert fake.last_kwargs["request_kind"] == "fetch"


def test_fetch_rejects_captcha_page_before_evidence_extraction():
    html = b"<html><head><title>\xe9\xaa\x8c\xe8\xaf\x81\xe7\xa0\x81_\xe5\x93\x94\xe5\x93\xa9\xe5\x93\x94\xe5\x93\xa9</title></head><body>captcha</body></html>"
    fake = FakeHttpClient(
        response({}, url="https://www.bilibili.com/opus/1", content=html, content_type="text/html")
    )
    with pytest.raises(SourceUnavailableError) as exc_info:
        asyncio.run(WebFetcher(SearchConfig(), fake).afetch("https://www.bilibili.com/opus/1"))
    assert exc_info.value.reason_code == "access_challenge"


def test_http_client_retries_transient_timeout():
    class RetryingClient:
        def __init__(self):
            self.calls = 0

        async def request(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise Timeout("temporary")
            return httpx.Response(200, request=httpx.Request("GET", "https://example.com"))

    raw = RetryingClient()
    config = SearchConfig(retry_attempts=2, retry_initial_seconds=0, retry_max_seconds=0)
    assert (
        asyncio.run(HttpClient(config, raw).arequest("GET", "https://example.com")).status_code
        == 200
    )
    assert raw.calls == 2


def test_http_client_raises_typed_error_after_retries():
    class FailingClient:
        async def request(self, *args, **kwargs):
            return httpx.Response(503, request=httpx.Request("GET", "https://example.com"))

    config = SearchConfig(retry_attempts=1, retry_initial_seconds=0, retry_max_seconds=0)
    with pytest.raises(ToolRequestError):
        asyncio.run(HttpClient(config, FailingClient()).arequest("GET", "https://example.com"))


class _StatusClient:
    """总是返回指定状态码与响应头的假客户端。"""

    def __init__(self, status: int, headers: dict[str, str] | None = None):
        self.status = status
        self.headers = headers or {}

    async def request(self, *args, **kwargs):
        return httpx.Response(
            self.status, headers=self.headers, request=httpx.Request("GET", "https://example.com")
        )


def _collect_sleeps(monkeypatch):
    import deepsearch_agent.tools.transport.http_client as http_client_module

    sleeps: list[float] = []

    async def record_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(http_client_module.asyncio, "sleep", record_sleep)
    return sleeps


def test_http_client_prefers_retry_after_over_local_backoff(monkeypatch):
    """429 的 Retry-After 优先于本地指数曲线，且不被 retry_max_seconds 截断。"""
    sleeps = _collect_sleeps(monkeypatch)
    config = SearchConfig(
        retry_attempts=2,
        retry_initial_seconds=1.0,
        retry_max_seconds=10.0,
        retry_after_max_seconds=60.0,
    )
    client = HttpClient(config, _StatusClient(429, {"Retry-After": "30"}))
    with pytest.raises(ToolRequestError):
        asyncio.run(client.arequest("GET", "https://example.com"))

    assert len(sleeps) == 1
    # 基准 30s，允许最多 +25% jitter
    assert 30.0 <= sleeps[0] <= 37.5


def test_http_client_caps_retry_after_at_independent_limit(monkeypatch):
    """服务端要求过长等待时按 retry_after_max_seconds 截断，避免病态挂起。"""
    sleeps = _collect_sleeps(monkeypatch)
    config = SearchConfig(
        retry_attempts=2,
        retry_initial_seconds=1.0,
        retry_max_seconds=10.0,
        retry_after_max_seconds=45.0,
    )
    client = HttpClient(config, _StatusClient(503, {"Retry-After": "3600"}))
    with pytest.raises(ToolRequestError):
        asyncio.run(client.arequest("GET", "https://example.com"))

    assert 45.0 <= sleeps[0] <= 56.25


def test_http_client_parses_http_date_retry_after(monkeypatch):
    from datetime import datetime, timedelta, timezone
    from email.utils import format_datetime

    when = datetime.now(timezone.utc) + timedelta(seconds=12)
    sleeps = _collect_sleeps(monkeypatch)
    config = SearchConfig(
        retry_attempts=2,
        retry_initial_seconds=1.0,
        retry_max_seconds=10.0,
        retry_after_max_seconds=60.0,
    )
    client = HttpClient(config, _StatusClient(429, {"Retry-After": format_datetime(when)}))
    with pytest.raises(ToolRequestError):
        asyncio.run(client.arequest("GET", "https://example.com"))

    assert 0 < sleeps[0] <= 15 + 15 * 0.25


def test_http_client_keeps_exponential_backoff_without_retry_after(monkeypatch):
    """无 Retry-After 时保持原指数曲线（外加至多 25% jitter）。"""
    sleeps = _collect_sleeps(monkeypatch)
    config = SearchConfig(
        retry_attempts=3,
        retry_initial_seconds=2.0,
        retry_max_seconds=10.0,
    )
    client = HttpClient(config, _StatusClient(500))
    with pytest.raises(ToolRequestError):
        asyncio.run(client.arequest("GET", "https://example.com"))

    assert len(sleeps) == 2
    assert 2.0 <= sleeps[0] <= 2.5
    assert 4.0 <= sleeps[1] <= 5.0


def test_parse_retry_after_rejects_garbage():
    from deepsearch_agent.tools.transport import parse_retry_after

    assert parse_retry_after(None) is None
    assert parse_retry_after("") is None
    assert parse_retry_after("not-a-date") is None
    assert parse_retry_after("7") == 7.0


def test_http_client_records_rate_limit_reset_on_429(monkeypatch):
    """429 重试耗尽后，错误携带约 30s（Retry-After 优先）的单调恢复点。"""
    _collect_sleeps(monkeypatch)
    import time as _time

    config = SearchConfig(retry_attempts=2, retry_initial_seconds=0, retry_max_seconds=0)
    client = HttpClient(config, _StatusClient(429, {"Retry-After": "30"}))
    with pytest.raises(ToolRequestError) as caught:
        asyncio.run(client.arequest("GET", "https://example.com"))

    assert _time.monotonic() + 29 <= caught.value.rate_limit_reset_ts <= _time.monotonic() + 31


def test_http_client_uses_default_window_for_429_without_header(monkeypatch):
    _collect_sleeps(monkeypatch)
    import time as _time

    config = SearchConfig(retry_attempts=1, retry_initial_seconds=0, retry_max_seconds=0)
    client = HttpClient(config, _StatusClient(429))
    with pytest.raises(ToolRequestError) as caught:
        asyncio.run(client.arequest("GET", "https://example.com"))

    assert _time.monotonic() + 29 <= caught.value.rate_limit_reset_ts <= _time.monotonic() + 31


def test_http_client_no_reset_ts_for_non_rate_limit_failure(monkeypatch):
    _collect_sleeps(monkeypatch)
    config = SearchConfig(retry_attempts=1, retry_initial_seconds=0, retry_max_seconds=0)
    client = HttpClient(config, _StatusClient(500))
    with pytest.raises(ToolRequestError) as caught:
        asyncio.run(client.arequest("GET", "https://example.com"))

    assert caught.value.rate_limit_reset_ts is None


class _RateLimitedProvider:
    """记录调用次数的假 provider；always_limited 时只抛非限流的普通错误。"""

    def __init__(self, error: ToolRequestError):
        self.error = error
        self.calls = 0

    async def asearch(self, query: str, limit: int):
        self.calls += 1
        raise self.error


def _client_with_fake_provider(monkeypatch, provider):
    import time as _time

    client = SearchClient(SearchConfig(provider="baidu", baidu_api_key="test"))
    monkeypatch.setattr(client, "_provider", lambda: provider)
    return client, _time


def test_search_client_breaker_opens_after_rate_limited_failure(monkeypatch):
    """provider 报限流后，窗口内的后续查询直接快速失败，不再触网。"""
    import time as _time

    error = ToolRequestError("请求失败（重试 3 次）")
    error.rate_limit_reset_ts = _time.monotonic() + 30
    provider = _RateLimitedProvider(error)
    client, _ = _client_with_fake_provider(monkeypatch, provider)

    with pytest.raises(ToolRequestError):
        asyncio.run(client.asearch("第一个查询"))
    assert provider.calls == 1

    with pytest.raises(ToolRequestError, match="已被限流") as caught:
        asyncio.run(client.asearch("第二个查询"))
    assert provider.calls == 1  # 第二个查询没有出网
    assert caught.value.retryable is False
    assert "约 30 秒后恢复" in str(caught.value) or "秒后恢复" in str(caught.value)


def test_search_client_breaker_expires_and_admits_probe(monkeypatch):
    """恢复窗口到期后，下一个请求放行做探针；成功则恢复正常。"""
    import time as _time

    error = ToolRequestError("请求失败")
    error.rate_limit_reset_ts = _time.monotonic() + 30
    provider = _RateLimitedProvider(error)
    client, _ = _client_with_fake_provider(monkeypatch, provider)

    with pytest.raises(ToolRequestError):
        asyncio.run(client.asearch("q1"))
    # 模拟窗口已过
    client._rate_limit_deadlines["baidu"] = _time.monotonic() - 1
    with pytest.raises(ToolRequestError):  # provider 仍然失败（非限流路径会重新开窗判断）
        asyncio.run(client.asearch("q2"))
    assert provider.calls == 2  # 探针请求真的出网了


def test_search_client_ignores_plain_failures_for_breaker(monkeypatch):
    """不带恢复点的普通失败（如 401）不开断路器。"""
    provider = _RateLimitedProvider(ToolRequestError("HTTP 401", retryable=False))
    client, _ = _client_with_fake_provider(monkeypatch, provider)

    for _ in range(2):
        with pytest.raises(ToolRequestError, match="401"):
            asyncio.run(client.asearch("q"))
    assert provider.calls == 2
    assert client._rate_limit_deadlines == {}


class _ConcurrencyTrackingProvider:
    """记录并发峰值的假 provider；每个调用短暂挂起以制造重叠。"""

    def __init__(self):
        self.in_flight = 0
        self.peak = 0
        self.calls = 0

    async def asearch(self, query: str, limit: int):
        import asyncio

        self.in_flight += 1
        self.calls += 1
        self.peak = max(self.peak, self.in_flight)
        await asyncio.sleep(0.01)
        self.in_flight -= 1
        return []


def test_search_client_caps_provider_concurrency(monkeypatch):
    """跨 worker 并发查询被收敛到 max_concurrent_requests 路在飞。"""
    provider = _ConcurrencyTrackingProvider()
    config = SearchConfig(provider="baidu", baidu_api_key="test", max_concurrent_requests=2)
    client = SearchClient(config)
    monkeypatch.setattr(client, "_provider", lambda: provider)

    async def run():
        await asyncio.gather(*(client.asearch(f"q{i}") for i in range(8)))

    asyncio.run(run())
    assert provider.calls == 8
    assert provider.peak <= 2


def test_search_client_queued_request_bails_when_breaker_opens(monkeypatch):
    """拿到并发许可后复查断路：排队期间被别的请求开窗，则不出网。"""
    import time as _time

    release = asyncio.Event()
    opened = asyncio.Event()

    class BlockingThenRateLimited:
        def __init__(self):
            self.calls = 0

        async def asearch(self, query, limit):
            self.calls += 1
            if self.calls == 1:
                await release.wait()  # 占住唯一许可,让第二个请求排队
                error = ToolRequestError("请求失败（重试 3 次）")
                error.rate_limit_reset_ts = _time.monotonic() + 30
                raise error
            opened.set()
            return []

    provider = BlockingThenRateLimited()
    config = SearchConfig(provider="baidu", baidu_api_key="test", max_concurrent_requests=1)
    client = SearchClient(config)
    monkeypatch.setattr(client, "_provider", lambda: provider)

    async def run():
        first = asyncio.create_task(client.asearch("q1"))
        await asyncio.sleep(0)  # 让 first 进入 provider 并占住许可
        second = asyncio.create_task(client.asearch("q2"))  # 卡在信号量上
        await asyncio.sleep(0)
        release.set()  # first 结束并开断路器；second 拿到许可后复查 → 快速失败
        with pytest.raises(ToolRequestError):
            await first
        with pytest.raises(ToolRequestError, match="已被限流"):
            await second

    asyncio.run(run())
    assert provider.calls == 1  # 第二个请求复查断路后未出网
    assert not opened.is_set()
