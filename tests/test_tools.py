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
from deepsearch_agent.tools.fetcher import WebFetcher
from deepsearch_agent.tools.http_client import HttpClient
from deepsearch_agent.tools.search import SearchClient


@pytest.fixture(autouse=True)
def _run_parser_inline(monkeypatch):
    """当前受限测试容器中 BeautifulSoup 在线程池会阻塞；不改变生产线程隔离。"""
    async def run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("deepsearch_agent.tools.fetcher.asyncio.to_thread", run_inline)


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


def test_search_parses_tavily_response_without_network():
    config = SearchConfig(tavily_api_key="test", max_results=2)
    client = SearchClient(
        config,
        FakeHttpClient(
            response(
                {
                    "results": [
                        {"title": "A", "url": "https://a.test", "content": "text", "score": 0.8}
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
        }
    ]


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
