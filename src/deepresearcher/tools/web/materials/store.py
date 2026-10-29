"""研究材料存储：搜索目录与规范化正文落在 Redis，跨 worker 共享。"""

from __future__ import annotations

import json
import logging
import zlib
from typing import Any

from deepresearcher.tools.web.documents import (
    DocumentOutlineItem,
    DocumentReadRange,
    DocumentRef,
    GrepPatternsResult,
)
from deepresearcher.tools.web.materials.models import (
    SearchResultSet,
    StoredDocument,
    build_stored_document,
)
from deepresearcher.tools.web.materials.paging import grep_patterns, read_lines
from deepresearcher.usage_runtime import record_cache_event
from deepresearcher.vocab import RETRIEVAL_ORIGIN_FETCH

logger = logging.getLogger("deepresearcher.tools.web.materials.store")


class ResearchMaterialStore:
    """在 Redis 中保存可丢弃的搜索目录和规范化正文。"""

    def __init__(
        self,
        client: Any,
        *,
        key_prefix: str,
        search_ttl_seconds: int,
        document_ttl_seconds: int,
    ) -> None:
        self._client = client
        self._prefix = key_prefix.strip(":") or "deepresearcher:material"
        self._search_ttl = max(1, search_ttl_seconds)
        self._document_ttl = max(1, document_ttl_seconds)

    async def put_search_results(self, result_set: SearchResultSet) -> None:
        await self._client.set(
            self._search_key(result_set.run_id, result_set.search_id),
            self._encode(result_set.model_dump(mode="json")),
            ex=self._search_ttl,
        )
        await record_cache_event(namespace="material_search", status="write")

    async def get_search_results(self, run_id: str, search_id: str) -> SearchResultSet:
        payload = await self._client.get(self._search_key(run_id, search_id))
        if payload is None:
            await record_cache_event(namespace="material_search", status="miss")
            raise FileNotFoundError(f"搜索结果不存在：{search_id}")
        # 与 material_document 命中对称：一次 search 缓存命中省去一次外部请求。不计的话
        # saved_external_request_count 恒为 0，要求 saved>0 的行为断言会误判失败。
        await record_cache_event(
            namespace="material_search",
            status="hit",
            detail={"saved_external_requests": 1},
        )
        return SearchResultSet.model_validate(self._decode(payload))

    async def put(
        self,
        *,
        text: str,
        title: str,
        source_url: str,
        published_at: str = "",
        retrieval_method: str = RETRIEVAL_ORIGIN_FETCH,
        support_ceiling: str = "strict",
        token_count: int = 0,
        outline: list[DocumentOutlineItem] | None = None,
        fetch_key: str = "",
    ) -> DocumentRef:
        document = build_stored_document(
            text=text,
            title=title,
            source_url=source_url,
            published_at=published_at,
            retrieval_method=retrieval_method,
            support_ceiling=support_ceiling,
            token_count=token_count,
            outline=outline or [],
        )
        ref = document.ref
        pipeline = self._client.pipeline(transaction=False)
        pipeline.set(
            self._document_key(ref.document_id),
            self._encode(document.model_dump(mode="json")),
            ex=self._document_ttl,
        )
        if fetch_key:
            pipeline.set(
                self._fetch_key(fetch_key),
                ref.document_id,
                ex=self._document_ttl,
            )
        await pipeline.execute()
        await record_cache_event(namespace="material_document", status="write")
        return ref

    async def resolve_fetch(self, fetch_key: str) -> DocumentRef | None:
        if not fetch_key:
            return None
        raw_document_id = await self._client.get(self._fetch_key(fetch_key))
        if raw_document_id is None:
            await record_cache_event(namespace="material_document", status="miss")
            return None
        document_id = self._text(raw_document_id)
        try:
            ref = await self.get(document_id)
        except FileNotFoundError:
            # 索引 TTL 长于正文时按 miss 处理。
            await self._client.delete(self._fetch_key(fetch_key))
            await record_cache_event(namespace="material_document", status="miss")
            return None
        await record_cache_event(
            namespace="material_document",
            status="hit",
            detail={"saved_external_requests": 1},
        )
        return ref

    async def get(self, document_id: str) -> DocumentRef:
        return (await self._document(document_id)).ref

    async def text(self, document_id: str) -> str:
        return (await self._document(document_id)).content

    async def read(
        self,
        document_id: str,
        ranges: list[tuple[int, int]],
        *,
        max_lines: int,
        max_chars: int,
    ) -> list[DocumentReadRange]:
        document = await self._document(document_id)
        return read_lines(
            document.content.splitlines(),
            ranges,
            max_lines=max_lines,
            max_chars=max_chars,
        )

    async def grep_patterns(
        self,
        document_id: str,
        patterns: list[str],
        *,
        context_lines: int,
        segments: list[tuple[int, int]],
        max_matches_per_segment: int,
        max_chars: int,
        max_line_chars: int,
    ) -> GrepPatternsResult:
        document = await self._document(document_id)
        return grep_patterns(
            document.content.splitlines(),
            patterns,
            context_lines=context_lines,
            segments=segments,
            max_matches_per_segment=max_matches_per_segment,
            max_chars=max_chars,
            max_line_chars=max_line_chars,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def _document(self, document_id: str) -> StoredDocument:
        payload = await self._client.get(self._document_key(document_id))
        if payload is None:
            raise FileNotFoundError(f"文档不存在：{document_id}")
        return StoredDocument.model_validate(self._decode(payload))

    def _search_key(self, run_id: str, search_id: str) -> str:
        return f"{self._prefix}:search:{run_id}:{search_id}"

    def _document_key(self, document_id: str) -> str:
        return f"{self._prefix}:document:{document_id}"

    def _fetch_key(self, fetch_key: str) -> str:
        return f"{self._prefix}:fetch:{fetch_key}"

    @staticmethod
    def _encode(value: dict[str, Any]) -> bytes:
        raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return zlib.compress(raw, level=6)

    @staticmethod
    def _decode(value: bytes | str) -> dict[str, Any]:
        raw = value.encode("latin1") if isinstance(value, str) else value
        return json.loads(zlib.decompress(raw).decode("utf-8"))

    @staticmethod
    def _text(value: bytes | str) -> str:
        return value.decode("utf-8") if isinstance(value, bytes) else value


async def create_research_material_store(
    *,
    redis_url: str,
    key_prefix: str,
    search_ttl_seconds: int,
    document_ttl_seconds: int,
) -> ResearchMaterialStore:
    """创建 Worker 拥有的材料存储；Redis 不可用即启动失败。"""

    from redis.asyncio import Redis

    client = Redis.from_url(
        redis_url,
        decode_responses=False,
        # redis-py 8 默认按 RESP3 握手（HELLO 3），服务端低于 6.0 会直接拒绝连接。
        # 材料读写的命令面只用 RESP2 就够，显式钉住，免得跟着库默认值漂移。
        protocol=2,
        socket_connect_timeout=1.0,
        socket_timeout=2.0,
    )
    await client.ping()
    return ResearchMaterialStore(
        client,
        key_prefix=key_prefix,
        search_ttl_seconds=search_ttl_seconds,
        document_ttl_seconds=document_ttl_seconds,
    )
