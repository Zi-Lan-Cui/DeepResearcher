"""Redis 研究材料存储适配器。"""

from __future__ import annotations

import json
import logging
import zlib
from typing import Any

from deepresearcher.observability.usage_runtime import record_cache_event
from deepresearcher.tools.web.documents import (
    DocumentOutlineItem,
    DocumentReadRange,
    DocumentRef,
    GrepResult,
)
from deepresearcher.tools.web.materials import (
    MemoryResearchMaterialStore,
    SearchResultSet,
    StoredDocument,
)
from deepresearcher.tools.web.materials.models import build_stored_document
from deepresearcher.tools.web.materials.store import grep_lines, read_lines

logger = logging.getLogger("deepresearcher.service.persistence.redis_material_store")


class RedisResearchMaterialStore:
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
        # Redis 写入失败时仍要让当前进程完成研究。
        self._fallback = MemoryResearchMaterialStore()

    async def put_search_results(self, result_set: SearchResultSet) -> None:
        payload = self._encode(result_set.model_dump(mode="json"))
        try:
            await self._client.set(
                self._search_key(result_set.run_id, result_set.search_id),
                payload,
                ex=self._search_ttl,
            )
            await record_cache_event(namespace="material_search", status="write")
        except Exception:  # noqa: BLE001 - 材料可重建，当前运行降级到内存
            logger.warning(
                "redis_material_search_write_failed search_id=%s",
                result_set.search_id,
                exc_info=True,
            )
            await self._fallback.put_search_results(result_set)

    async def get_search_results(self, run_id: str, search_id: str) -> SearchResultSet:
        try:
            payload = await self._client.get(self._search_key(run_id, search_id))
        except Exception:  # noqa: BLE001 - 尝试当前进程的降级副本
            logger.warning(
                "redis_material_search_read_failed search_id=%s",
                search_id,
                exc_info=True,
            )
            return await self._fallback.get_search_results(run_id, search_id)
        if payload is None:
            await record_cache_event(namespace="material_search", status="miss")
            return await self._fallback.get_search_results(run_id, search_id)
        # 与 material_document 命中对称：一次 search 缓存命中即省去一次外部请求，
        # 否则 usage 只记 cache_hit_count、saved_external_request_count 恒为 0，
        # cache-reuse 行为断言（要求 saved>0）会误判失败。
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
        retrieval_method: str = "origin_fetch",
        support_ceiling: str = "direct",
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
        try:
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
        except Exception:  # noqa: BLE001 - 内存副本保证当前运行可继续
            logger.warning(
                "redis_material_document_write_failed document_id=%s",
                ref.document_id,
                exc_info=True,
            )
            await self._fallback.put(
                text=text,
                title=title,
                source_url=source_url,
                published_at=published_at,
                retrieval_method=retrieval_method,
                support_ceiling=support_ceiling,
                token_count=token_count,
                outline=outline,
                fetch_key=fetch_key,
            )
        return ref

    async def resolve_fetch(self, fetch_key: str) -> DocumentRef | None:
        if not fetch_key:
            return None
        try:
            raw_document_id = await self._client.get(self._fetch_key(fetch_key))
            if raw_document_id is None:
                await record_cache_event(namespace="material_document", status="miss")
                return await self._fallback.resolve_fetch(fetch_key)
            document_id = self._text(raw_document_id)
            try:
                ref = await self.get(document_id)
                await record_cache_event(
                    namespace="material_document",
                    status="hit",
                    detail={"saved_external_requests": 1},
                )
                return ref
            except FileNotFoundError:
                # 索引比正文活得更久时按 miss 处理。
                await self._client.delete(self._fetch_key(fetch_key))
                await record_cache_event(namespace="material_document", status="miss")
                return None
        except Exception:  # noqa: BLE001 - 降级副本可能依然命中
            logger.warning("redis_material_fetch_resolve_failed", exc_info=True)
            return await self._fallback.resolve_fetch(fetch_key)

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

    async def grep(
        self,
        document_id: str,
        query: str,
        *,
        context_lines: int,
        max_matches: int,
        max_chars: int,
        offset: int = 0,
    ) -> GrepResult:
        document = await self._document(document_id)
        return grep_lines(
            document.content.splitlines(),
            query,
            context_lines=context_lines,
            max_matches=max_matches,
            max_chars=max_chars,
            offset=offset,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def _document(self, document_id: str) -> StoredDocument:
        try:
            payload = await self._client.get(self._document_key(document_id))
        except Exception:  # noqa: BLE001 - 尝试当前进程副本
            logger.warning(
                "redis_material_document_read_failed document_id=%s",
                document_id,
                exc_info=True,
            )
            ref = await self._fallback.get(document_id)
            return StoredDocument(ref=ref, content=await self._fallback.text(document_id))
        if payload is None:
            try:
                ref = await self._fallback.get(document_id)
            except FileNotFoundError as exc:
                raise FileNotFoundError(f"文档不存在：{document_id}") from exc
            return StoredDocument(ref=ref, content=await self._fallback.text(document_id))
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
    backend: str,
    redis_url: str,
    key_prefix: str,
    search_ttl_seconds: int,
    document_ttl_seconds: int,
) -> MemoryResearchMaterialStore | RedisResearchMaterialStore:
    """创建 Worker 拥有的材料存储；Redis 不可用时显式降级。"""

    if backend == "memory":
        return MemoryResearchMaterialStore()
    if backend != "redis":
        raise ValueError(f"不支持的研究材料存储后端：{backend}")
    try:
        from redis.asyncio import Redis

        client = Redis.from_url(
            redis_url,
            decode_responses=False,
            socket_connect_timeout=1.0,
            socket_timeout=2.0,
        )
        await client.ping()
    except Exception:  # noqa: BLE001 - 材料可重建，启动降级不阻断 Worker
        logger.warning("redis_material_store_unavailable; using process memory", exc_info=True)
        return MemoryResearchMaterialStore()
    return RedisResearchMaterialStore(
        client,
        key_prefix=key_prefix,
        search_ttl_seconds=search_ttl_seconds,
        document_ttl_seconds=document_ttl_seconds,
    )
