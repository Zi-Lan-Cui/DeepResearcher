"""基于内容哈希的本地 DocumentStore。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from deepresearcher.tools.web.documents.models import (
    DocumentGrepMatch,
    DocumentOutlineItem,
    DocumentReadRange,
    DocumentRef,
)

_DOCUMENT_ID = re.compile(r"^doc-[0-9a-f]{64}$")


class LocalDocumentStore:
    """将规范化文本写入配置目录，支持同机多进程共享。

    多服务器部署必须让 ``root`` 指向共享文件系统，或替换为实现同一协议的
    对象存储后端。文件名完全由内容哈希生成，不接受调用方路径。
    """

    def __init__(self, root: Path):
        self.root = root

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
    ) -> DocumentRef:
        return await asyncio.to_thread(
            self._put,
            text=text,
            title=title,
            source_url=source_url,
            published_at=published_at,
            retrieval_method=retrieval_method,
            support_ceiling=support_ceiling,
            token_count=token_count,
            outline=outline or [],
        )

    def _put(self, **values: Any) -> DocumentRef:
        text = str(values.pop("text"))
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        # 相同正文可能由不同 URL 发布；document_id 必须同时固定正文与来源，
        # 避免缓存复用时把后一个来源错误归因给第一个来源。
        identity = f"{values['source_url']}\0{text}"
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        document_id = f"doc-{digest}"
        lines = text.splitlines()
        ref = DocumentRef(
            document_id=document_id,
            content_hash=content_hash,
            line_count=len(lines),
            **values,
        )
        path = self.path_for(document_id)
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"metadata": ref.model_dump(mode="json"), "lines": lines}
            temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(temporary, path)
        return ref

    async def get(self, document_id: str) -> DocumentRef:
        payload = await asyncio.to_thread(self._load, document_id)
        return DocumentRef.model_validate(payload["metadata"])

    async def text(self, document_id: str) -> str:
        payload = await asyncio.to_thread(self._load, document_id)
        return "\n".join(payload["lines"])

    async def read(
        self,
        document_id: str,
        ranges: list[tuple[int, int]],
        *,
        max_lines: int,
        max_chars: int,
    ) -> list[DocumentReadRange]:
        payload = await asyncio.to_thread(self._load, document_id)
        lines = list(payload["lines"])
        results: list[DocumentReadRange] = []
        used_lines = 0
        used_chars = 0
        for requested_start, requested_end in ranges:
            if requested_end < requested_start:
                raise ValueError("end_line 不能小于 start_line")
            start = min(requested_start, len(lines) + 1)
            end = min(requested_end, len(lines))
            if start > end:
                continue
            remaining_lines = max_lines - used_lines
            remaining_chars = max_chars - used_chars
            if remaining_lines <= 0 or remaining_chars <= 0:
                break
            selected = lines[start - 1 : min(end, start + remaining_lines - 1)]
            rendered: list[str] = []
            for offset, line in enumerate(selected, start=start):
                item = f"L{offset}: {line}"
                if rendered and len("\n".join([*rendered, item])) > remaining_chars:
                    break
                if not rendered and len(item) > remaining_chars:
                    item = item[:remaining_chars]
                rendered.append(item)
                if len("\n".join(rendered)) >= remaining_chars:
                    break
            if not rendered:
                break
            actual_end = start + len(rendered) - 1
            content = "\n".join(rendered)
            results.append(
                DocumentReadRange(start_line=start, end_line=actual_end, content=content)
            )
            used_lines += len(rendered)
            used_chars += len(content)
        return results

    async def grep(
        self,
        document_id: str,
        queries: list[str],
        *,
        context_lines: int,
        max_matches: int,
        max_chars: int,
    ) -> list[DocumentGrepMatch]:
        payload = await asyncio.to_thread(self._load, document_id)
        lines = list(payload["lines"])
        results: list[DocumentGrepMatch] = []
        used_chars = 0
        seen_ranges: set[tuple[int, int, str]] = set()
        for query in dict.fromkeys(item.strip() for item in queries if item.strip()):
            needle = query.casefold()
            for index, line in enumerate(lines):
                if needle not in line.casefold():
                    continue
                start = max(1, index + 1 - context_lines)
                end = min(len(lines), index + 1 + context_lines)
                key = (start, end, query)
                if key in seen_ranges:
                    continue
                content = "\n".join(
                    f"L{line_number}: {lines[line_number - 1]}"
                    for line_number in range(start, end + 1)
                )
                remaining = max_chars - used_chars
                if remaining <= 0:
                    return results
                content = content[:remaining]
                results.append(
                    DocumentGrepMatch(
                        query=query,
                        start_line=start,
                        end_line=end,
                        content=content,
                    )
                )
                seen_ranges.add(key)
                used_chars += len(content)
                if len(results) >= max_matches:
                    return results
        return results

    def path_for(self, document_id: str) -> Path:
        if not _DOCUMENT_ID.fullmatch(document_id):
            raise ValueError("document_id 格式无效")
        digest = document_id.removeprefix("doc-")
        return self.root / digest[:2] / f"{digest}.json"

    def _load(self, document_id: str) -> dict[str, Any]:
        path = self.path_for(document_id)
        if not path.is_file():
            raise FileNotFoundError(f"文档不存在：{document_id}")
        return json.loads(path.read_text(encoding="utf-8"))
