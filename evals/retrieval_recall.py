"""用本地 fetch 缓存与历史 checkpoint 离线评估块召回。

只比较真实抓取后保存的完整 ``DocumentBlock``；没有 fetch 缓存时明确返回
``no_fetch_documents``，不使用 Evidence 的局部 audit_chunk 冒充完整网页。
"""

from __future__ import annotations

import asyncio
import json

from sqlalchemy import select

from deepresearcher.config import get_settings
from deepresearcher.evidence.models import Evidence
from deepresearcher.evidence.retrieval import BM25Retriever
from deepresearcher.evidence.tokens import get_token_estimator
from deepresearcher.service.persistence.database import make_engine, make_session_factory
from deepresearcher.service.persistence.models import Run, ToolCacheEntry
from deepresearcher.service.settings import checkpoint_dsn, get_service_config
from deepresearcher.tools.cache_keys import canonical_url
from deepresearcher.tools.web.parsing.models import DocumentBlock


def _tokens(blocks: list[DocumentBlock]) -> int:
    estimator = get_token_estimator()
    return sum(
        estimator.count(
            f"[{block.get('block_id', '')}] "
            f"{' > '.join(block.get('heading_path', []))}\n{block.get('text', '')}"
        )
        for block in blocks
    )


async def evaluate() -> dict[str, object]:
    """返回长文历史 Evidence 的 quote 召回率与输入 token 压缩统计。"""
    service = get_service_config()
    settings = get_settings()
    engine = make_engine(service.database_url)
    factory = make_session_factory(engine)
    try:
        async with factory() as session:
            rows = (
                await session.execute(
                    select(ToolCacheEntry.value_json).where(ToolCacheEntry.namespace == "fetch")
                )
            ).scalars()
            documents: dict[str, list[DocumentBlock]] = {}
            for raw in rows:
                if not isinstance(raw, dict) or not isinstance(raw.get("blocks"), list):
                    continue
                url = canonical_url(str(raw.get("final_url") or raw.get("source_url") or ""))
                if url:
                    documents[url] = list(raw["blocks"])
            run_ids = list((await session.execute(select(Run.id))).scalars())
    finally:
        await engine.dispose()

    if not documents:
        return {
            "status": "no_fetch_documents",
            "fetch_documents": 0,
            "eligible_evidences": 0,
            "quote_hits": 0,
            "quote_recall": None,
            "average_full_tokens": None,
            "average_recalled_tokens": None,
        }

    dsn = checkpoint_dsn(service.database_url)
    if dsn is None:
        return {"status": "checkpoint_unavailable", "fetch_documents": len(documents)}

    retriever = BM25Retriever()
    eligible = hits = full_tokens = recalled_tokens = 0
    seen: set[tuple[str, str, str]] = set()
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    async with AsyncPostgresSaver.from_conn_string(dsn) as checkpointer:
        for run_id in run_ids:
            checkpoint = await checkpointer.aget_tuple(
                {"configurable": {"thread_id": run_id, "checkpoint_ns": ""}}
            )
            if checkpoint is None:
                continue
            values = checkpoint.checkpoint.get("channel_values", {})
            raw_evidences = values.get("evidences", []) if isinstance(values, dict) else []
            for raw in raw_evidences:
                try:
                    evidence = raw if isinstance(raw, Evidence) else Evidence.model_validate(raw)
                except (TypeError, ValueError):
                    continue
                url = canonical_url(evidence.source_url)
                blocks = documents.get(url)
                key = (url, evidence.research_direction, evidence.quote)
                if not blocks or key in seen:
                    continue
                seen.add(key)
                before = _tokens(blocks)
                if before <= settings.agent.evidence_full_context_max_tokens:
                    continue
                selected = retriever.select(
                    blocks,
                    evidence.research_direction,
                    settings.agent.evidence_bm25_top_k,
                    settings.agent.evidence_bm25_window,
                )
                selected_ids = {block.get("block_id", "") for block in selected}
                hit = bool(set(evidence.locator.block_ids) & selected_ids) or any(
                    evidence.quote in block.get("text", "") for block in selected
                )
                eligible += 1
                hits += int(hit)
                full_tokens += before
                recalled_tokens += _tokens(selected)

    return {
        "status": "ok" if eligible else "no_eligible_long_document_evidence",
        "fetch_documents": len(documents),
        "eligible_evidences": eligible,
        "quote_hits": hits,
        "quote_recall": round(hits / eligible, 4) if eligible else None,
        "average_full_tokens": round(full_tokens / eligible, 1) if eligible else None,
        "average_recalled_tokens": round(recalled_tokens / eligible, 1) if eligible else None,
    }


def main() -> None:
    print(json.dumps(asyncio.run(evaluate()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
