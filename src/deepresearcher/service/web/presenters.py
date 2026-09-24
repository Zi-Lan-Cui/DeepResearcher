"""服务模型对应的稳定 HTTP 响应投影。"""

from datetime import datetime, timezone
from typing import Any

from deepresearcher.service.persistence.models import Run


def run_summary(run: Run) -> dict[str, Any]:
    elapsed_ms = None
    if run.started_at:
        started_at = run.started_at
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=timezone.utc)
        end = run.finished_at or datetime.now(timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        elapsed_ms = max(0, round((end - started_at).total_seconds() * 1000))
    return {
        "id": run.id,
        "query": run.query,
        "status": run.status,
        "answer_mode": run.answer_mode,
        "terminal_reason": run.terminal_reason,
        "evidence_count": run.evidence_count,
        "source_count": run.source_count,
        "llm_call_count": run.llm_call_count,
        "input_tokens": run.input_tokens,
        "output_tokens": run.output_tokens,
        "cached_input_tokens": run.cached_input_tokens,
        "external_request_count": run.external_request_count,
        "peak_llm_concurrency": run.peak_llm_concurrency,
        "estimated_cost_usd": float(run.estimated_cost_usd or 0),
        "cache_hit_count": run.cache_hit_count,
        "saved_external_request_count": run.saved_external_request_count,
        "saved_llm_call_count": run.saved_llm_call_count,
        "saved_tokens": run.saved_tokens,
        "saved_cost_usd": float(run.saved_cost_usd or 0),
        "elapsed_ms": elapsed_ms,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
    }
