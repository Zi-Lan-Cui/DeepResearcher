"""服务模型对应的稳定 HTTP 响应投影。"""

from datetime import datetime, timezone
from typing import Any

from deepresearcher.service.storage.tables import Run
from deepresearcher.vocab import INCOMPLETE_REPORT_HEADING

# 空壳页的旧标题：改词前落库的报告仍按诊断页呈现，否则存量数据会被判成完整正文。
_LEGACY_INCOMPLETE_HEADINGS = ("研究未完成",)

# 「弃权」是审阅失败的旧词：换词前落库的行仍写着它，出口统一成新词，前端只认一个。
_LEGACY_TERMINAL_REASONS = {"review_abstained": "review_failed"}


def _terminal_reason(value: str | None) -> str | None:
    if not value:
        return value
    return _LEGACY_TERMINAL_REASONS.get(value, value)


def report_kind(markdown: str | None) -> str:
    """报告正文形态：none=没有可读正文，full=有完整正文。"""
    text = (markdown or "").lstrip()
    if not text:
        return "none"
    headings = (INCOMPLETE_REPORT_HEADING, *_LEGACY_INCOMPLETE_HEADINGS)
    return "none" if any(text.startswith(f"# {item}") for item in headings) else "full"


def run_summary(run: Run) -> dict[str, Any]:
    elapsed_ms = None
    if run.started_at:
        started_at = run.started_at
        end = run.finished_at or datetime.now(timezone.utc)
        elapsed_ms = max(0, round((end - started_at).total_seconds() * 1000))
    return {
        "id": run.id,
        "query": run.query,
        "headline": run.headline,
        "title": run.title,
        "status": run.status,
        "answer_mode": run.answer_mode,
        "terminal_reason": _terminal_reason(run.terminal_reason),
        # 列表与详情共用的交付形态：徽章据此判定「有没有可读正文」。
        "report_kind": report_kind(run.report_markdown),
        "evidence_count": run.evidence_count,
        "source_count": run.source_count,
        "llm_call_count": run.llm_call_count,
        "input_tokens": run.input_tokens,
        "output_tokens": run.output_tokens,
        "cached_input_tokens": run.cached_input_tokens,
        "external_request_count": run.external_request_count,
        "peak_llm_concurrency": run.peak_llm_concurrency,
        "estimated_cost_cny": float(run.estimated_cost_cny or 0),
        "cache_hit_count": run.cache_hit_count,
        "saved_external_request_count": run.saved_external_request_count,
        "elapsed_ms": elapsed_ms,
        # 已提交事件的序号：前端据此区分"回放段"与"live 段"，历史 clarification 帧不再劫持页面。
        "event_seq": run.event_seq,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
    }
