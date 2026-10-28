"""确定性校验评测"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from evals.schemas import Criterion, CriterionResult

# 兼容旧来源标识
CITATION_MARKER = re.compile(r"\[(?:证据|来源)(\d+)\]")


@dataclass
class Artifact:
    """一次 run 的评测素材,来源两类:产品 API 返回、DB 的 run_events 原始记录。"""

    case_id: str
    attempt: int
    run_id: str
    detail: dict[str, Any]  # GET /api/runs/{id} 返回
    events: list[dict[str, Any]] = field(default_factory=list)  # run_events.record

    @property
    def report(self) -> str:
        return str(self.detail.get("report_markdown") or "")

    @property
    def citations(self) -> list[dict[str, Any]]:
        return list(self.detail.get("citations") or [])

    def events_of(self, event_type: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("event_type") == event_type]

    def span_failures(self) -> list[dict[str, Any]]:
        """返回可直接用于归因的标准化失败 Span，不依赖业务 payload。"""
        failures: list[dict[str, Any]] = []
        for event in self.events_of("span_failed"):
            raw_payload = event.get("payload")
            payload = raw_payload if isinstance(raw_payload, dict) else {}

            def value(name: str, default: Any = "") -> Any:
                return event.get(name, payload.get(name, default))

            failures.append(
                {
                    "trace_id": str(value("trace_id")),
                    "span_id": str(value("span_id")),
                    "parent_span_id": str(value("parent_span_id")),
                    "name": str(value("name", "unknown_span")),
                    "kind": str(value("kind", "unknown")),
                    "node_id": str(value("node_id")),
                    "duration_ms": float(value("duration_ms", 0.0) or 0.0),
                    "error": str(value("error", "UnknownError") or "UnknownError"),
                }
            )
        return failures


Check = Callable[[Artifact, Criterion], CriterionResult]


def _result(
    artifact: Artifact,
    criterion: Criterion,
    verdict: str,
    reason: str = "",
) -> CriterionResult:
    return CriterionResult(
        case_id=artifact.case_id,
        attempt=artifact.attempt,
        criterion_id=criterion.id,
        dimension=criterion.dimension,
        verdict=verdict,  # type: ignore[arg-type]
        source="deterministic",
        reason=reason,
    )


def check_citation_integrity(artifact: Artifact, criterion: Criterion) -> CriterionResult:
    """引用正确性校验:正文引用的证据/来源标号必须在 citations 列表中有对应条目,且条目必须有 url 与 quote 字段。"""
    status = artifact.detail.get("status")
    if status != "completed":
        return _result(artifact, criterion, "unknown", f"未到 completed（status={status}）")
    report = artifact.report
    if not report:
        return _result(artifact, criterion, "no", "completed 但报告为空")
    refs = {int(m) for m in CITATION_MARKER.findall(report)}
    citations = artifact.citations
    dangling = sorted(n for n in refs if not 1 <= n <= len(citations))
    hollow = [
        i + 1
        for i, c in enumerate(citations)
        if not str(c.get("url") or "").strip() or not str(c.get("quote") or "").strip()
    ]
    if not refs:
        return _result(artifact, criterion, "no", "正文零引用")
    if dangling or hollow:
        return _result(
            artifact,
            criterion,
            "no",
            f"悬空引用={dangling[:5]} 空字段来源={hollow[:5]}",
        )
    return _result(artifact, criterion, "yes", f"{len(refs)} 个引用位全部可溯")


def check_done_exactly_once(artifact: Artifact, criterion: Criterion) -> CriterionResult:
    """run_done 事件恰一帧。"""
    dones = artifact.events_of("run_done")
    if len(dones) == 1:
        return _result(artifact, criterion, "yes")
    return _result(artifact, criterion, "no", f"run_done 帧数={len(dones)}")


def check_seq_continuous(artifact: Artifact, criterion: Criterion) -> CriterionResult:
    """事件 seq 连续性校验:事件序列必须是连续的数字。"""
    seqs = [int(e["seq"]) for e in artifact.events if "seq" in e]
    if seqs and seqs == list(range(1, len(seqs) + 1)):
        return _result(artifact, criterion, "yes", f"seq 1..{len(seqs)} 连续")
    return _result(artifact, criterion, "no", "seq 有洞/乱序（跨世续号回归）")


CHECKS: dict[str, Check] = {
    "citation_integrity": check_citation_integrity,
    "done_exactly_once": check_done_exactly_once,
    "seq_continuous": check_seq_continuous,
}

UNIVERSAL_GATES = (
    "citation_integrity",
    "done_exactly_once",
    "seq_continuous",
)


def score_artifact(
    artifact: Artifact,
    *,
    gates: tuple[str, ...] = UNIVERSAL_GATES,
) -> list[CriterionResult]:
    results: list[CriterionResult] = []
    for name in gates:
        probe = Criterion(id=f"gate:{name}", dimension="grounding", text=name)
        results.append(CHECKS[name](artifact, probe))
    return results


def process_metrics(artifact: Artifact) -> dict[str, Any]:
    """收集过程指标,只记录,不判 pass/fail。"""
    detail = artifact.detail
    return {
        "status": detail.get("status"),
        "terminal_reason": detail.get("terminal_reason"),
        "answer_mode": detail.get("answer_mode"),
        "evidence_count": detail.get("evidence_count"),
        "source_count": detail.get("source_count"),
        "llm_call_count": detail.get("llm_call_count"),
        "input_tokens": detail.get("input_tokens"),
        "output_tokens": detail.get("output_tokens"),
        "cached_input_tokens": detail.get("cached_input_tokens"),
        "estimated_cost_usd": detail.get("estimated_cost_usd"),
        "external_request_count": detail.get("external_request_count"),
        "cache_hit_count": detail.get("cache_hit_count"),
        "saved_external_request_count": detail.get("saved_external_request_count"),
        "elapsed_ms": detail.get("elapsed_ms"),
    }
