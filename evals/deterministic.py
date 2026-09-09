"""确定性 scorer：代码可确证的事绝不交给 LLM。

三类确定性判定：
1. **引用完整性门**（grounding 维）——正文 `[来源N]` 全部指向 citations_json
   合法下标、来源条目 url/quote 非空。运行期校验已保证过，评测独立复算是
   双保险，也是外轨报告的"下限不被内容华丽掩盖"的那道闸。
2. **状态机行为断言**（behavior 维）——澄清流程、run_done 恰好一次、
   seq 连续、恢复接管（attempt/lease 直读 DB，评测 harness 是特权操作者
   视图；产品 API 保持用户最小面）。
3. **过程指标**只记录不判分（成本/时长/缓存命中随 artifact 落盘）。

轨迹用于归因与行为断言，**绝不按"工具调用顺序"给报告质量打分**。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from evals.schemas import Criterion, CriterionResult

CITATION_MARKER = re.compile(r"\[来源(\d+)\]")
TERMINAL_STATUSES = ("completed", "failed", "cancelled")


@dataclass
class Artifact:
    """一次 run 的评测素材：产品 API 视角 + DB 特权视角 + runner 控制面回执。"""

    case_id: str
    attempt: int
    run_id: str
    detail: dict[str, Any]  # GET /api/runs/{id} 返回
    events: list[dict[str, Any]] = field(default_factory=list)  # run_events.record
    run_row: dict[str, Any] = field(default_factory=dict)  # runs 表行（attempt 等）
    control: dict[str, Any] = field(default_factory=dict)  # runner 侧记录（如重复回答的 HTTP 码）

    @property
    def report(self) -> str:
        return str(self.detail.get("report_markdown") or "")

    @property
    def citations(self) -> list[dict[str, Any]]:
        return list(self.detail.get("citations") or [])

    def status_sequence(self) -> list[str]:
        return [
            str((event.get("payload") or {}).get("status", ""))
            for event in self.events
            if event.get("event_type") == "run_status"
        ]

    def events_of(self, event_type: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("event_type") == event_type]


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
    """门：正文引用的每个 [来源N] 都有出处，且出处条目自带 url+quote。"""
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
    dones = artifact.events_of("run_done")
    if len(dones) == 1:
        return _result(artifact, criterion, "yes")
    return _result(artifact, criterion, "no", f"run_done 帧数={len(dones)}")


def check_seq_continuous(artifact: Artifact, criterion: Criterion) -> CriterionResult:
    seqs = [int(e["seq"]) for e in artifact.events if "seq" in e]
    if seqs and seqs == list(range(1, len(seqs) + 1)):
        return _result(artifact, criterion, "yes", f"seq 1..{len(seqs)} 连续")
    return _result(artifact, criterion, "no", "seq 有洞/乱序（跨世续号回归）")


def check_clarify_flow(artifact: Artifact, criterion: Criterion) -> CriterionResult:
    """等待回答 → 提交 → 排队 → 续跑，且澄清不重跑 Router。"""
    statuses = artifact.status_sequence()
    if "awaiting_input" not in statuses:
        return _result(artifact, criterion, "no", "从未进入 awaiting_input")
    clarification = artifact.events_of("clarification_requested")
    if not clarification:
        return _result(artifact, criterion, "no", "缺 clarification_requested 事件")
    payload = clarification[-1].get("payload") or {}
    options = payload.get("options")
    if isinstance(options, list) and len(options) > 3:
        return _result(artifact, criterion, "no", f"选项 {len(options)} 个 > 3")
    after = statuses[statuses.index("awaiting_input") + 1 :]
    if "queued" not in after or "running" not in after:
        return _result(artifact, criterion, "no", "回答后未走完 queued→running")
    router_starts = [e for e in artifact.events_of("node_started") if e.get("node") == "router"]
    if len(router_starts) != 1:
        return _result(artifact, criterion, "no", f"Router 执行 {len(router_starts)} 次（恢复重跑）")
    return _result(artifact, criterion, "yes")


def check_duplicate_answer_rejected(artifact: Artifact, criterion: Criterion) -> CriterionResult:
    code = artifact.control.get("second_resume_status")
    if code == 409:
        return _result(artifact, criterion, "yes")
    return _result(artifact, criterion, "no", f"重复回答返回 {code}，期待 409")


def check_lease_takeover(artifact: Artifact, criterion: Criterion) -> CriterionResult:
    """强杀接管：attempt≥2、done 恰一次、seq 连续（跨世续号的现场证明）。"""
    attempt = int(artifact.run_row.get("attempt") or 0)
    if attempt < 2:
        return _result(artifact, criterion, "unknown", f"attempt={attempt}，故障注入未生效？")
    dones = len(artifact.events_of("run_done"))
    seqs = [int(e["seq"]) for e in artifact.events if "seq" in e]
    ok = dones == 1 and seqs == list(range(1, len(seqs) + 1))
    return _result(
        artifact,
        criterion,
        "yes" if ok else "no",
        f"attempt={attempt} done={dones} seq={'连续' if seqs == list(range(1, len(seqs) + 1)) else '断裂'}",
    )


def check_graceful_interrupt_resume(artifact: Artifact, criterion: Criterion) -> CriterionResult:
    statuses = artifact.status_sequence()
    if "resuming" not in statuses:
        return _result(artifact, criterion, "no", "未见 resuming 播报")
    attempt = int(artifact.run_row.get("attempt") or 0)
    if attempt < 2:
        return _result(artifact, criterion, "unknown", f"attempt={attempt} 非接管产物")
    return _result(artifact, criterion, "yes", f"interrupted→resuming，attempt={attempt}")


def check_cache_reuse(artifact: Artifact, criterion: Criterion) -> CriterionResult:
    hits = int(artifact.detail.get("cache_hit_count") or 0)
    saved = int(artifact.detail.get("saved_external_request_count") or 0)
    if hits >= 1 and saved >= 1:
        return _result(artifact, criterion, "yes", f"命中 {hits} 次，省外部调用 {saved}")
    return _result(artifact, criterion, "no", f"cache_hit={hits} saved={saved}")


CHECKS: dict[str, Check] = {
    "citation_integrity": check_citation_integrity,
    "done_exactly_once": check_done_exactly_once,
    "seq_continuous": check_seq_continuous,
    "clarify_flow": check_clarify_flow,
    "duplicate_answer_rejected": check_duplicate_answer_rejected,
    "lease_takeover": check_lease_takeover,
    "graceful_interrupt_resume": check_graceful_interrupt_resume,
    "cache_reuse": check_cache_reuse,
}

# 每一轨每一份完成产物都自动附挂的门（不写进题集，避免 30 份重复）。
UNIVERSAL_GATES = ("citation_integrity", "done_exactly_once", "seq_continuous")


def score_artifact(
    artifact: Artifact,
    case: Any,  # EvalCase（避免循环导入用鸭子类型）
    *,
    gates: tuple[str, ...] = UNIVERSAL_GATES,
) -> list[CriterionResult]:
    results: list[CriterionResult] = []
    for name in gates:
        probe = Criterion(id=f"gate:{name}", dimension="grounding", text=name)
        results.append(CHECKS[name](artifact, probe))
    for criterion in case.deterministic_criteria():
        check = CHECKS.get(criterion.scorer.split(":", 1)[1])
        if check is None:
            raise KeyError(f"题 {case.case_id} 引用了未注册 check: {criterion.scorer}")
        results.append(check(artifact, criterion))
    return results


def process_metrics(artifact: Artifact) -> dict[str, Any]:
    """过程指标：tracked，不做 pass/fail。"""
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
        "estimated_cost_usd": detail.get("estimated_cost_usd"),
        "external_request_count": detail.get("external_request_count"),
        "cache_hit_count": detail.get("cache_hit_count"),
        "elapsed_ms": detail.get("elapsed_ms"),
    }
