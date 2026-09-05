"""Run 执行面：驱动单个 LangGraph Run 并收敛其持久化结果。

RunExecutor 不受理用户请求、不检查队列配额、不选择下一个 Run；
它只由 WorkerCoordinator 在执行面调用。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import update

from deepsearch_agent.config import Settings
from deepsearch_agent.observability import JsonlSink
from deepsearch_agent.orchestration.graph import build_graph
from deepsearch_agent.service.event_publisher import RunEventPublisher
from deepsearch_agent.service.events.store import RunEventStore
from deepsearch_agent.service.events.stream import CompositeSink, FanoutSink
from deepsearch_agent.service.models import Run
from deepsearch_agent.service.runs.queue import RunWork
from deepsearch_agent.service.settings import ServiceConfig
from deepsearch_agent.service.usage import (
    CapacityGate,
    ProviderRateLimiter,
    RunUsageCallback,
    UsageBudgetExceeded,
    UsageRuntime,
    UsageStore,
    bind_usage_runtime,
    reset_usage_runtime,
)
from deepsearch_agent.tools.cache import ToolCache

logger = logging.getLogger("deepsearch_agent.service.executor")

TERMINAL_STATUSES = ("completed", "failed", "cancelled")
_FLUSH_INTERVAL_SECONDS = 2.0


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RunExecutor:
    """Execute one claimed run; scheduling and ownership remain outside this class."""

    def __init__(
        self,
        *,
        settings: Settings,
        session_factory: Callable[[], Any],
        config: ServiceConfig,
        fanout: FanoutSink,
        event_store: RunEventStore,
        event_publisher: RunEventPublisher | None = None,
        usage_store: UsageStore,
        llm_gate: CapacityGate,
        llm_rate_limiter: ProviderRateLimiter,
        http_client: Any,
        graph_factory: Callable[..., Any] = build_graph,
        checkpointer: Any = None,
        tool_cache: ToolCache | None = None,
    ) -> None:
        self._settings = settings
        self._session_factory = session_factory
        self._config = config
        self._fanout = fanout
        self._event_store = event_store
        self._event_publisher = event_publisher or RunEventPublisher(
            session_factory=session_factory,
            fanout=fanout,
            event_store=event_store,
        )
        self._usage_store = usage_store
        self._llm_gate = llm_gate
        self._llm_rate_limiter = llm_rate_limiter
        self._http_client = http_client
        self._graph_factory = graph_factory
        self._checkpointer = checkpointer
        self._tool_cache = tool_cache
        self._shutdown_interrupts: set[str] = set()
        self._lost_leases: set[str] = set()
        self._cancellation_requests: set[str] = set()

    def mark_shutdown(self, run_ids: Iterable[str]) -> None:
        """Mark cancellation as process shutdown before local tasks are cancelled."""
        self._shutdown_interrupts.update(run_ids)

    def mark_lease_lost(self, run_id: str) -> None:
        """Prevent a cancelled stale owner from publishing state or a done frame."""
        self._lost_leases.add(run_id)

    def mark_cancellation_requested(self, run_id: str) -> None:
        """Identify cancellation driven by the durable control-plane intent."""
        self._cancellation_requests.add(run_id)

    async def execute(
        self,
        run_id: str,
        user_id: int,
        query: str,
        *,
        resume: bool = False,
        resume_input: Any = None,
        claim: RunWork | None = None,
    ) -> None:
        # Worker 可能与受理该 Run 的 API 不在同一进程；执行面必须
        # 自行打开本地 sink，不能依赖 API 进程中的 fanout.open().
        self._fanout.open(run_id)
        sinks: list[Any] = [self._fanout]
        if self._config.jsonl_events:
            sinks.append(JsonlSink(self._config.service_log_dir / "events" / f"{run_id}.jsonl"))
        sink = CompositeSink(*sinks)
        flusher = asyncio.create_task(self._periodic_flush(run_id))
        usage_token = bind_usage_runtime(
            UsageRuntime(run_id=run_id, store=self._usage_store, config=self._settings.llm)
        )
        suspended = False
        try:
            if not await self._mark_running(run_id, resume=resume, claim=claim):
                if run_id not in self._cancellation_requests:
                    self.mark_lease_lost(run_id)
                return
            graph = self._graph_factory(
                settings=self._settings_for(user_id),
                event_sink=sink,
                http_client=self._http_client,
                checkpointer=self._checkpointer,
                tool_cache=self._tool_cache,
            )
            # resume 时传 None 或 Command，由 checkpointer + thread_id 从断点继续。
            inputs = (
                resume_input if resume else {"query": query, "run_id": run_id, "session_id": run_id}
            )
            callback = RunUsageCallback(
                run_id=run_id,
                store=self._usage_store,
                gate=self._llm_gate,
                rate_limiter=self._llm_rate_limiter,
                config=self._settings.llm,
            )
            result, interruption = await self._run_graph(
                run_id, graph, inputs, callbacks=[callback]
            )
            if interruption is not None:
                if not await self._persist_awaiting_input(run_id, claim=claim):
                    self.mark_lease_lost(run_id)
                    return
                suspended = True
                self._fanout.write(
                    {
                        "run_id": run_id,
                        "event_type": "clarification_requested",
                        "payload": interruption,
                    }
                )
                await self.publish_status(run_id, "awaiting_input")
            else:
                if not await self._persist_terminal(run_id, result, claim=claim):
                    self.mark_lease_lost(run_id)
        except asyncio.CancelledError:
            if run_id not in self._shutdown_interrupts and run_id not in self._lost_leases:
                persisted = await self.persist_status(
                    run_id,
                    status="cancelled",
                    terminal_reason="user_cancelled",
                    claim=claim,
                )
                if claim is not None and not persisted:
                    self.mark_lease_lost(run_id)
        except UsageBudgetExceeded as exc:
            logger.info("research_budget_exhausted run_id=%s reason=%s", run_id, exc)
            persisted = await self.persist_status(
                run_id,
                status="failed",
                terminal_reason="budget_exhausted",
                error_message="本次研究已达用量上限，请调整配额后重试。",
                claim=claim,
            )
            if claim is not None and not persisted:
                self.mark_lease_lost(run_id)
        except Exception:  # noqa: BLE001 - 后台执行必须自收口
            logger.exception("research_run_failed run_id=%s", run_id)
            persisted = await self.persist_status(
                run_id,
                status="failed",
                terminal_reason="run_exception",
                error_message="运行执行失败，请稍后重试或重新发起。",
                claim=claim,
            )
            if claim is not None and not persisted:
                self.mark_lease_lost(run_id)
        finally:
            flusher.cancel()
            await asyncio.gather(flusher, return_exceptions=True)
            if (
                run_id not in self._shutdown_interrupts
                and run_id not in self._lost_leases
                and not suspended
            ):
                await self.publish_done(run_id)
            await self.flush_events(run_id)
            if run_id not in self._lost_leases:
                self._fanout.close(run_id)
            self._lost_leases.discard(run_id)
            self._cancellation_requests.discard(run_id)
            reset_usage_runtime(usage_token)

    def _settings_for(self, user_id: int) -> Settings:
        """BYO-keys 预留缝：未来可按用户返回 replace(...) 的 Settings。"""
        return self._settings

    async def _run_graph(
        self, run_id: str, graph: Any, inputs: Any, *, callbacks: list[Any] | None = None
    ) -> tuple[dict, dict | None]:
        """Stream root state, interrupts and safe token previews from the graph."""
        final: dict = {}
        interruption: dict | None = None
        async for namespace, mode, chunk in graph.astream(
            inputs,
            config={"configurable": {"thread_id": run_id}, "callbacks": callbacks or []},
            stream_mode=["values", "messages", "updates"],
            subgraphs=True,
        ):
            if mode == "values":
                if namespace == () and isinstance(chunk, dict):
                    final = chunk
                continue
            if mode == "updates" and isinstance(chunk, dict):
                interrupts = chunk.get("__interrupt__") or ()
                if interrupts:
                    value = getattr(interrupts[0], "value", None)
                    if isinstance(value, dict):
                        interruption = value
                continue
            if mode == "messages":
                self._publish_message_preview(run_id, namespace, chunk)
        return final, interruption

    def _publish_message_preview(self, run_id: str, namespace: Any, chunk: Any) -> None:
        try:
            message, _metadata = chunk
        except (TypeError, ValueError):
            return
        if getattr(message, "type", None) != "ai":
            return
        channel = self._preview_channel(namespace)
        if channel is None:
            return
        for block in getattr(message, "content_blocks", None) or []:
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            text = str(block.get("text") or "")
            if not text:
                continue
            try:
                self._fanout.publish_ephemeral(
                    run_id,
                    {
                        "run_id": run_id,
                        "event_type": "text_delta",
                        "payload": {"channel": channel, "text": text[:200]},
                    },
                )
            except Exception:  # noqa: BLE001 - 预览通道不反噬运行
                logger.debug("delta_publish_failed run_id=%s", run_id, exc_info=True)

    @staticmethod
    def _preview_channel(namespace: Any) -> str | None:
        if not isinstance(namespace, tuple) or len(namespace) != 1:
            return None
        head = str(namespace[0]).split(":", 1)[0]
        return head if head == "supervisor" else None

    async def _mark_running(
        self, run_id: str, *, resume: bool = False, claim: RunWork | None = None
    ) -> bool:
        transitioned = False
        async with self._session_factory() as session:
            run = await session.get(Run, run_id)
            if claim is not None:
                if (
                    run is None
                    or run.status != "running"
                    or run.lease_owner != claim.lease_owner
                    or run.attempt != claim.attempt
                ):
                    return False
                transitioned = True
            elif run is not None and run.status in ("queued", "interrupted", "awaiting_input"):
                run.status = "running"
                if run.started_at is None:
                    run.started_at = _utcnow()
                transitioned = True
                await session.commit()
        if transitioned and not resume:
            await self.publish_status(run_id, "running")
        return transitioned

    async def persist_interrupted(self, run_id: str) -> bool:
        """Persist a resumable shutdown without publishing the terminal done frame."""
        self._shutdown_interrupts.add(run_id)
        async with self._session_factory() as session:
            run = await session.get(Run, run_id)
            if run is None or run.status in TERMINAL_STATUSES:
                self._shutdown_interrupts.discard(run_id)
                return False
            run.status = "interrupted"
            run.terminal_reason = "server_shutdown"
            run.error_message = None
            run.finished_at = None
            await session.commit()
        return True

    async def _persist_awaiting_input(self, run_id: str, *, claim: RunWork | None = None) -> bool:
        if claim is not None:
            async with self._session_factory() as session:
                result = await session.execute(
                    update(Run)
                    .where(
                        Run.id == run_id,
                        Run.status == "running",
                        Run.lease_owner == claim.lease_owner,
                        Run.attempt == claim.attempt,
                    )
                    .values(
                        status="awaiting_input",
                        terminal_reason=None,
                        error_message=None,
                        finished_at=None,
                        lease_owner=None,
                        lease_expires_at=None,
                        resume_payload=None,
                    )
                )
                await session.commit()
                return result.rowcount == 1
        async with self._session_factory() as session:
            run = await session.get(Run, run_id)
            if run is None or run.status in TERMINAL_STATUSES:
                return False
            run.status = "awaiting_input"
            run.terminal_reason = None
            run.error_message = None
            run.finished_at = None
            await session.commit()
            return True

    async def _persist_terminal(
        self, run_id: str, result: dict, *, claim: RunWork | None = None
    ) -> bool:
        lifecycle = result.get("run") or {}
        phase = _field(lifecycle, "phase", "")
        status = "completed" if phase == "completed" else "failed"
        error = _field(lifecycle, "error", None)
        citations = [
            item.model_dump(exclude_none=True) if hasattr(item, "model_dump") else dict(item)
            for item in (result.get("citations") or [])
        ]
        return await self.persist_status(
            run_id,
            status=status,
            terminal_reason=_field(lifecycle, "terminal_reason", "") or None,
            answer_mode=result.get("answer_mode"),
            report_markdown=result.get("report") or None,
            citations_json=citations,
            evidence_count=int(result.get("evidence_count") or len(result.get("evidences") or [])),
            source_count=int(result.get("source_count") or len(result.get("source_refs") or [])),
            error_message=(
                f"阶段 {_field(error, 'stage', 'unknown')} 执行失败，请稍后重试或重新发起。"
                if error
                else None
            ),
            claim=claim,
        )

    async def persist_status(
        self, run_id: str, *, status: str, claim: RunWork | None = None, **extra: Any
    ) -> bool:
        values = {"status": status, "finished_at": _utcnow()}
        values.update({key: value for key, value in extra.items() if value is not None})
        if status in TERMINAL_STATUSES:
            values.update(lease_owner=None, lease_expires_at=None, resume_payload=None)
        if claim is not None:
            async with self._session_factory() as session:
                result = await session.execute(
                    update(Run)
                    .where(
                        Run.id == run_id,
                        Run.status == "running",
                        Run.lease_owner == claim.lease_owner,
                        Run.attempt == claim.attempt,
                    )
                    .values(**values)
                )
                await session.commit()
                return result.rowcount == 1
        async with self._session_factory() as session:
            run = await session.get(Run, run_id)
            if run is None or run.status in TERMINAL_STATUSES:
                return False
            for key, value in values.items():
                setattr(run, key, value)
            await session.commit()
            return True

    async def _periodic_flush(self, run_id: str) -> None:
        try:
            while True:
                await asyncio.sleep(_FLUSH_INTERVAL_SECONDS)
                await self.flush_events(run_id)
        except asyncio.CancelledError:
            return

    async def flush_events(self, run_id: str) -> None:
        await self._event_publisher.flush(run_id)

    async def publish_status(self, run_id: str, status: str) -> None:
        await self._event_publisher.publish_status(run_id, status)

    async def publish_done(self, run_id: str) -> None:
        await self._event_publisher.publish_done(run_id)


def _field(container: Any, key: str, default: Any = None) -> Any:
    """Read a field from checkpoint-restored dicts or Pydantic/domain models."""
    if isinstance(container, dict):
        return container.get(key, default)
    return getattr(container, key, default)
