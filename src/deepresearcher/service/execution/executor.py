"""Run 执行面：驱动单个 LangGraph Run 并收敛其持久化结果。

RunExecutor 不受理用户请求、不检查队列配额、不选择下一个 Run；
它只由 WorkerRuntime 在执行面调用。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from sqlalchemy import update

from deepresearcher.config import Settings
from deepresearcher.graph import build_graph
from deepresearcher.llm import classify_llm_error
from deepresearcher.observability import JsonlSink
from deepresearcher.observability.events.names import EventName
from deepresearcher.observability.tracing import ledger, spans
from deepresearcher.routing import PREVIEW_CHANNELS
from deepresearcher.schemas.limits import RUN_HEADLINE_MAX_CHARS
from deepresearcher.service.events.hub import RunEventHub
from deepresearcher.service.events.sinks import CompositeSink
from deepresearcher.service.execution.queue import RunWork
from deepresearcher.service.execution.usage import (
    CapacityGate,
    ProviderRateLimiter,
    RunUsageCallback,
    UsageBudgetExceeded,
    UsageRuntime,
    UsageStore,
    bind_usage_runtime,
    reset_usage_runtime,
)
from deepresearcher.service.persistence.models import Run
from deepresearcher.service.persistence.models import utcnow as _utcnow
from deepresearcher.service.persistence.provider_health import PostgresProviderHealth
from deepresearcher.service.preview.protocol import EphemeralEventBus
from deepresearcher.service.runs.transitions import (
    apply_transition,
    may_overwrite,
    side_effects,
    transition_for,
)
from deepresearcher.service.settings import ServiceConfig
from deepresearcher.tools.web.materials import ResearchMaterialStore

logger = logging.getLogger("deepresearcher.service.execution.executor")

_FLUSH_INTERVAL_SECONDS = 2.0


def prune_event_jsonl(events_dir: Path, retention_days: int) -> int:
    """按 mtime 删除超龄的 run 事件 JSONL,返回删除数。

    清理是尽力而为:单个文件出错跳过,绝不让运维杂务打断进程启动。
    retention_days<=0 表示关闭清理。
    """
    if retention_days <= 0 or not events_dir.is_dir():
        return 0
    cutoff = time.time() - retention_days * 86_400
    removed = 0
    for path in events_dir.glob("run-*.jsonl"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            continue
    return removed


# 账户级 LLM 不可用 → 面向用户的安全文案（不泄露内部错误串）。
_LLM_UNAVAILABLE_MESSAGE = {
    "invalid_key": "模型服务密钥无效，请检查配置后重试。",
    "insufficient_credit": "模型服务余额不足，请充值或更换密钥后重试。",
    "forbidden": "模型服务拒绝访问（权限不足或模型不可用）。",
    "rate_limited": "模型服务当前限流，请稍后重试。",
}


class RunExecutor:
    """执行一个已领取的 run；调度与所有权在本类之外。"""

    def __init__(
        self,
        *,
        settings: Settings,
        session_factory: Callable[[], Any],
        config: ServiceConfig,
        hub: RunEventHub,
        usage_store: UsageStore,
        llm_gate: CapacityGate,
        llm_rate_limiter: ProviderRateLimiter,
        http_client: Any,
        graph_factory: Callable[..., Any] = build_graph,
        checkpointer: Any = None,
        material_store: ResearchMaterialStore | None = None,
        ephemeral_bus: EphemeralEventBus | None = None,
    ) -> None:
        self._settings = settings
        self._session_factory = session_factory
        self._config = config
        self._hub = hub
        # 投递与 notify 只发生在 RunEventHub.flush:executor 必须复用协调层注入的
        # 同一实例，不设默认构造，避免出现绕过它的第二个 Hub。
        self._usage_store = usage_store
        self._llm_gate = llm_gate
        self._llm_rate_limiter = llm_rate_limiter
        self._http_client = http_client
        self._graph_factory = graph_factory
        self._checkpointer = checkpointer
        self._material_store = material_store
        self._ephemeral_bus = ephemeral_bus
        # 搜索提供方账户级健康跨 worker 共享（PG）；非 PG（测试/SQLite）用 SearchService 的内存默认。
        self._provider_health = (
            PostgresProviderHealth(session_factory)
            if config.database_url.startswith("postgresql")
            else None
        )
        # Langfuse 是镜像用的观测出口: callbacks 级挂入,失败或断网不得影响 run
        # (与预览总线同一合同)。三键缺任一 = 完全关闭;导入失败(未装 extra)同样降级关闭。
        self._langfuse_client: Any | None = None
        self._langfuse_handler: Callable[[], Any] | None = None
        self._langfuse_scope: Callable[..., Any] | None = None
        if config.langfuse_public_key and config.langfuse_secret_key and config.langfuse_base_url:
            try:
                from langfuse import Langfuse, propagate_attributes
                from langfuse.langchain import CallbackHandler

                self._langfuse_client = Langfuse(
                    public_key=config.langfuse_public_key,
                    secret_key=config.langfuse_secret_key,
                    base_url=config.langfuse_base_url,
                )
                public_key = config.langfuse_public_key
                self._langfuse_handler = lambda: CallbackHandler(public_key=public_key)
                self._langfuse_scope = lambda run_id, user_id: propagate_attributes(
                    session_id=run_id,
                    user_id=str(user_id),
                    tags=["deepresearcher"],
                    trace_name="research_run",
                )
            except Exception:  # noqa: BLE001 - 观测出口坏了不拦执行
                logger.warning("langfuse_init_failed", exc_info=True)
                self._langfuse_client = None
                self._langfuse_handler = None
                self._langfuse_scope = None
        self._shutdown_interrupts: set[str] = set()
        self._lost_leases: set[str] = set()
        self._cancellation_requests: set[str] = set()

    def mark_shutdown(self, run_ids: Iterable[str]) -> None:
        """在本地任务被取消前，把该取消标记为进程 shutdown。"""
        self._shutdown_interrupts.update(run_ids)

    def mark_lease_lost(self, run_id: str) -> None:
        """阻止已被取消的过期持有者再发布状态或 done 帧。"""
        self._lost_leases.add(run_id)

    def mark_cancellation_requested(self, run_id: str) -> None:
        """标记该取消由持久化的控制面意图驱动。"""
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
        """执行一个已领取的 run 直至终态或挂起。

        Worker 可能与受理该 Run 的 API 不在同一进程,执行面必须自行打开本地
        sink,不依赖 API 进程中的 hub.open()。

        参数:
            run_id/user_id/query: run 身份与研究问题。
            resume: 是否以恢复模式续跑。
            resume_input: 恢复输入（澄清回答等）。
            claim: 本 worker 的领取凭据；None 表示不经领取的直连路径。
        """
        self._hub.open(run_id)
        sinks: list[Any] = [self._hub]
        if self._config.jsonl_events:
            sinks.append(JsonlSink(self._config.service_log_dir / "events" / f"{run_id}.jsonl"))
        sink = CompositeSink(*sinks)
        ledger.attach_run_sink(run_id, sink)
        flusher = asyncio.create_task(self._periodic_flush(run_id))
        usage_token = bind_usage_runtime(
            UsageRuntime(run_id=run_id, store=self._usage_store, config=self._settings.llm)
        )
        suspended = False
        try:
            with spans.trace(
                "research_run",
                run_id=run_id,
                session_id=run_id,
                input={"query": query},
                metadata={
                    "attempt": claim.attempt if claim is not None else 0,
                    "resume": resume,
                    "worker": claim.lease_owner if claim is not None else "",
                },
            ):
                suspended = await self._execute_traced(
                    run_id,
                    user_id,
                    query,
                    sink=sink,
                    resume=resume,
                    resume_input=resume_input,
                    claim=claim,
                )
        except asyncio.CancelledError:
            if run_id not in self._shutdown_interrupts and run_id not in self._lost_leases:
                # shield:终态落库事务不再被二次 cancel 打断(shutdown 后紧跟
                # 取消的二次时序);本协程照常退出,写库由被 shield 的任务完成。
                try:
                    persisted = await asyncio.shield(
                        self.persist_status(
                            run_id,
                            status="cancelled",
                            terminal_reason="user_cancelled",
                            claim=claim,
                        )
                    )
                except asyncio.CancelledError:
                    raise
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
        except Exception as exc:  # noqa: BLE001 - 后台执行必须自收口
            # 冒泡的原始 LLM 异常若属账户级不可用（key/余额/硬限流），按类型化原因收口，
            # 让前端能渲染明确错误，而不是笼统"运行失败"。
            llm_code = classify_llm_error(exc)
            if llm_code is not None:
                persisted = await self._fail_llm_unavailable(run_id, llm_code, claim)
            else:
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
                self._hub.close(run_id)
            self._lost_leases.discard(run_id)
            self._cancellation_requests.discard(run_id)
            self._shutdown_interrupts.discard(run_id)
            ledger.detach_run_sink(run_id)
            reset_usage_runtime(usage_token)
            if self._langfuse_client is not None:
                try:
                    self._langfuse_client.flush()
                except Exception:  # noqa: BLE001 - 观测出口不许反噬收尾
                    logger.debug("langfuse_flush_failed run_id=%s", run_id, exc_info=True)

    def langfuse_enabled(self) -> bool:
        return self._langfuse_client is not None

    def shutdown_langfuse(self) -> None:
        """进程退出前把缓冲的 trace 刷出;失败静默(与预览面同一合同)。"""
        if self._langfuse_client is not None:
            try:
                self._langfuse_client.shutdown()
            except Exception:  # noqa: BLE001
                logger.debug("langfuse_shutdown_failed", exc_info=True)
            self._langfuse_client = None

    async def _execute_traced(
        self,
        run_id: str,
        user_id: int,
        query: str,
        *,
        sink: CompositeSink,
        resume: bool,
        resume_input: Any,
        claim: RunWork | None,
    ) -> bool:
        """在 run 的 TraceContext 绑定期间执行图。"""
        # 系统重启续跑已经播报 resuming，保持该状态直到后续阶段事件；
        # 人工澄清恢复则必须在 Worker 真正 claim 后从 queued 切到 running。
        announce_running = not resume or resume_input is not None
        if not await self._confirm_running(
            run_id,
            announce_running=announce_running,
            claim=claim,
        ):
            if run_id not in self._cancellation_requests:
                self.mark_lease_lost(run_id)
            return False
        graph = self._graph_factory(
            settings=self._settings,
            event_sink=sink,
            http_client=self._http_client,
            checkpointer=self._checkpointer,
            material_store=self._material_store,
            provider_health=self._provider_health,
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
        callbacks: list[Any] = [callback]
        if self._langfuse_handler is not None:
            callbacks.append(self._langfuse_handler())
        # propagate_attributes 必须在图执行期间生效,langchain callbacks 生成的
        # trace 才会带上 session/user 归属。
        scope = (
            self._langfuse_scope(run_id=run_id, user_id=user_id)
            if self._langfuse_scope is not None
            else nullcontext()
        )
        with scope:
            result, interruption = await self._run_graph(run_id, graph, inputs, callbacks=callbacks)
        if interruption is not None:
            spans.record_output({"phase": "awaiting_input"})
            if not await self._persist_awaiting_input(run_id, claim=claim):
                self.mark_lease_lost(run_id)
                return False
            self._hub.write(
                {
                    "run_id": run_id,
                    "event_type": EventName.CLARIFICATION_REQUESTED,
                    "payload": interruption,
                }
            )
            await self.publish_status(run_id, "awaiting_input")
            return True
        lifecycle = result.get("run") or {}
        spans.record_output(
            {
                "phase": _field(lifecycle, "phase", ""),
                "terminal_reason": _field(lifecycle, "terminal_reason", ""),
                "answer_mode": result.get("answer_mode") or "",
                "report_chars": len(str(result.get("report") or "")),
                "evidence_count": int(
                    result.get("evidence_count") or len(result.get("evidences") or [])
                ),
                "source_count": int(
                    result.get("source_count") or len(result.get("source_refs") or [])
                ),
            }
        )
        if not await self._persist_terminal(run_id, result, claim=claim):
            self.mark_lease_lost(run_id)
        return False

    async def _run_graph(
        self, run_id: str, graph: Any, inputs: Any, *, callbacks: list[Any] | None = None
    ) -> tuple[dict, dict | None]:
        """从图流式输出根状态、中断与安全的 token 预览。"""
        final: dict = {}
        interruption: dict | None = None
        headline_persisted = False
        async for namespace, mode, chunk in graph.astream(
            inputs,
            config={"configurable": {"thread_id": run_id}, "callbacks": callbacks or []},
            stream_mode=["values", "messages", "updates"],
            subgraphs=True,
        ):
            if mode == "values":
                if namespace == () and isinstance(chunk, dict):
                    final = chunk
                    headline = str(chunk.get("run_headline") or "")
                    if headline and not headline_persisted:
                        headline_persisted = True
                        if await self._persist_headline(run_id, headline):
                            # 落库成功即推帧:进行中的头部当场从长问题收敛为短题。
                            # 根 values 块在 clarify 那一超步后即携带 run_headline,
                            # 故本帧在澄清出口、而非 run 结束时发出。
                            self._hub.write(
                                {
                                    "run_id": run_id,
                                    "event_type": EventName.RUN_HEADLINE_UPDATED,
                                    "payload": {"headline": headline[:RUN_HEADLINE_MAX_CHARS]},
                                }
                            )
                            # 立即 flush 越过 ≤2s 周期:落库→写库→NOTIFY 门铃一次走完。
                            await self._hub.flush(run_id)
                continue
            if mode == "updates" and isinstance(chunk, dict):
                interrupts = chunk.get("__interrupt__") or ()
                if interrupts:
                    value = getattr(interrupts[0], "value", None)
                    if isinstance(value, dict):
                        interruption = value
                continue
            if mode == "messages":
                await self._publish_message_preview(run_id, namespace, chunk)
        return final, interruption

    async def _publish_message_preview(self, run_id: str, namespace: Any, chunk: Any) -> None:
        try:
            message, _metadata = chunk
        except (TypeError, ValueError):
            return
        # langchain 1.x 的流式片段 .type 为 "AIMessageChunk" 而非 "ai"——只认 "ai"
        # 会把全部 token 静默丢掉(流式失踪事故的根因)。
        if getattr(message, "type", None) not in {"ai", "AIMessageChunk"}:
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
            event = {
                "run_id": run_id,
                "event_type": EventName.TEXT_DELTA,
                "payload": {"channel": channel, "text": text[:200]},
            }
            try:
                # 预览只走注入的总线这一条路径(生产 Redis,单栈 harness 注入 Local bus)。
                # 已提交帧从不经过这里——SSE 从 DB tail 取。
                if self._ephemeral_bus is not None:
                    await self._ephemeral_bus.publish(run_id, event)
            except Exception:  # noqa: BLE001 - 预览通道不反噬运行
                logger.debug("delta_publish_failed run_id=%s", run_id, exc_info=True)

    @staticmethod
    def _preview_channel(namespace: Any) -> str | None:
        if not isinstance(namespace, tuple) or len(namespace) != 1:
            return None
        head = str(namespace[0]).split(":", 1)[0]
        return head if head in PREVIEW_CHANNELS else None

    async def _confirm_running(
        self,
        run_id: str,
        *,
        announce_running: bool = True,
        claim: RunWork | None = None,
    ) -> bool:
        """确认该行处于 running 且仍由本次执行持有,否则返回 False。

        claim 路径:领取的 UPDATE 已把行置为 running,这里只核实租约与 attempt;
        直连路径(claim=None):按 mark_running 迁移把行写进 running。
        """
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
            elif run is not None and run.status in transition_for("mark_running").sources:
                apply_transition(run, "mark_running", now=_utcnow())
                if run.started_at is None:
                    run.started_at = _utcnow()
                transitioned = True
                await session.commit()
        if transitioned and announce_running:
            await self.publish_status(run_id, "running")
        return transitioned

    async def _persist_headline(self, run_id: str, headline: str) -> bool:
        """澄清完成即把历史标题写进行:列表行从原 query 收敛为浓缩标题。

        只写未落过 headline 的行——首个有效标题为准,重复到达幂等丢弃。
        返回是否由本次调用写入(决定要不要向 SSE 推更新帧)。
        """
        async with self._session_factory() as session:
            result = await session.execute(
                update(Run)
                .where(Run.id == run_id, Run.headline.is_(None))
                .values(headline=headline[:RUN_HEADLINE_MAX_CHARS])
            )
            await session.commit()
            return result.rowcount == 1

    async def _persist_awaiting_input(self, run_id: str, *, claim: RunWork | None = None) -> bool:
        if claim is not None:
            async with self._session_factory() as session:
                # WHERE 固定 running:只有当前 claim 持有者可写(所有权检查),
                # 故意严于命令表来源并集,不查表。
                result = await session.execute(
                    update(Run)
                    .where(
                        Run.id == run_id,
                        Run.status == "running",
                        Run.lease_owner == claim.lease_owner,
                        Run.attempt == claim.attempt,
                    )
                    .values(**side_effects("awaiting_input", now=_utcnow()))
                )
                await session.commit()
                return result.rowcount == 1
        async with self._session_factory() as session:
            run = await session.get(Run, run_id)
            # 对账写(不经领取的直连路径):只受"不覆盖终态"约束,来源不必是
            # running——自愈正是为残余竞态窗口准备的,故不走 await_input 迁移。
            if run is None or not may_overwrite(run.status):
                return False
            for key, value in side_effects("awaiting_input", now=_utcnow()).items():
                setattr(run, key, value)
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
            title=result.get("report_title") or None,
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
        # 对账写:结构副作用查表,站点字段(extra)覆写其上;来源纪律只有"不覆盖终态"。
        """把生命周期事实写成行状态，并落对应事件。

        参数:
            run_id: 目标 run。
            status: 目标状态。
            claim: 提供时用 owner+attempt CAS 写，防止过期持有者覆盖。
            **extra: 附加列；None 值跳过。

        返回:
            bool: 是否写入成功（CAS 未命中或行已终态为 False）。
        """
        values = side_effects(status, now=_utcnow())
        values.update({key: value for key, value in extra.items() if value is not None})
        if claim is not None:
            # 同上:WHERE 固定 running 是所有权检查,严于迁移表来源并集,不查表。
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
            if run is None or not may_overwrite(run.status):
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
        await self._hub.flush(run_id)

    async def publish_status(self, run_id: str, status: str) -> None:
        await self._hub.publish_status(run_id, status)

    async def publish_done(self, run_id: str) -> None:
        await self._hub.publish_done(run_id)

    async def _fail_llm_unavailable(
        self, run_id: str, user_code: str, claim: RunWork | None
    ) -> bool:
        """LLM 网关账户级不可用：类型化 terminal_reason + 安全文案，快速失败。"""
        logger.warning("research_llm_unavailable run_id=%s code=%s", run_id, user_code)
        return await self.persist_status(
            run_id,
            status="failed",
            terminal_reason=f"llm_unavailable:{user_code}",
            error_message=_LLM_UNAVAILABLE_MESSAGE.get(
                user_code, "模型服务暂时不可用，请稍后重试。"
            ),
            claim=claim,
        )


def _field(container: Any, key: str, default: Any = None) -> Any:
    """从 checkpoint 恢复的 dict 或 Pydantic/领域模型上读取字段。"""
    if isinstance(container, dict):
        return container.get(key, default)
    return getattr(container, key, default)
