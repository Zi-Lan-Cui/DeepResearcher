"""Run 执行层：驱动单个 LangGraph Run 并写入其持久化结果。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from dataclasses import replace
from typing import Any

from sqlalchemy import select, update

from deepresearcher.config import Settings
from deepresearcher.graph import build_graph
from deepresearcher.llm import classify_llm_error
from deepresearcher.observability import JsonlSink
from deepresearcher.observability.events.names import EventName
from deepresearcher.observability.tracing import ledger, spans
from deepresearcher.routing import PREVIEW_CHANNELS
from deepresearcher.schemas.limits import RUN_HEADLINE_MAX_CHARS
from deepresearcher.service.events.buffer import RunEventBuffer
from deepresearcher.service.events.sinks import CompositeSink
from deepresearcher.service.execution.limits import CapacityGate, ProviderRateLimiter
from deepresearcher.service.execution.provider_health import ProviderHealthStore
from deepresearcher.service.execution.usage import RunUsageCallback, UsageStore
from deepresearcher.service.preferences import apply_preferences
from deepresearcher.service.preview.protocol import EphemeralEventBus
from deepresearcher.service.runs.store import RunWork, TerminalState
from deepresearcher.service.runs.transitions import side_effects
from deepresearcher.service.settings import ServiceConfig
from deepresearcher.service.storage.tables import Run
from deepresearcher.service.storage.tables import utcnow as _utcnow
from deepresearcher.tools.web.materials import ResearchMaterialStore
from deepresearcher.usage_runtime import (
    UsageBudgetExceeded,
    UsageRuntime,
    bind_usage_runtime,
    reset_usage_runtime,
)

logger = logging.getLogger("deepresearcher.service.execution.executor")

# 事件刷新间隔
_FLUSH_INTERVAL_SECONDS = 0.2

# 写入失败后的退避上限
_FLUSH_BACKOFF_MAX_SECONDS = 5.0


# LLM 账户级不可用时面向用户的安全文案，不含内部错误串。
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
        buffer: RunEventBuffer,
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
        self._buffer = buffer
        # 写库与 notify 只发生在 RunEventBuffer.flush：executor 必须复用协调层注入的
        # 同一实例，不设默认构造，避免出现绕过该写入点的第二个 Buffer。
        self._usage_store = usage_store
        self._llm_gate = llm_gate
        self._llm_rate_limiter = llm_rate_limiter
        self._http_client = http_client
        self._graph_factory = graph_factory
        self._checkpointer = checkpointer
        self._material_store = material_store
        self._ephemeral_bus = ephemeral_bus
        # 搜索提供方账户级健康跨 worker 共享。
        self._provider_health = ProviderHealthStore(session_factory)
        # langfuse 为可选，缺少配置或初始化失败时不影响执行。
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

    def mark_shutdown(self, run_ids: Iterable[str]) -> None:
        """在本地任务被取消前，把该取消标记为进程 shutdown。"""
        self._shutdown_interrupts.update(run_ids)

    def mark_lease_lost(self, run_id: str) -> None:
        """租约判定丢失后，本持有者停止为该 run 发布状态与 done 帧。"""
        self._lost_leases.add(run_id)

    def shutdown_langfuse(self) -> None:
        """进程退出前把缓冲的 trace 刷出；失败静默（与预览通道同一约定）。"""
        if self._langfuse_client is not None:
            try:
                self._langfuse_client.shutdown()
            except Exception:  # noqa: BLE001
                logger.debug("langfuse_shutdown_failed", exc_info=True)
            self._langfuse_client = None

    async def _has_checkpoint(self, run_id: str) -> bool:
        """这一行有没有可续跑的断点；没有 checkpointer 时一律视为没有。"""
        if self._checkpointer is None:
            return False
        tuple_ = await self._checkpointer.aget_tuple(
            {"configurable": {"thread_id": run_id, "checkpoint_ns": ""}}
        )
        return tuple_ is not None

    async def execute(self, work: RunWork, *, resume_input: Any = None) -> None:
        """执行一条已接收的分派，直至终态或挂起。

        参数：
            work: 接收凭据
            resume_input: 恢复输入，仅续跑时使用
        """
        run_id = work.run_id
        self._buffer.open(run_id)
        sinks: list[Any] = [self._buffer]
        if self._config.jsonl_events:
            sinks.append(JsonlSink(self._config.service_log_dir / "events" / f"{run_id}.jsonl"))
        sink = CompositeSink(*sinks)
        ledger.attach_run_sink(run_id, sink)
        flusher = asyncio.create_task(self._periodic_flush(run_id))
        usage_token = bind_usage_runtime(
            UsageRuntime(run_id=run_id, store=self._usage_store, config=self._settings.llm)
        )
        # 终态写入后取回的那一行；None 表示没有写成终态（挂起、丢权或写失败），不发 done。
        written: TerminalState | None = None
        try:
            with spans.trace(
                "research_run",
                run_id=run_id,
                session_id=run_id,
                input={"query": work.query},
                metadata={
                    "fence_token": work.fence_token,
                    "resume": work.resume,
                    "worker": work.lease_owner or "",
                },
            ):
                written = await self._execute_traced(work, sink=sink, resume_input=resume_input)
        except asyncio.CancelledError:
            if run_id in self._shutdown_interrupts:
                # 关机取消没有终态可写，由消费侧把行置为 interrupted 再让位。必须向上抛：
                # 消费侧据「执行是否正常返回」决定要不要确认消息，吞掉就等于确认掉，
                # 这条行再没有消息指向它，永久停在 interrupted。
                raise
            if run_id not in self._lost_leases:
                # 用 shield 保护终态写入数据库事务不被随后的第二次 cancel 打断
                # （shutdown 后任务取消会再次到来）；本协程照常退出，写库由被 shield 的任务完成。
                try:
                    written = await asyncio.shield(
                        self.persist_status(
                            work, status="cancelled", terminal_reason="user_cancelled"
                        )
                    )
                except asyncio.CancelledError:
                    raise
                if written is None:
                    self.mark_lease_lost(run_id)
        except UsageBudgetExceeded as exc:
            logger.info("research_budget_exhausted run_id=%s reason=%s", run_id, exc)
            written = await self.persist_status(
                work,
                status="failed",
                terminal_reason="budget_exhausted",
                error_message="本次研究已达用量上限，请调整配额后重试。",
            )
            if written is None:
                self.mark_lease_lost(run_id)
        except Exception as exc:  # noqa: BLE001 - 后台执行必须自行收尾
            # 向上传播的原始 LLM 异常若属账户级不可用（key/余额/硬限流），按类型化原因终止，
            llm_code = classify_llm_error(exc)
            if llm_code is not None:
                written = await self._fail_llm_unavailable(work, llm_code)
            else:
                logger.exception("research_run_failed run_id=%s", run_id)
                written = await self.persist_status(
                    work,
                    status="failed",
                    terminal_reason="run_exception",
                    error_message="运行执行失败，请稍后重试或重新发起。",
                )
            if written is None:
                self.mark_lease_lost(run_id)
        finally:
            flusher.cancel()
            await asyncio.gather(flusher, return_exceptions=True)
            if (
                written is not None
                and run_id not in self._shutdown_interrupts
                and run_id not in self._lost_leases
            ):
                self._buffer.write_done(
                    run_id,
                    status=written.status,
                    answer_mode=written.answer_mode,
                    report_available=written.report_available,
                )
            await self._buffer.flush(run_id)
            if run_id not in self._lost_leases:
                self._buffer.close(run_id)
            self._lost_leases.discard(run_id)
            self._shutdown_interrupts.discard(run_id)
            ledger.detach_run_sink(run_id)
            reset_usage_runtime(usage_token)
            if self._langfuse_client is not None:
                try:
                    self._langfuse_client.flush()
                except Exception:  # noqa: BLE001 - 观测出口不许反噬收尾
                    logger.debug("langfuse_flush_failed run_id=%s", run_id, exc_info=True)

    async def _execute_traced(
        self,
        work: RunWork,
        *,
        sink: CompositeSink,
        resume_input: Any,
    ) -> TerminalState | None:
        """在 run 的 TraceContext 绑定期间执行图。

        返回：
            终态写入后取回的那一行；挂起等澄清或写库未命中时为 None。
        """
        run_id = work.run_id
        resume = work.resume
        # 系统重启续跑已经由 consumer 播过 resuming，保持该状态直到后续阶段事件；
        # 人工澄清恢复从 queued 进入，前端还停在 queued，必须补一帧 running。
        if not resume or resume_input is not None:
            self._buffer.write_status(run_id, "running")
        # 获取该 RUN 的偏好快照
        async with self._session_factory() as session:
            preferences_snapshot = await session.scalar(
                select(Run.preferences_snapshot).where(Run.id == run_id)
            )
        run_settings = replace(
            self._settings, agent=apply_preferences(self._settings.agent, preferences_snapshot)
        )
        graph = self._graph_factory(
            settings=run_settings,
            event_sink=sink,
            http_client=self._http_client,
            checkpointer=self._checkpointer,
            material_store=self._material_store,
            provider_health=self._provider_health,
        )
        # resume 时传 None 或 Command，由 checkpointer + thread_id 从断点继续。
        if resume and resume_input is None and not await self._has_checkpoint(run_id):
            # 没有断点，需要重新开始。
            logger.warning("resume_without_checkpoint_restart run_id=%s", run_id)
            resume = False
        inputs = (
            resume_input
            if resume
            else {"query": work.query, "run_id": run_id, "session_id": run_id}
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
        # propagate_attributes 必须在图执行期间生效，langchain callbacks 生成的
        # trace 才会带上 session/user 归属。
        scope = (
            self._langfuse_scope(run_id=run_id, user_id=work.user_id)
            if self._langfuse_scope is not None
            else nullcontext()
        )
        with scope:
            result, interruption = await self._run_graph(work, graph, inputs, callbacks=callbacks)
        if interruption is not None:
            spans.record_output({"phase": "awaiting_input"})
            if not await self._persist_awaiting_input(work):
                self.mark_lease_lost(run_id)
                return None
            self._buffer.write(
                {
                    "run_id": run_id,
                    "event_type": EventName.CLARIFICATION_REQUESTED,
                    "payload": interruption,
                }
            )
            self._buffer.write_status(run_id, "awaiting_input")
            return None
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
        written = await self._persist_terminal(work, result)
        if written is None:
            self.mark_lease_lost(run_id)
        return written

    async def _run_graph(
        self,
        work: RunWork,
        graph: Any,
        inputs: Any,
        *,
        callbacks: list[Any] | None = None,
    ) -> tuple[dict, dict | None]:
        """从图流式输出根状态、中断与安全的 token 预览。"""
        run_id = work.run_id
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
                        if await self._persist_headline(work, headline):
                            # 写库成功即推帧，进行中的头部当场从长问题换成短题。根 values 块
                            # 在 clarify 之后即带 run_headline，所以本帧在澄清中断处发出。
                            self._buffer.write(
                                {
                                    "run_id": run_id,
                                    "event_type": EventName.RUN_HEADLINE_UPDATED,
                                    "payload": {"headline": headline[:RUN_HEADLINE_MAX_CHARS]},
                                }
                            )
                            # 立即 flush，不等周期：持久化与 NOTIFY 唤醒一次走完。
                            await self._buffer.flush(run_id)
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
        # langchain 1.x 的流式片段 .type 为 "AIMessageChunk" 而非 "ai"
        if getattr(message, "type", None) not in {"ai", "AIMessageChunk"}:
            return
        channel = _preview_channel(namespace)
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
                # 发布到预览通道，失败不反噬执行。
                if self._ephemeral_bus is not None:
                    await self._ephemeral_bus.publish(run_id, event)
            except Exception:  # noqa: BLE001 - 预览通道不反噬运行
                logger.debug("delta_publish_failed run_id=%s", run_id, exc_info=True)

    async def _persist_headline(self, work: RunWork, headline: str) -> bool:
        """澄清完成即把历史标题写进行：列表行从原 query 换成浓缩标题。"""
        async with self._session_factory() as session:
            result = await session.execute(
                update(Run)
                .where(
                    Run.id == work.run_id,
                    Run.headline.is_(None),
                    Run.status == "running",
                    Run.lease_owner == work.lease_owner,
                    Run.fence_token == work.fence_token,
                )
                .values(headline=headline[:RUN_HEADLINE_MAX_CHARS])
            )
            await session.commit()
            return result.rowcount == 1

    async def _persist_awaiting_input(self, work: RunWork) -> bool:
        async with self._session_factory() as session:
            result = await session.execute(
                update(Run)
                .where(*_owned(work))
                .values(**side_effects("awaiting_input", now=_utcnow()))
            )
            await session.commit()
            return result.rowcount == 1

    async def _persist_terminal(self, work: RunWork, result: dict) -> TerminalState | None:
        lifecycle = result.get("run") or {}
        phase = _field(lifecycle, "phase", "")
        status = "completed" if phase == "completed" else "failed"
        error = _field(lifecycle, "error", None)
        citations = [
            item.model_dump(exclude_none=True) if hasattr(item, "model_dump") else dict(item)
            for item in (result.get("citations") or [])
        ]
        return await self.persist_status(
            work,
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
        )

    async def persist_status(
        self, work: RunWork, *, status: str, **extra: Any
    ) -> TerminalState | None:
        """把生命周期事实写成行状态。

        参数：
            work: 写权凭据；谓词取它的 owner 与 fence_token。
            status: 目标状态。
            **extra: 附加列；None 值跳过。

        返回：
            写入后取回的状态与 done 帧字段，随写入一起返回；CAS 未命中或行已被
            接管时为 None。
        """
        values = side_effects(status, now=_utcnow())
        values.update({key: value for key, value in extra.items() if value is not None})
        async with self._session_factory() as session:
            result = await session.execute(
                update(Run)
                .where(*_owned(work))
                .values(**values)
                .returning(Run.status, Run.answer_mode, Run.report_markdown)
            )
            row = result.one_or_none()
            await session.commit()
        if row is None:
            return None
        return TerminalState(
            status=row.status,
            answer_mode=row.answer_mode or "",
            report_available=bool(row.report_markdown),
        )

    async def _periodic_flush(self, run_id: str) -> None:
        """周期性把结构性帧落库，写入失败则指数退避。

        成功即复位，间歇性失败不该拖慢稳态延迟。
        """
        delay = _FLUSH_INTERVAL_SECONDS
        try:
            while True:
                await asyncio.sleep(delay)
                if await self._buffer.flush(run_id):
                    delay = _FLUSH_INTERVAL_SECONDS
                else:
                    delay = min(delay * 2, _FLUSH_BACKOFF_MAX_SECONDS)
        except asyncio.CancelledError:
            return

    async def _fail_llm_unavailable(self, work: RunWork, user_code: str) -> TerminalState | None:
        """LLM 网关账户级不可用：类型化 terminal_reason + 安全文案，快速失败。"""
        logger.warning("research_llm_unavailable run_id=%s code=%s", work.run_id, user_code)
        return await self.persist_status(
            work,
            status="failed",
            terminal_reason=f"llm_unavailable:{user_code}",
            error_message=_LLM_UNAVAILABLE_MESSAGE.get(
                user_code, "模型服务暂时不可用，请稍后重试。"
            ),
        )


def _owned(work: RunWork) -> tuple[Any, ...]:
    """检查所有权。行仍是 running 且 owner 与 fence 仍一致"""
    return (
        Run.id == work.run_id,
        Run.status == "running",
        Run.lease_owner == work.lease_owner,
        Run.fence_token == work.fence_token,
    )


def _field(container: Any, key: str, default: Any = None) -> Any:
    """从 checkpoint 恢复的 dict 或 Pydantic/领域模型上读取字段。"""
    if isinstance(container, dict):
        return container.get(key, default)
    return getattr(container, key, default)


def _preview_channel(namespace: Any) -> str | None:
    """取 namespace 命中的预览通道；不是单层子图或不在白名单时返回 None。"""
    if not isinstance(namespace, tuple) or len(namespace) != 1:
        return None
    head = str(namespace[0]).split(":", 1)[0]
    return head if head in PREVIEW_CHANNELS else None
