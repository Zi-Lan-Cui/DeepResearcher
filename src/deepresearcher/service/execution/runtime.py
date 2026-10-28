"""worker 执行层 runtime：进程装配与生命周期。"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from deepresearcher.config import Settings, get_settings
from deepresearcher.graph import build_graph
from deepresearcher.observability.logging_config import get_logger
from deepresearcher.service.bootstrap.layers import MessagingPorts, build_runtime_stack
from deepresearcher.service.events.buffer import RunEventBuffer
from deepresearcher.service.execution.consumer import RunConsumer
from deepresearcher.service.execution.executor import RunExecutor
from deepresearcher.service.execution.limits import CapacityGate, ProviderRateLimiter
from deepresearcher.service.execution.telemetry import (
    configure_tracer_provider,
    shutdown_tracer_provider,
)
from deepresearcher.service.execution.usage import UsageStore
from deepresearcher.service.messaging.publisher import DispatchPublisher
from deepresearcher.service.messaging.receiver import DispatchReceiver
from deepresearcher.service.preview.protocol import EphemeralEventBus
from deepresearcher.service.runs.retention import prune_event_jsonl, sweep_run_traces
from deepresearcher.service.runs.store import RunStore
from deepresearcher.service.settings import ServiceConfig, get_service_config
from deepresearcher.tools.web.materials import ResearchMaterialStore

logger = get_logger("deepresearcher.service.execution.runtime")

# 保留清扫循环首轮延迟：等连接池与启动归类集中处理，随机抖动另防副本齐动。
_FIRST_SWEEP_DELAY_SECONDS = 60


class WorkerRuntime:
    """只归 Worker 进程使用的全部资源与恢复策略。"""

    def __init__(
        self,
        *,
        settings: Settings,
        session_factory: Callable[[], Any],
        config: ServiceConfig,
        buffer: RunEventBuffer,
        http_client: Any,
        publisher: DispatchPublisher,
        receiver: DispatchReceiver,
        graph_factory: Callable[..., Any] = build_graph,
        checkpointer: Any = None,
        material_store: ResearchMaterialStore | None = None,
        ephemeral_bus: EphemeralEventBus | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._buffer = buffer
        self._checkpointer = checkpointer
        self._config = config
        self._publisher = publisher
        self._retention_task: asyncio.Task[None] | None = None
        self.usage_store = UsageStore(session_factory)
        self.llm_gate = CapacityGate(settings.llm.max_concurrent_requests)
        self.llm_rate_limiter = ProviderRateLimiter(
            requests_per_minute=settings.llm.provider_requests_per_minute,
            tokens_per_minute=settings.llm.provider_tokens_per_minute,
        )
        self.store = RunStore(session_factory)
        self.executor = RunExecutor(
            settings=settings,
            session_factory=session_factory,
            config=config,
            buffer=buffer,
            usage_store=self.usage_store,
            llm_gate=self.llm_gate,
            llm_rate_limiter=self.llm_rate_limiter,
            http_client=http_client,
            graph_factory=graph_factory,
            checkpointer=checkpointer,
            material_store=material_store,
            ephemeral_bus=ephemeral_bus,
        )
        self.consumer = RunConsumer(
            store=self.store,
            executor=self.executor,
            buffer=buffer,
            receiver=receiver,
            max_running=config.worker_max_running_runs,
            lease_check_seconds=config.worker_lease_check_seconds,
            invisible_seconds=config.rocketmq_invisible_seconds,
            visibility_renew_seconds=config.rocketmq_visibility_renew_seconds,
            max_attempts=config.rocketmq_dispatch_max_attempts,
        )

    @property
    def worker_id(self) -> str:
        return self.consumer.worker_id

    @property
    def tasks(self) -> dict[str, Any]:
        return self.consumer.tasks

    async def start(self) -> None:
        await self.consumer.start()
        self._start_retention_loop()

    async def handle_cancel_notification(self, run_id: str) -> None:
        """跨进程取消的快速路径：先回查队列行再行动。"""
        await self.consumer.cancel_if_confirmed(run_id)

    def _start_retention_loop(self) -> None:
        config = self._config
        if config.run_trace_retention_days <= 0 and config.jsonl_retention_days <= 0:
            return
        self._retention_task = asyncio.create_task(
            self._retention_loop(), name=f"retention-sweep-{self.consumer.worker_id}"
        )

    async def _retention_loop(self) -> None:
        """保留期清扫的常驻循环：首轮短延迟后错峰，此后按配置间隔。

        首轮不等满一个间隔，让重启也起到清淤作用；随机抖动防多副本齐动。
        """
        config = self._config
        logger.info(
            "retention_loop_started trace_days=%d jsonl_days=%d interval_seconds=%d",
            config.run_trace_retention_days,
            config.jsonl_retention_days,
            config.retention_sweep_interval_seconds,
        )
        delay_seconds = _FIRST_SWEEP_DELAY_SECONDS + random.uniform(0.0, 30.0)
        try:
            while True:
                await asyncio.sleep(delay_seconds)
                delay_seconds = config.retention_sweep_interval_seconds
                try:
                    summary = await sweep_run_traces(
                        self._session_factory,
                        self._checkpointer,
                        retention_days=config.run_trace_retention_days,
                    )
                    pruned_files = prune_event_jsonl(
                        config.service_log_dir / "events", config.jsonl_retention_days
                    )
                except Exception:  # noqa: BLE001 - 瞬断下轮重试，清扫不得波及在线与关停
                    logger.warning("retention_sweep_failed", exc_info=True)
                    continue
                if summary or pruned_files:
                    logger.info("retention_sweep summary=%s jsonl_files=%d", summary, pruned_files)
        except asyncio.CancelledError:
            return

    async def shutdown(self) -> None:
        if self._retention_task is not None:
            self._retention_task.cancel()
            await asyncio.gather(self._retention_task, return_exceptions=True)
            self._retention_task = None
        await self.consumer.shutdown()
        self.executor.shutdown_langfuse()

    async def recover_run(self, run_id: str) -> bool:
        """显式恢复入口：把重试耗尽的行重新交给消息队列。

        参数：
            run_id: 需要恢复的 run。

        返回：
            bool: 完成了退回与重投；行不是该原因结案时为 False。
        """
        if not await self.store.reopen_exhausted(run_id):
            return False
        logger.warning("dispatch_recovered run_id=%s", run_id)
        self._buffer.open(run_id)
        await self._publisher.redispatch(run_id)
        return True


@asynccontextmanager
async def worker_lifespan(
    settings: Settings | None = None,
    config: ServiceConfig | None = None,
    *,
    graph_factory: Callable[..., Any] = build_graph,
    messaging: MessagingPorts | None = None,
) -> AsyncIterator[WorkerRuntime]:
    """创建一个独立 Worker 进程拥有的全部资源。

    参数：
        messaging: 消息队列端口；为 None 时连真实 RocketMQ。测试注入进程内替身。

    返回：
        已进入启动流程的 worker 运行时，上下文退出时释放。
    """
    service_config = config or get_service_config()
    engine_settings = settings or get_settings()
    # 先于 RunExecutor(首次 Langfuse())装配全局 provider，让 langfuse 接管同一棵树。
    configure_tracer_provider(service_config)
    async with build_runtime_stack(
        service_config, engine_settings, role="worker", messaging=messaging
    ) as stack:
        if stack.receiver is None:
            raise RuntimeError("worker 角色必须装配拉取端。")
        # 保留期清扫（jsonl 与 run 过程数据）只由 WorkerRuntime 的背景循环承担。
        worker = WorkerRuntime(
            settings=engine_settings,
            session_factory=stack.session_factory,
            config=service_config,
            buffer=stack.buffer,
            http_client=stack.http_client,
            publisher=stack.publisher,
            receiver=stack.receiver,
            graph_factory=graph_factory,
            checkpointer=stack.checkpointer,
            material_store=stack.material_store,
            ephemeral_bus=stack.ephemeral_bus,
        )
        cancel_subscription = stack.signal_bus.subscribe(
            "run_cancel_requested", worker.handle_cancel_notification
        )
        try:
            await worker.start()
            logger.info("worker_started worker_id=%s", worker.worker_id)
            yield worker
        finally:
            logger.info("worker_stopping worker_id=%s", worker.worker_id)
            stack.signal_bus.unsubscribe("run_cancel_requested", cancel_subscription)
            await worker.shutdown()
            shutdown_tracer_provider()
