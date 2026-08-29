"""RunManager：一次研究运行从受理到终态的全部生命周期。

三条贯穿本模块的不变式（都有对应回归测试）：

1. **run_id 由服务生成并强制注入**：行主键、ainvoke 输入、事件路由三者同一个
   id。引擎 instrumentation 的 setdefault 兜底只在 CLI 有意义；服务里缺了它，
   每个节点会各自生成新 id，事件与库行全部失联。
2. **终态恰好写一次**：``_persisted`` 同步预占（check-and-set 之间无 await，
   单循环内原子），自然完成与用户取消谁先到谁赢，行不振荡。
3. **run_done 是流的唯一收尾符**：经同一 fanout.write 拿 seq，天然排在全部
   引擎事件之后；发布后才做最终 flush，所以断线重连从 DB 回放也能看到它。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select

from deepsearch_agent.config import Settings
from deepsearch_agent.observability import JsonlSink
from deepsearch_agent.observability.tracing.context import new_id
from deepsearch_agent.orchestration.graph import build_graph
from deepsearch_agent.service.events import CompositeSink, FanoutSink
from deepsearch_agent.service.models import Run, RunEvent
from deepsearch_agent.service.settings import ServiceConfig

logger = logging.getLogger("deepsearch_agent.service.runs")

TERMINAL_STATUSES = ("completed", "failed", "cancelled")
_FLUSH_INTERVAL_SECONDS = 2.0


class QuotaExceededError(Exception):
    """用户活跃运行数已达并发额度。"""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RunManager:
    def __init__(
        self,
        *,
        settings: Settings,
        session_factory: Callable[[], Any],
        config: ServiceConfig,
        fanout: FanoutSink,
        http_client: Any,
        graph_factory: Callable[..., Any] = build_graph,
    ):
        self._settings = settings
        self._session_factory = session_factory
        self._config = config
        self._fanout = fanout
        self._http_client = http_client
        self._graph_factory = graph_factory
        self._tasks: dict[str, asyncio.Task] = {}
        self._quota_lock = asyncio.Lock()
        self._persisted: set[str] = set()
        self._done_published: set[str] = set()
        self._seen_delta_sources: set[tuple[str, str]] = set()

    # ---- 受理 ----

    async def start(self, user_id: int, query: str) -> str:
        query = query.strip()
        async with self._quota_lock:  # 计数+插入同锁内：并发 POST 不会双双过闸
            async with self._session_factory() as session:
                active = await session.scalar(
                    select(func.count())
                    .select_from(Run)
                    .where(Run.user_id == user_id, Run.status.in_(("queued", "running")))
                )
                if (active or 0) >= self._config.max_concurrent_runs_per_user:
                    raise QuotaExceededError(
                        f"同时进行的运行已达上限（{self._config.max_concurrent_runs_per_user}）。"
                    )
                run_id = new_id("run")
                session.add(
                    Run(id=run_id, user_id=user_id, query=query, status="queued")
                )
                await session.commit()
        self._fanout.open(run_id)
        await self._publish_status(run_id, "queued")
        task = asyncio.create_task(
            self._execute(run_id, user_id, query), name=f"research-{run_id}"
        )
        self._tasks[run_id] = task
        task.add_done_callback(lambda _t: self._tasks.pop(run_id, None))
        return run_id

    async def cancel(self, user_id: int, run_id: str) -> Run:
        async with self._session_factory() as session:
            run = await session.get(Run, run_id)
            if run is None or run.user_id != user_id:
                raise LookupError(run_id)  # API 层统一转 404
            if run.status in TERMINAL_STATUSES:
                return run  # 幂等：已终结的运行原样返回
        task = self._tasks.get(run_id)
        if task is not None and not task.done():
            task.cancel()  # 终态写入发生在 _execute 的 cancelled 分支
        else:
            # 本进程没有活任务（如 reconcile 前残留）：直接落终态。
            await self._persist_status(
                run_id, status="cancelled", terminal_reason="user_cancelled"
            )
            await self._publish_done(run_id)
        async with self._session_factory() as session:
            return await session.get(Run, run_id)

    # ---- 启动/停机 ----

    async def reconcile_startup(self) -> int:
        """上个进程死掉时残留的 queued/running 一律收敛为 failed。

        同时为每个孤儿 run 补一条持久化的 run_done 事件：它们的进程已蒸发，
        事件表天然缺少收尾帧——不补，前端打开历史详情页的 SSE 回放永远等不到
        done，会陷入无限重连。
        """
        async with self._session_factory() as session:
            stale = (
                await session.scalars(select(Run).where(Run.status.in_(("queued", "running"))))
            ).all()
            for run in stale:
                run.status = "failed"
                run.terminal_reason = "server_restart"
                run.error_message = "进程重启导致运行中断，请重新发起。"
                run.finished_at = _utcnow()
                max_seq = await session.scalar(
                    select(func.max(RunEvent.seq)).where(RunEvent.run_id == run.id)
                )
                done_record = {
                    "run_id": run.id,
                    "event_type": "run_done",
                    "seq": int(max_seq or 0) + 1,
                    "payload": {
                        "status": "failed",
                        "answer_mode": run.answer_mode or "",
                        "report_available": False,
                    },
                }
                session.add(
                    RunEvent(
                        run_id=run.id,
                        seq=done_record["seq"],
                        event_type="run_done",
                        record=done_record,
                    )
                )
            await session.commit()
            return len(stale)

    async def shutdown(self) -> None:
        for task in list(self._tasks.values()):
            task.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)

    # ---- 执行 ----

    async def _execute(self, run_id: str, user_id: int, query: str) -> None:
        sinks: list[Any] = [self._fanout]
        if self._config.jsonl_events:
            sinks.append(
                JsonlSink(self._config.service_log_dir / "events" / f"{run_id}.jsonl")
            )
        sink = CompositeSink(*sinks)
        flusher = asyncio.create_task(self._periodic_flush(run_id))
        try:
            await self._mark_running(run_id)
            graph = self._graph_factory(
                settings=self._settings_for(user_id),
                event_sink=sink,
                http_client=self._http_client,
            )
            # run_id 必须显式进入输入（见模块不变式 1）。
            result = await self._astream_run(
                run_id, graph, {"query": query, "run_id": run_id, "session_id": run_id}
            )
            await self._persist_terminal(run_id, result)
        except asyncio.CancelledError:
            # 引擎保证 CancelledError 干净重抛（执行边界不做业务失败化），
            # 这里持久化后吞掉：任务已把该做的事做完。
            await self._persist_status(
                run_id, status="cancelled", terminal_reason="user_cancelled"
            )
        except Exception as exc:  # noqa: BLE001 - 后台任务必须自收口
            logger.exception("research_run_failed run_id=%s", run_id)
            await self._persist_status(
                run_id,
                status="failed",
                terminal_reason="run_exception",
                error_message=str(exc)[:500],
            )
        finally:
            flusher.cancel()
            await asyncio.gather(flusher, return_exceptions=True)
            await self._publish_done(run_id)  # 先发布 done …
            await self._flush_events(run_id)  # … 再最终排水，done 因此也进 RunEvent
            self._fanout.close(run_id)

    def _settings_for(self, user_id: int) -> Settings:
        """BYO-keys 预留缝：将来按用户返回 replace(...) 的 Settings，仅此一处。"""
        return self._settings

    # ---- token 级预览（ephemeral 旁路；事实仍是落库的聚合帧） ----

    # 预览的交付节奏按"人眼流式"定，不按省事件定：实测网关约 25ms/字，
    # 8 字≈200ms 一帧；回合文本常不足 60 字，大阈值会把整个回合憋到
    # 流末尾一次性吐出——观感上等于没有流式（第一版就是这么错的）。
    _DELTA_FLUSH_CHARS = 8
    _DELTA_FLUSH_SECONDS = 0.12
    # 只有这两个 agent 的模型文本对用户可见（白名单，不是黑名单）；
    # 匹配不到归属的 chunk 一律静默并记一次观测日志，便于按真实
    # checkpoint_ns 形态扩表。
    _DELTA_CHANNEL_MARKERS = (("supervisor", "supervisor"), ("writer", "writer"))

    async def _astream_run(self, run_id: str, graph: Any, inputs: dict) -> dict:
        """values 模式取最终状态（与 ainvoke 等价），messages 模式引出 token 预览。

        节点内部拿到的仍是装配完整的 AIMessage（回调旁路不改返回值路径），
        因此日志/持久化/校验逻辑零改动。
        """
        final_state: dict = {}
        buffers: dict[str, list[str]] = {}
        last_flush: dict[str, float] = {}
        try:
            async for mode, chunk in graph.astream(
                inputs, stream_mode=["values", "messages"]
            ):
                if mode == "values":
                    if isinstance(chunk, dict):
                        final_state = chunk
                    # 超步边界 = 该步模型文本已终结：全部预览落屏，
                    # 聚合帧替换前不留未交付的碎尾巴。
                    for channel, parts in buffers.items():
                        self._flush_channel(run_id, channel, parts)
                    buffers.clear()
                    continue
                if mode != "messages":
                    continue
                delta = self._extract_text_delta(chunk)
                if delta is None:
                    continue
                channel, text = delta
                parts = buffers.setdefault(channel, [])
                parts.append(text)
                now = time.monotonic()
                if (
                    sum(len(part) for part in parts) >= self._DELTA_FLUSH_CHARS
                    or now - last_flush.get(channel, 0.0) >= self._DELTA_FLUSH_SECONDS
                ):
                    self._flush_channel(run_id, channel, parts)
                    buffers[channel] = []
                    last_flush[channel] = now
        finally:
            for channel, parts in buffers.items():
                self._flush_channel(run_id, channel, parts)
        return final_state

    def _extract_text_delta(self, chunk: Any) -> tuple[str, str] | None:
        try:
            message, metadata = chunk
        except (TypeError, ValueError):
            return None
        if not isinstance(metadata, Mapping):
            return None
        node = metadata.get("langgraph_node")
        ns = str(metadata.get("checkpoint_ns", ""))
        if node != "model":
            return None
        channel = next(
            (name for name, marker in self._DELTA_CHANNEL_MARKERS if marker in ns), None
        )
        if channel is None:
            self._log_unmatched_delta(node, ns)
            return None
        blocks = getattr(message, "content_blocks", None) or []
        text = "".join(
            str(block.get("text", ""))
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
        # tool_call_chunk / 推理块 type 不是 text，天然被排除在预览之外。
        return (channel, text) if text else None

    def _flush_channel(self, run_id: str, channel: str, parts: list[str]) -> None:
        text = "".join(parts)
        if not text:
            return
        self._fanout.publish_ephemeral(
            run_id,
            {
                "run_id": run_id,
                "event_type": "text_delta",
                "payload": {"channel": channel, "text": text},
            },
        )

    def _log_unmatched_delta(self, node: Any, ns: str) -> None:
        key = (str(node), ns.split(":")[0])
        if key not in self._seen_delta_sources:
            self._seen_delta_sources.add(key)
            # INFO 且每种来源只记一次：首局真跑需要看到真实 metadata 形态，
            # 以便校准 _DELTA_CHANNEL_MARKERS 白名单；之后可降回 debug。
            logger.info("text_delta_unmatched_source node=%s ns_prefix=%s", node, key[1])

    # ---- 持久化 ----

    async def _mark_running(self, run_id: str) -> None:
        async with self._session_factory() as session:
            run = await session.get(Run, run_id)
            if run is not None and run.status == "queued":
                run.status = "running"
                run.started_at = _utcnow()
                await session.commit()
        await self._publish_status(run_id, "running")

    async def _persist_terminal(self, run_id: str, result: dict) -> None:
        lifecycle = result.get("run") or {}
        phase = _field(lifecycle, "phase", "")
        status = "completed" if phase == "completed" else "failed"
        error = _field(lifecycle, "error", None)
        citations = [
            item.model_dump(exclude_none=True) if hasattr(item, "model_dump") else dict(item)
            for item in (result.get("citations") or [])
        ]
        await self._persist_status(
            run_id,
            status=status,
            terminal_reason=_field(lifecycle, "terminal_reason", "") or None,
            answer_mode=result.get("answer_mode"),
            report_markdown=result.get("report") or None,
            citations_json=citations,
            evidence_count=int(result.get("evidence_count") or len(result.get("evidences") or [])),
            source_count=int(result.get("source_count") or len(result.get("source_refs") or [])),
            error_message=(_field(error, "message", "") or None) if error else None,
        )

    async def _persist_status(self, run_id: str, *, status: str, **extra: Any) -> None:
        if run_id in self._persisted:  # 不变式 2：同步预占，无 await 间隙
            return
        self._persisted.add(run_id)
        async with self._session_factory() as session:
            run = await session.get(Run, run_id)
            if run is None or run.status in TERMINAL_STATUSES:
                return
            run.status = status
            run.finished_at = _utcnow()
            for key, value in extra.items():
                if value is not None:
                    setattr(run, key, value)
            await session.commit()

    # ---- 事件排水与合成帧 ----

    async def _periodic_flush(self, run_id: str) -> None:
        try:
            while True:
                await asyncio.sleep(_FLUSH_INTERVAL_SECONDS)
                await self._flush_events(run_id)
        except asyncio.CancelledError:
            return

    async def _flush_events(self, run_id: str) -> None:
        pending = self._fanout.take_pending(run_id)
        if not pending:
            return
        try:
            async with self._session_factory() as session:
                session.add_all(
                    [
                        RunEvent(
                            run_id=run_id,
                            seq=int(item["seq"]),
                            event_type=str(item.get("event_type", "")),
                            record=item,
                        )
                        for item in pending
                    ]
                )
                await session.commit()
        except Exception:  # noqa: BLE001 - 排水失败只影响回放完整性，绝不反噬运行
            logger.warning("run_event_flush_failed run_id=%s", run_id, exc_info=True)

    async def _publish_status(self, run_id: str, status: str) -> None:
        self._fanout.write(
            {"run_id": run_id, "event_type": "run_status", "payload": {"status": status}}
        )

    async def _publish_done(self, run_id: str) -> None:
        if run_id in self._done_published:
            return
        self._done_published.add(run_id)
        async with self._session_factory() as session:
            run = await session.get(Run, run_id)
        status = run.status if run is not None else "failed"
        self._fanout.write(
            {
                "run_id": run_id,
                "event_type": "run_done",
                "payload": {
                    "status": status,
                    "answer_mode": (run.answer_mode if run else None) or "",
                    "report_available": bool(run and run.report_markdown),
                },
            }
        )


def _field(container: Any, key: str, default: Any = None) -> Any:
    """pydantic 模型与普通 dict 双形态读取（checkpoint 恢复后的 state 是 dict）。"""
    if isinstance(container, dict):
        return container.get(key, default)
    return getattr(container, key, default)
