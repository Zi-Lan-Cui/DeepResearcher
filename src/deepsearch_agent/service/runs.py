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
from collections.abc import Callable
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
            result = await self._run_graph(
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

    # ---- token 级预览：官方嵌套流式（subgraphs=True），引擎零修改 ----

    async def _run_graph(self, run_id: str, graph: Any, inputs: dict) -> dict:
        """astream(values+messages, subgraphs=True)：values 根命名空间 = ainvoke 返回值；

        messages 携带**嵌套 agent**（create_agent 子图，含 context 注入调用）的逐字
        token——实测四种嵌套/组合形态，唯有 subgraphs=True 能同时给出
        根最终状态与子图 token 流（见 docs/service-p0.md）。异常与取消在
        async-for 处抛出，语义与原 ainvoke 一致，_execute 的 except 分支不动。
        """
        final: dict = {}
        async for namespace, mode, chunk in graph.astream(
            inputs, stream_mode=["values", "messages"], subgraphs=True
        ):
            if mode == "values":
                if namespace == () and isinstance(chunk, dict):
                    final = chunk
                continue
            if mode == "messages":
                # 同步直发：publish_ephemeral 内部只有锁+put_nowait，不阻塞；
                # 绝不可放线程池——乱序完成的 to_thread 会打乱 token 帧序。
                self._publish_message_preview(run_id, namespace, chunk)
        return final

    def _publish_message_preview(self, run_id: str, namespace: Any, chunk: Any) -> None:
        try:
            message, _metadata = chunk
        except (TypeError, ValueError):
            return
        # messages 模式同样投递 ToolMessage（如 ReadWorkingSet 的 JSON 回执）：
        # 工具结果是整块消息不是 token 流，混入预览会"啪"地弹出内部文本。
        # 只放行语言模型的输出（AIMessage/AIMessageChunk 的 type == "ai"）。
        if getattr(message, "type", None) != "ai":
            return
        channel = self._preview_channel(namespace)
        if channel is None:
            return
        for block in getattr(message, "content_blocks", None) or []:
            # tool_call_chunk / 推理块 type 不是 text：报告 JSON 半成品永不进预览。
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
        """ns 路径恰好落在 supervisor 宿主节点 → supervisor 思考通道。

        深度大于 1 的路径（如 ('supervisor:…','tools:…')）是 researcher 在
        工具内嵌套执行的 token，混入会串卡；writer/reflection 不在白名单——
        用户只应看到它们的聚合结果。
        """
        if not isinstance(namespace, tuple) or len(namespace) != 1:
            return None
        head = str(namespace[0]).split(":", 1)[0]
        return head if head == "supervisor" else None

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
