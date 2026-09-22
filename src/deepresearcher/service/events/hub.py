"""RunEventHub：事件面的唯一闸口——收(缓冲/席位)、放(落库/投递/门铃)、服务合成帧。

铁律(全类的存在理由,review 时逐条盯):
- ``write()`` 是"收":同步、任意线程可调;持锁、**永不 await**;
- ``flush()`` 是"放":仅事件循环线程调用;锁内只摘数据,**锁外 await**;
  append 成功后投递抛错只告警、绝不回插(持久覆水难收,回插=制造重复批次);
- 接缝就是方法边界:除 RunEventHub.flush 外任何代码不得直调 store.append——
  绕过本闸口 = 消费端失去本地投递与门铃,只靠轮询恢复。

席位(open)是本进程概念:executor/manager 声明"这个 run 的帧从这里进出",
超上限按插入序 FIFO 回收(回收≠终结:静默拆线降级 DB tail,不发 CLOSE_STREAM)。
逐字预览直推在 LocalPreviewBus;本类只在 close/drop 时指挥它的生命周期。
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from deepresearcher.observability.logging_config import get_logger
from deepresearcher.service.events.preview import LocalPreviewBus
from deepresearcher.service.events.store import RunEventStore
from deepresearcher.service.persistence.models import Run
from deepresearcher.service.signals import PostgresSignalBus

logger = get_logger("deepresearcher.service.events")

#: open 席位上限(API/Worker 进程各自持有):终结路径有 close,
#: "从未被订阅也从未跑完"的 run 由超额回收兜底,常驻集合不许无界增长。
_OPEN_MAX = 2048
_DONE_DEDUP_MAX = 2048


class RunEventHub:
    def __init__(
        self,
        *,
        session_factory: Callable[[], Any],
        store: RunEventStore,
        preview: LocalPreviewBus,
        signal_bus: PostgresSignalBus,
        open_max: int = _OPEN_MAX,
    ) -> None:
        self._session_factory = session_factory
        self.store = store  # 只读句柄:manager/SSE 经它 tail;写入唯一入口仍是本类 flush
        self._preview = preview
        self._signal_bus = signal_bus
        self._open_max = max(1, open_max)
        self._lock = threading.Lock()
        # 键序即 open 的 FIFO 序(dict 保序);超限回收最旧席位。
        self._open: dict[str, None] = {}
        self._pending: dict[str, list[dict]] = {}
        # done 幂等只需覆盖"同进程相邻两次发布"，插入序即 FIFO 淘汰序。
        self._done_published: dict[str, None] = {}
        self._announced_unrouted = False

    # ---- 收：引擎 sink 契约（任意线程；持锁；绝不 await）----

    def write(self, record: BaseModel | Mapping[str, Any] | Any) -> None:
        """把无 seq 记录放入该 run 的 pending 缓冲；未 open 的 run 丢弃并告警一次。"""
        if isinstance(record, BaseModel):
            data = record.model_dump(exclude_none=True)
        elif isinstance(record, Mapping):
            data = dict(record)
        else:
            return
        run_id = data.get("run_id")
        with self._lock:
            if not isinstance(run_id, str) or run_id not in self._open:
                self._drop_unrouted_locked(run_id)
                return
            self._pending.setdefault(run_id, []).append(data)

    # ---- 席位生命周期 ----

    def open(self, run_id: str) -> None:
        with self._lock:
            if run_id not in self._open:
                self._open[run_id] = None
            self._pending.setdefault(run_id, [])
            while len(self._open) > self._open_max:
                self._evict_oldest_locked()

    def _evict_oldest_locked(self) -> None:
        # 回收 ≠ 终结:preview.drop 静默拆线(不发 CLOSE_STREAM,run 可能还活着);
        # 在途 flush 之后对已回收席位的投递天然是 no-op。
        oldest = next(iter(self._open))
        del self._open[oldest]
        self._pending.pop(oldest, None)
        self._preview.drop(oldest)

    def close(self, run_id: str) -> None:
        """终止该 run 的分发：先向订阅者发 CLOSE_STREAM 哨兵，再摘席位与缓冲。

        未取走的 pending 一并丢弃——终结顺序(take→最后一次 flush→close)
        是 executor 的职责,正常路径走不到丢数据;走到即记 warning。
        """
        with self._lock:
            had_open = self._open.pop(run_id, None) is not None
            pending = self._pending.pop(run_id, None)
        if had_open and pending:
            logger.warning(
                "run_event_pending_dropped_on_close run_id=%s count=%d", run_id, len(pending)
            )
        self._preview.close(run_id)

    def is_open(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._open

    # ---- 服务层合成帧（低频编排；write 的语法糖，不 flush）----

    async def publish_status(self, run_id: str, status: str) -> None:
        self.write({"run_id": run_id, "event_type": "run_status", "payload": {"status": status}})

    async def publish_done(self, run_id: str) -> None:
        with self._lock:
            if run_id in self._done_published:
                return
            self._done_published[run_id] = None
            while len(self._done_published) > _DONE_DEDUP_MAX:
                self._done_published.pop(next(iter(self._done_published)))
        async with self._session_factory() as session:
            run = await session.get(Run, run_id)
        status = run.status if run is not None else "failed"
        self.write(
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

    # ---- 放：排水编排（仅事件循环线程；锁内摘数据，锁外 await）----

    async def flush(self, run_id: str) -> None:
        with self._lock:
            records = self._pending.pop(run_id, [])
        if not records:
            return
        try:
            assigned = await self.store.append(run_id, records)
        except Exception:  # noqa: BLE001 - 排水失败不毒化 run,帧回插队首等下轮
            logger.warning("run_event_flush_failed run_id=%s", run_id, exc_info=True)
            self._requeue(run_id, records)
            return
        # commit 成功即覆水难收:以下两步的异常只记日志,绝不回插(那会造重复批次)。
        try:
            self._preview.deliver(run_id, assigned)
        except Exception:  # noqa: BLE001 - 本地直推尽力而为,DB tail 是地板
            logger.warning("run_event_deliver_failed run_id=%s", run_id, exc_info=True)
        await self._signal_bus.notify_event(run_id)

    def _requeue(self, run_id: str, records: Sequence[dict]) -> None:
        """持久化失败时把批次放回队首——只在席位仍在时回插;
        close 之后的迟到批次按既有契约丢弃,客户端由 SSE 的行终态兜底收敛。"""
        with self._lock:
            if run_id not in self._open:
                return
            self._pending.setdefault(run_id, [])[:0] = records

    # ---- 内部 ----

    def _drop_unrouted_locked(self, run_id: Any) -> None:
        if not self._announced_unrouted:
            self._announced_unrouted = True
            logger.warning(
                "run_event_hub_dropped_unrouted：收到无 run_id 或 run 未 open 的事件"
                "（首个，run_id=%r），此类事件不再生成告警。",
                run_id,
            )
