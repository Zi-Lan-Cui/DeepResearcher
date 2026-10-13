"""方向到达归并中间件：派发解屏障后的收口点。

派发批只登记 pending_directions 通道并拉起进程内后台任务；本中间件在每个模型回合前
把已到齐的方向吸收进工作集：登记但无任务体即进程重启的丢失判定，完成即吸收并销账，
差量随本钩子的 superstep 落回通道，方向报告以"研究管理观察（方向到达）"送进模型。
ResearchComplete 因在途被拒后置 arrival_wait_requested，此处先等一个到达再进模型，
避免空转轮询。supervisor 包与本模块互相引用，重依赖在函数体内延迟导入。
"""

import asyncio
from typing import Any, cast

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage

from deepresearcher.prompts import render_data_section
from deepresearcher.state import SubTask


class DirectionArrivalsMiddleware(AgentMiddleware):
    """每模型回合前吸收已到齐方向；无事可做返回 None，非派发路径零开销。"""

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        from deepresearcher.agents.supervisor import services

        context = getattr(runtime, "context", None)
        registry = getattr(context, "direction_tasks", None)
        loop_state = getattr(context, "loop_state", None)
        if context is None or registry is None or loop_state is None:
            return None
        if not loop_state.pending_directions and not registry:
            return None

        reports: list[dict[str, object]] = []
        delta: dict[str, Any] = {}

        def merge(part: dict[str, Any]) -> None:
            for key, value in part.items():
                if key == "active_evidence_ids":
                    existing = delta.setdefault(key, {"add": [], "remove": []})
                    existing["add"] = sorted(set(existing["add"]) | set(value["add"]))
                    existing["remove"] = sorted(set(existing["remove"]) | set(value["remove"]))
                elif key == "pending_directions":
                    existing = delta.setdefault(key, {"add": {}, "remove": []})
                    existing["add"].update(value["add"])
                    existing["remove"] = sorted(set(existing["remove"]) | set(value["remove"]))
                else:
                    delta[key] = value

        # 丢失判定与节点收口共用 services.absorb_lost_registrations:回合可能不再来,两处都要结算。
        for report, lost_delta in services.absorb_lost_registrations(
            context.deps, loop_state, set(registry)
        ):
            reports.append(report)
            merge(lost_delta)

        # 等待模式：上一回合 ResearchComplete 被在途拒绝，此刻又无已到结果，先等一个到达。
        live = [item for item in registry.values() if not item.done()]
        if (
            context.arrival_wait_requested
            and live
            and not any(item.done() for item in registry.values())
        ):
            await asyncio.wait(live, return_when=asyncio.FIRST_COMPLETED)

        for task_id in sorted(list(registry)):
            task = registry[task_id]
            if not task.done():
                continue
            execution = task.result()  # 任务体 total：除取消外不抛。
            entry = loop_state.pending_directions.get(task_id) or {}
            subtask = dict(entry.get("subtask") or {"id": task_id, "question": ""})
            registry.pop(task_id)
            report, task_delta = services.absorb_arrival(
                context.deps,
                loop_state,
                cast(SubTask, subtask),
                execution,
                merged_via="model_turn",
            )
            reports.append(report)
            merge(task_delta)

        if context.arrival_wait_requested and not loop_state.pending_directions:
            context.arrival_wait_requested = False
        if not reports:
            return None
        payload: dict[str, object] = {
            "arrived_directions": reports,
            "working_set_revision": loop_state.working_set_revision,
            "inflight": [
                {
                    "task_id": key,
                    "display_title": str((item or {}).get("display_title") or ""),
                }
                for key, item in sorted(loop_state.pending_directions.items())
            ],
        }
        message = HumanMessage(content=render_data_section("研究管理观察（方向到达）", payload))
        return {"messages": [message], **delta}
