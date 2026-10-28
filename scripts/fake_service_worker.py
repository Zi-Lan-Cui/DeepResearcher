"""仅供测试的 Worker 进程，仅被 `verify_service_processes.py` 调用。"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from deepresearcher.config import get_settings
from deepresearcher.observability import configure_logging
from deepresearcher.service.bootstrap.loop import asyncio_loop_factory, install_stop_handlers
from deepresearcher.service.execution.runtime import worker_lifespan


class HarnessState(TypedDict, total=False):
    query: str
    run_id: str
    session_id: str
    run: dict
    answer_mode: str
    report: str
    citations: list[dict]
    evidence_count: int
    source_count: int


def build_harness_graph(*, event_sink, checkpointer=None, **_kwargs):
    marker_dir = Path(os.environ["WORKER_MARKER_DIR"])

    async def checkpoint_ready(_state: HarnessState) -> dict:
        # 先提交一个空 superstep 作为恢复点，controlled_work 才被文件信号阻塞。
        return {}

    async def controlled_work(state: HarnessState) -> dict:
        run_id = state["run_id"]
        marker_dir.mkdir(parents=True, exist_ok=True)
        # 标记名里带 `WORKER_INSTANCE`：Windows 的 venv `python.exe` 是启动器，
        # 它再起真实解释器，于是 `Popen.pid` 与写标记的 pid 不是同一个进程。
        # 验收脚本要按「哪个 worker 进了图」去杀/停进程，只能靠名字回查。
        instance = os.environ.get("WORKER_INSTANCE", "worker")
        (marker_dir / f"entered-{run_id}-{instance}-{os.getpid()}").touch()
        event_sink.write(
            {
                "run_id": run_id,
                "event_type": "harness_work_started",
                "payload": {"pid": os.getpid()},
            }
        )
        release = marker_dir / f"release-{run_id}"
        while not release.exists():
            await asyncio.sleep(0.05)
        return {
            "run": {"phase": "completed", "terminal_reason": "harness_completed"},
            "answer_mode": "deep_research",
            "report": f"# Service 自动验收\n\nRun `{run_id}` completed.",
            "citations": [],
            "evidence_count": 0,
            "source_count": 0,
        }

    builder = StateGraph(HarnessState)
    builder.add_node("checkpoint_ready", checkpoint_ready)
    builder.add_node("controlled_work", controlled_work)
    builder.add_edge(START, "checkpoint_ready")
    builder.add_edge("checkpoint_ready", "controlled_work")
    builder.add_edge("controlled_work", END)
    return builder.compile(checkpointer=checkpointer)


async def run() -> None:
    stopped = asyncio.Event()
    install_stop_handlers(stopped)
    async with worker_lifespan(graph_factory=build_harness_graph):
        await stopped.wait()


def main() -> None:
    settings = get_settings()
    configure_logging(
        settings.app.log_level,
        log_path=settings.observability.log_dir / settings.observability.log_file,
    )
    asyncio.run(run(), loop_factory=asyncio_loop_factory())


if __name__ == "__main__":
    main()
