"""Test-only Worker process with a checkpointed, externally released graph."""

from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from deepsearch_agent.config import get_settings
from deepsearch_agent.observability import configure_logging
from deepsearch_agent.service.execution.runtime import worker_lifespan


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
    marker_dir = Path(os.environ["M8_MARKER_DIR"])

    async def checkpoint_ready(_state: HarnessState) -> dict:
        # This completed superstep is the recovery point before the blocking work.
        return {}

    async def controlled_work(state: HarnessState) -> dict:
        run_id = state["run_id"]
        marker_dir.mkdir(parents=True, exist_ok=True)
        (marker_dir / f"entered-{run_id}-{os.getpid()}").touch()
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
            "report": f"# M8 自动验收\n\nRun `{run_id}` completed.",
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
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stopped.set)
    async with worker_lifespan(graph_factory=build_harness_graph):
        await stopped.wait()


def main() -> None:
    settings = get_settings()
    configure_logging(
        settings.app.log_level,
        log_path=settings.observability.log_dir / settings.observability.log_file,
    )
    asyncio.run(run())


if __name__ == "__main__":
    main()
