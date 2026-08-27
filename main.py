import argparse
import asyncio
from uuid import uuid4

from deepsearch_agent.config import get_settings
from deepsearch_agent.observability import JsonlSink, configure_logging
from deepsearch_agent.observability.tracing import TraceRecorder
from deepsearch_agent.orchestration.graph import build_graph
from deepsearch_agent.tools import HttpClient


async def main() -> None:
    settings = get_settings()
    log_dir = settings.observability.log_dir
    configure_logging(settings.app.log_level, log_path=log_dir / settings.observability.log_file)
    parser = argparse.ArgumentParser(description="Run a first-generation deep research agent")
    parser.add_argument("query", nargs="?", help="research question")
    args = parser.parse_args()
    query = args.query or input("研究问题：").strip()
    if not query:
        print("研究问题不能为空。")
        return
    event_sink = JsonlSink(log_dir / settings.observability.event_file)
    trace_recorder = TraceRecorder(JsonlSink(log_dir / settings.observability.trace_file))
    http_client = HttpClient(settings.search)
    run_id = f"run-{uuid4().hex}"
    try:
        with trace_recorder.trace("research", run_id=run_id, session_id="cli"):
            result = await build_graph(
                settings=settings,
                event_sink=event_sink,
                trace_recorder=trace_recorder,
                http_client=http_client,
            ).ainvoke({"query": query, "run_id": run_id, "session_id": "cli", "raw_documents": []})
    finally:
        await http_client.aclose()
    print(result["report"])


if __name__ == "__main__":
    asyncio.run(main())
