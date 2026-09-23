import asyncio

import httpx
import pytest

from deepresearcher.service.api import create_app
from deepresearcher.service.execution.runtime import worker_lifespan
from fakes_service import FakeGraph, service_config, service_settings

pytestmark = pytest.mark.asyncio


async def _wait_status(client, token, run_id, expected):
    async with asyncio.timeout(3):
        while True:
            response = await client.get(
                f"/api/runs/{run_id}", headers={"Authorization": f"Bearer {token}"}
            )
            if response.json()["status"] in expected:
                return response.json()
            await asyncio.sleep(0.01)


async def _wait_graph_count(graphs, expected):
    # Claim commits ``running`` immediately before graph construction; observe
    # the second condition rather than assuming both actions are atomic.
    async with asyncio.timeout(3):
        while len(graphs) != expected:
            await asyncio.sleep(0.01)


async def test_two_workers_execute_once_and_api_restart_does_not_cancel(tmp_path):
    """Exercise the M8 process boundaries against one shared durable database."""
    settings = service_settings(tmp_path)
    config = service_config(tmp_path, worker_poll_seconds=0.01)
    gate = asyncio.Event()
    built_graphs: list[FakeGraph] = []

    def worker_graph_factory(**_kwargs):
        graph = FakeGraph(gate=gate)
        graph._sink = _kwargs["event_sink"]
        built_graphs.append(graph)
        return graph

    async with worker_lifespan(settings, config, graph_factory=worker_graph_factory):
        async with worker_lifespan(settings, config, graph_factory=worker_graph_factory):
            first_app = create_app(settings, config)
            assert not hasattr(first_app.state, "execution")
            async with first_app.router.lifespan_context(first_app):
                transport = httpx.ASGITransport(app=first_app)
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://first-api"
                ) as first_client:
                    registered = await first_client.post(
                        "/api/register",
                        json={"email": "m8@test.dev", "password": "goodpassword"},
                    )
                    token = registered.json()["token"]
                    created = await first_client.post(
                        "/api/runs",
                        json={"query": "survive API restart"},
                        headers={"Authorization": f"Bearer {token}"},
                    )
                    run_id = created.json()["run_id"]
                    await _wait_status(first_client, token, run_id, {"running"})
                    await _wait_graph_count(built_graphs, 1)
                    assert len(built_graphs) == 1

            # First API is now gone while the independently owned graph remains live.
            gate.set()
            second_app = create_app(settings, config)
            async with second_app.router.lifespan_context(second_app):
                transport = httpx.ASGITransport(app=second_app)
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://second-api"
                ) as second_client:
                    completed = await _wait_status(second_client, token, run_id, {"completed"})
                    assert completed["report_markdown"].startswith("# 研究报告")
                    assert len(built_graphs) == 1
