import asyncio

import httpx
import pytest

from deepresearcher.service.api import create_app
from deepresearcher.service.execution.runtime import worker_lifespan
from deepresearcher.service.persistence.database import make_engine, make_session_factory
from deepresearcher.service.persistence.models import Run, User
from fakes_service import FakeGraph, service_config, service_settings

pytestmark = pytest.mark.asyncio


async def test_langfuse_unreachable_still_completes_run(tmp_path):
    """观测出口合同:langfuse 后端不可达,run 照常完成,不反噬执行。

    base_url 指向 discard 端口(连接拒绝,立即失败),导出侧没有服务。
    """
    settings = service_settings(tmp_path)
    config = service_config(
        tmp_path,
        worker_poll_seconds=0.01,
        langfuse_public_key="pk-lf-test",
        langfuse_secret_key="sk-lf-test",
        langfuse_base_url="http://127.0.0.1:9",
    )
    engine = make_engine(config.database_url)
    session_factory = make_session_factory(engine)
    try:
        async with worker_lifespan(
            settings, config, graph_factory=lambda **_kwargs: FakeGraph()
        ) as runtime:
            assert runtime.executor.langfuse_enabled()  # 已装 extra 且三键齐备
            async with session_factory() as session:
                session.add(User(id=77, email="lf@test.dev", password_hash="h"))
                session.add(Run(id="run-lf", user_id=77, query="观测出口可用性", status="queued"))
                await session.commit()
            await runtime.wake()
            async with asyncio.timeout(15):
                while True:
                    async with session_factory() as session:
                        run = await session.get(Run, "run-lf")
                        if run is not None and run.status in {"completed", "failed"}:
                            break
                    await asyncio.sleep(0.02)
            assert run.status == "completed"
    finally:
        await engine.dispose()


async def test_otel_console_exporter_completes_run(tmp_path, capsys):
    """无 langfuse、console 导出:span 打到 stdout,run 照常完成。"""
    settings = service_settings(tmp_path)
    config = service_config(tmp_path, worker_poll_seconds=0.01, otel_exporter="console")
    engine = make_engine(config.database_url)
    session_factory = make_session_factory(engine)
    try:
        async with worker_lifespan(
            settings, config, graph_factory=lambda **_kwargs: FakeGraph()
        ) as runtime:
            assert not runtime.executor.langfuse_enabled()
            async with session_factory() as session:
                session.add(User(id=78, email="otel@test.dev", password_hash="h"))
                session.add(Run(id="run-otel", user_id=78, query="OTel 导出自检", status="queued"))
                await session.commit()
            await runtime.wake()
            async with asyncio.timeout(15):
                while True:
                    async with session_factory() as session:
                        run = await session.get(Run, "run-otel")
                        if run is not None and run.status in {"completed", "failed"}:
                            break
                    await asyncio.sleep(0.02)
            assert run.status == "completed"
        assert "research_run" in capsys.readouterr().out
    finally:
        await engine.dispose()


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
    # 领取在构图之前就把 ``running`` 提交;观测第二个条件,
    # 而不是假设两个动作原子发生。
    async with asyncio.timeout(3):
        while len(graphs) != expected:
            await asyncio.sleep(0.01)


async def test_two_workers_execute_once_and_api_restart_does_not_cancel(tmp_path):
    """在一个共享持久数据库上验证 M8 的进程边界。"""
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

            # 第一个 API 已退出,被 Worker 独立持有的图仍在运行。
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
