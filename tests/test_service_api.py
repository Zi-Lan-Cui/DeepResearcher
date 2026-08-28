import asyncio

import httpx
import pytest
import pytest_asyncio

from deepsearch_agent.service.api import create_app
from fakes_service import (
    FakeGraph,
    parse_sse,
    service_config,
    service_settings,
)

pytestmark = pytest.mark.asyncio

PASSWORD = "goodpassword"


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _tick_events(frames):
    return [data["text"] for event, data in frames if event == "tick"]


@pytest_asyncio.fixture
async def client(tmp_path):
    graphs: list[FakeGraph] = []

    def graph_factory(*, settings, event_sink, http_client):
        graph = graphs.pop(0) if graphs else FakeGraph()
        graph._sink = event_sink
        return graph

    app = create_app(
        service_settings(tmp_path), service_config(tmp_path), graph_factory=graph_factory
    )
    # httpx ASGITransport 不执行 lifespan：在当前循环内手动进出，
    # 使 engine/fanout/manager 与测试共享同一个事件循环。
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://svc") as c:
            c.graphs = graphs  # type: ignore[attr-defined]
            c.app = app  # type: ignore[attr-defined]
            yield c


async def register(client, email="a@test.dev") -> str:
    response = await client.post(
        "/api/register", json={"email": email, "password": PASSWORD}
    )
    assert response.status_code == 201
    return response.json()["token"]


async def wait_status(client, token, run_id, statuses, timeout=5.0):
    async with asyncio.timeout(timeout):
        while True:
            response = await client.get(f"/api/runs/{run_id}", headers=_auth(token))
            assert response.status_code == 200
            body = response.json()
            if body["status"] in statuses:
                return body
            await asyncio.sleep(0.02)


async def read_sse(client, token, run_id, timeout=10.0):
    text = ""
    async with asyncio.timeout(timeout):
        async with client.stream(
            "GET", f"/api/runs/{run_id}/events", headers=_auth(token)
        ) as response:
            assert response.status_code == 200
            async for chunk in response.aiter_text():
                text += chunk
                if "event: done" in text:
                    break
    return parse_sse(text)


# ---- 鉴权 ----


async def test_register_login_and_me_roundtrip(client):
    token = await register(client)
    me = await client.get("/api/me", headers=_auth(token))
    assert me.status_code == 200 and me.json()["email"] == "a@test.dev"

    ok = await client.post("/api/login", json={"email": "A@Test.dev ", "password": PASSWORD})
    assert ok.status_code == 200
    assert ok.json()["token"]

    wrong = await client.post("/api/login", json={"email": "a@test.dev", "password": "bad-password"})
    assert wrong.status_code == 401
    ghost = await client.post("/api/login", json={"email": "no@test.dev", "password": "whatever1"})
    assert ghost.json()["detail"] == wrong.json()["detail"]  # 不区分“无此人/密码错”

    dup = await client.post("/api/register", json={"email": "a@test.dev", "password": PASSWORD})
    assert dup.status_code == 409
    weak = await client.post("/api/register", json={"email": "w@test.dev", "password": "short"})
    assert weak.status_code == 422
    assert (await client.get("/api/me")).status_code == 401
    assert (await client.get("/api/me", headers=_auth("garbage"))).status_code == 401


# ---- 运行生命周期 ----


async def test_run_lifecycle_detail_and_list(client):
    token = await register(client)
    created = await client.post("/api/runs", json={"query": "  研究一下  "}, headers=_auth(token))
    assert created.status_code == 202
    run_id = created.json()["run_id"]

    body = await wait_status(client, token, run_id, {"completed"})
    assert body["query"] == "研究一下"  # strip 落库
    assert body["report_markdown"].startswith("# 研究报告")
    assert body["citations"][0]["id"] == "e1"
    assert (body["evidence_count"], body["source_count"]) == (3, 2)

    listing = await client.get("/api/runs", headers=_auth(token))
    assert [item["id"] for item in listing.json()] == [run_id]
    assert "report_markdown" not in listing.json()[0]  # 列表不带正文
    assert (await client.post("/api/runs", json={"query": "   "}, headers=_auth(token))).status_code == 422
    assert (await client.post("/api/runs", json={"query": "x" * 2001}, headers=_auth(token))).status_code == 422


async def test_cross_user_access_always_404(client):
    alice = await register(client, "alice@test.dev")
    bob = await register(client, "bob@test.dev")
    run_id = (
        await client.post("/api/runs", json={"query": "q"}, headers=_auth(alice))
    ).json()["run_id"]

    assert (await client.get(f"/api/runs/{run_id}", headers=_auth(bob))).status_code == 404
    assert (
        await client.post(f"/api/runs/{run_id}/cancel", headers=_auth(bob))
    ).status_code == 404
    async with client.stream(
        "GET", f"/api/runs/{run_id}/events", headers=_auth(bob)
    ) as response:
        assert response.status_code == 404
    assert (await client.get("/api/runs/nope", headers=_auth(alice))).status_code == 404


async def test_quota_blocks_third_concurrent_run(client):
    token = await register(client)
    first_gate, second_gate = asyncio.Event(), asyncio.Event()
    client.graphs.extend(
        [FakeGraph(gate=first_gate), FakeGraph(gate=second_gate)]
    )
    first = (
        await client.post("/api/runs", json={"query": "q1"}, headers=_auth(token))
    ).json()["run_id"]
    second = (
        await client.post("/api/runs", json={"query": "q2"}, headers=_auth(token))
    ).json()["run_id"]
    third = await client.post("/api/runs", json={"query": "q3"}, headers=_auth(token))
    assert third.status_code == 429
    first_gate.set()
    second_gate.set()
    await wait_status(client, token, first, {"completed", "failed", "cancelled"})
    await wait_status(client, token, second, {"completed", "failed", "cancelled"})


async def test_cancel_endpoint_converges_to_cancelled(client):
    token = await register(client)
    gate = asyncio.Event()
    client.graphs.append(FakeGraph(gate=gate))
    run_id = (
        await client.post("/api/runs", json={"query": "q"}, headers=_auth(token))
    ).json()["run_id"]
    await wait_status(client, token, run_id, {"running"})
    response = await client.post(f"/api/runs/{run_id}/cancel", headers=_auth(token))
    assert response.status_code == 200
    gate.set()
    body = await wait_status(client, token, run_id, {"cancelled"})
    assert body["terminal_reason"] == "user_cancelled"
    again = await client.post(f"/api/runs/{run_id}/cancel", headers=_auth(token))
    assert again.status_code == 200 and again.json()["status"] == "cancelled"


# ---- SSE ----


async def test_sse_replays_completed_run_and_ends_with_done(client):
    token = await register(client)
    client.graphs.append(
        FakeGraph(
            emit=[
                {
                    "event_type": "node_started",
                    "node": "supervisor",
                    "status": "started",
                    "payload": {},
                },
                {
                    "event_type": "direction_search_completed",
                    "payload": {"research_direction": "性善论溯源", "candidate_count": 5},
                },
            ]
        )
    )
    run_id = (
        await client.post("/api/runs", json={"query": "q"}, headers=_auth(token))
    ).json()["run_id"]
    await wait_status(client, token, run_id, {"completed"})

    frames = await read_sse(client, token, run_id)
    events = [event for event, _ in frames]
    assert events[-1] == "done"  # 不变式 3：done 恒为收尾
    ticks = _tick_events(frames)
    assert "正在拆解研究任务…" in ticks
    assert any("性善论溯源" in line and "5 条候选来源" in line for line in ticks)
    seqs = [data["seq"] for _, data in frames]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)  # 单调无重复
    done_data = frames[-1][1]
    assert done_data["status"] == "completed" and done_data["report_available"] is True


async def test_sse_live_backlog_and_gate_release(client):
    """订阅者接入于事件写入之后、运行结束之前：backlog + 实时无缝续接。"""
    token = await register(client)
    gate = asyncio.Event()
    client.graphs.append(
        FakeGraph(
            gate=gate,
            emit=[
                {
                    "event_type": "research_round_completed",
                    "payload": {
                        "round": 1,
                        "task_count": 2,
                        "completed_tasks": 2,
                        "evidence_added": 3,
                        "total_evidence_count": 3,
                    },
                }
            ],
        )
    )
    run_id = (
        await client.post("/api/runs", json={"query": "q"}, headers=_auth(token))
    ).json()["run_id"]
    reader = asyncio.create_task(read_sse(client, token, run_id))
    await asyncio.sleep(0.05)
    gate.set()
    frames = await reader
    events = [event for event, _ in frames]
    assert events[-1] == "done"
    assert "第 1 轮研究完成：方向 2/2，新增证据 3（累计 3）" in _tick_events(frames)


async def test_static_frontend_served(client):
    response = await client.get("/")
    assert response.status_code == 200
    assert "DeepSearch" in response.text
