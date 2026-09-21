import asyncio

import httpx
import pytest
import pytest_asyncio

from deepresearcher.service.api import create_app
from deepresearcher.service.events.ephemeral import EphemeralSubscription
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

    def graph_factory(
        *,
        settings,
        event_sink,
        trace_recorder,
        http_client,
        checkpointer=None,
        material_store=None,
        provider_health=None,
    ):
        del trace_recorder
        del material_store
        del provider_health
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
    response = await client.post("/api/register", json={"email": email, "password": PASSWORD})
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

    wrong = await client.post(
        "/api/login", json={"email": "a@test.dev", "password": "bad-password"}
    )
    assert wrong.status_code == 401
    ghost = await client.post("/api/login", json={"email": "no@test.dev", "password": "whatever1"})
    assert ghost.json()["detail"] == wrong.json()["detail"]  # 不区分“无此人/密码错”

    dup = await client.post("/api/register", json={"email": "a@test.dev", "password": PASSWORD})
    assert dup.status_code == 409
    weak = await client.post("/api/register", json={"email": "w@test.dev", "password": "short"})
    assert weak.status_code == 422
    assert (await client.get("/api/me")).status_code == 401
    assert (await client.get("/api/me", headers=_auth("garbage"))).status_code == 401


async def test_login_rate_limit_blocks_account_and_returns_retry_after(client):
    await register(client)
    payload = {"email": "a@test.dev", "password": "wrong-password"}

    for _ in range(5):
        assert (await client.post("/api/login", json=payload)).status_code == 401
    blocked = await client.post("/api/login", json=payload)

    assert blocked.status_code == 429
    assert blocked.json()["detail"] == "登录尝试过于频繁，请稍后再试。"
    assert int(blocked.headers["retry-after"]) > 0


async def test_successful_login_clears_only_the_account_budget(client):
    await register(client)
    wrong = {"email": "a@test.dev", "password": "wrong-password"}
    for _ in range(4):
        assert (await client.post("/api/login", json=wrong)).status_code == 401

    success = await client.post("/api/login", json={"email": "a@test.dev", "password": PASSWORD})
    assert success.status_code == 200

    # The previous account failures were cleared; a fresh full budget is available.
    for _ in range(5):
        assert (await client.post("/api/login", json=wrong)).status_code == 401
    assert (await client.post("/api/login", json=wrong)).status_code == 429


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
    assert (
        await client.post("/api/runs", json={"query": "   "}, headers=_auth(token))
    ).status_code == 422
    assert (
        await client.post("/api/runs", json={"query": "x" * 2001}, headers=_auth(token))
    ).status_code == 422


async def test_api_control_plane_does_not_execute_queued_run(tmp_path):
    def forbidden_graph_factory(**_kwargs):
        raise AssertionError("control-plane API must not build or execute a graph")

    app = create_app(
        service_settings(tmp_path),
        service_config(tmp_path, api_embedded_worker=False),
        graph_factory=forbidden_graph_factory,
    )
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://svc") as isolated:
            token = await register(isolated, "control@test.dev")
            created = await isolated.post(
                "/api/runs", json={"query": "queued for worker"}, headers=_auth(token)
            )
            assert created.status_code == 202
            await asyncio.sleep(0.1)
            detail = await isolated.get(
                f"/api/runs/{created.json()['run_id']}", headers=_auth(token)
            )
            assert detail.json()["status"] == "queued"
            assert app.state.execution is None
            cancelled = await isolated.post(
                f"/api/runs/{created.json()['run_id']}/cancel", headers=_auth(token)
            )
            assert cancelled.json()["status"] == "cancelled"
            frames = await read_sse(isolated, token, created.json()["run_id"])
            assert [event for event, _data in frames][-1] == "done"


async def test_cross_user_access_always_404(client):
    alice = await register(client, "alice@test.dev")
    bob = await register(client, "bob@test.dev")
    run_id = (await client.post("/api/runs", json={"query": "q"}, headers=_auth(alice))).json()[
        "run_id"
    ]

    assert (await client.get(f"/api/runs/{run_id}", headers=_auth(bob))).status_code == 404
    assert (await client.post(f"/api/runs/{run_id}/cancel", headers=_auth(bob))).status_code == 404
    async with client.stream("GET", f"/api/runs/{run_id}/events", headers=_auth(bob)) as response:
        assert response.status_code == 404
    assert (await client.get("/api/runs/nope", headers=_auth(alice))).status_code == 404


async def test_quota_blocks_third_concurrent_run(client):
    token = await register(client)
    first_gate, second_gate = asyncio.Event(), asyncio.Event()
    client.graphs.extend([FakeGraph(gate=first_gate), FakeGraph(gate=second_gate)])
    first = (await client.post("/api/runs", json={"query": "q1"}, headers=_auth(token))).json()[
        "run_id"
    ]
    second = (await client.post("/api/runs", json={"query": "q2"}, headers=_auth(token))).json()[
        "run_id"
    ]
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
    run_id = (await client.post("/api/runs", json={"query": "q"}, headers=_auth(token))).json()[
        "run_id"
    ]
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
                    "event_type": "research_task_started",
                    "payload": {"task_id": "task-0001", "question": "方向甲的局部事实"},
                },
                {
                    "event_type": "direction_search_completed",
                    "payload": {
                        "task_id": "task-0001",
                        "research_direction": "方向甲",
                        "candidate_count": 5,
                    },
                },
                {
                    "event_type": "research_task_completed",
                    "payload": {
                        "task_id": "task-0001",
                        "execution_status": "completed",
                        "evidence_count": 2,
                        "source_count": 1,
                    },
                },
            ]
        )
    )
    run_id = (await client.post("/api/runs", json={"query": "q"}, headers=_auth(token))).json()[
        "run_id"
    ]
    await wait_status(client, token, run_id, {"completed"})

    frames = await read_sse(client, token, run_id)
    events = [event for event, _ in frames]
    assert events[-1] == "done"  # 不变式 3：done 恒为收尾
    stage_opens = [data["stage"] for event, data in frames if event == "stage_open"]
    assert "supervisor" in stage_opens  # 阶段块由 node_started 事件驱动出现
    by_event = {event: data for event, data in frames if event.startswith("task_")}
    assert by_event["task_open"]["title"] == "方向甲的局部事实"
    assert by_event["task_update"]["text"] == "检索完成：5 条候选来源"
    assert by_event["task_done"]["summary"] == "证据 2 · 来源 1"
    assert by_event["task_update"]["task"] == "task-0001"
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
    run_id = (await client.post("/api/runs", json={"query": "q"}, headers=_auth(token))).json()[
        "run_id"
    ]
    reader = asyncio.create_task(read_sse(client, token, run_id))
    await asyncio.sleep(0.05)
    gate.set()
    frames = await reader
    events = [event for event, _ in frames]
    assert events[-1] == "done"
    stats = [data for event, data in frames if event == "stats"]
    assert stats and stats[0]["round"] == 1
    assert stats[0]["evidence_total"] == 3


async def test_sse_merges_cross_process_preview_with_durable_events(client):
    """Redis 预览只补充临时 token，done 仍由持久事件收尾。"""

    class RecordingBus:
        def __init__(self):
            self.queue = None
            self.closed = False

        async def subscribe(self, _run_id):
            self.queue = asyncio.Queue()

            async def close():
                self.closed = True

            return EphemeralSubscription(queue=self.queue, _close=close)

    token = await register(client)
    gate = asyncio.Event()
    client.graphs.append(FakeGraph(gate=gate))
    run_id = (await client.post("/api/runs", json={"query": "q"}, headers=_auth(token))).json()[
        "run_id"
    ]
    bus = RecordingBus()
    client.app.state.ephemeral_bus = bus
    reader = asyncio.create_task(read_sse(client, token, run_id))
    async with asyncio.timeout(1):
        while bus.queue is None:
            await asyncio.sleep(0)
    await bus.queue.put(
        {
            "run_id": run_id,
            "event_type": "text_delta",
            "payload": {"channel": "supervisor", "text": "跨进程预览"},
        }
    )
    await asyncio.sleep(0.05)
    gate.set()

    frames = await reader
    previews = [data for event, data in frames if event == "text_delta"]
    assert previews == [{"channel": "supervisor", "text": "跨进程预览"}]
    assert frames[-1][0] == "done"
    assert bus.closed


async def test_static_frontend_served(client):
    response = await client.get("/")
    assert response.status_code == 200
    assert "DeepResearcher" in response.text
    assert 'id="report-copy"' in response.text
    assert 'id="report-download"' in response.text
    assert "html2pdf.bundle.min.js" in response.text
    assert '<link rel="stylesheet" href="/styles.css?v=' in response.text
    assert '<script type="module" src="/app.js?v=' in response.text
    assert "<style>" not in response.text

    stylesheet = await client.get("/styles.css")
    assert stylesheet.status_code == 200
    assert stylesheet.headers["content-type"].startswith("text/css")
    assert ".stage-block.running" in stylesheet.text

    application = await client.get("/app.js")
    assert application.status_code == 200
    assert "javascript" in application.headers["content-type"]
    assert "reportExportText()" in application.text
    assert "createPdfExportNode()" in application.text
    assert ".from(exported.report)" in application.text
    assert 'avoid: [".pdf-paragraph-fragment", "li", "table"' in application.text
    assert 'report.removeAttribute("id")' in application.text
    assert "splitLongPdfParagraphs(report)" in application.text
    assert 'setStatus(resumed.status || "queued")' in application.text
    assert 'setStatus("running")' not in application.text
    assert ".pdf-export-host" in stylesheet.text
    assert "page-break-inside:avoid" in stylesheet.text


async def test_concurrent_register_same_email_single_winner(client):
    """快速路径查重可被并发穿透，UNIQUE 索引兜底且必须翻成 409 而非 500。"""
    payload = {"email": "twin@test.dev", "password": PASSWORD}
    first, second = await asyncio.gather(
        client.post("/api/register", json=payload),
        client.post("/api/register", json=payload),
    )
    codes = sorted([first.status_code, second.status_code])
    assert codes == [201, 409], (first.status_code, second.status_code)
    ok = await client.post("/api/login", json=payload)
    assert ok.status_code == 200  # 恰有一个账号存在且可登录


async def test_sse_synthesizes_done_when_persisted_done_frame_missing(client):
    """done 帧整批丢在失败 flush 里的场景:行状态才是终止权威,
    SSE 必须靠兜底合成 done 收敛,而不是每秒轮询挂着不返回。"""
    token = await register(client)
    client.graphs.append(FakeGraph())
    run_id = (await client.post("/api/runs", json={"query": "q"}, headers=_auth(token))).json()[
        "run_id"
    ]
    await wait_status(client, token, run_id, {"completed"})

    from sqlalchemy import delete

    from deepresearcher.service.persistence.models import RunEvent

    async with client.app.state.session_factory() as session:
        await session.execute(
            delete(RunEvent).where(RunEvent.run_id == run_id, RunEvent.event_type == "run_done")
        )
        await session.commit()

    frames = await read_sse(client, token, run_id, timeout=8.0)
    events = [event for event, _ in frames]
    assert events[-1] == "done"
    assert frames[-1][1]["status"] == "completed"
