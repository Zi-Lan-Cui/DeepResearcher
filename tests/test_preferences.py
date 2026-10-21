"""个人研究偏好:纯函数校验/冻结/应用,settings 与 stats 端点,创建快照不回溯。"""

import asyncio

import httpx
import pytest
import pytest_asyncio

from deepresearcher.config import AgentConfig
from deepresearcher.service.api import create_app
from deepresearcher.service.execution.runtime import worker_lifespan
from deepresearcher.service.persistence.models import Run
from deepresearcher.service.preferences import (
    apply_preferences,
    clamp_preferences,
    frozen_snapshot,
)
from fakes_service import FakeGraph, service_config, service_settings

PASSWORD = "goodpassword"
FULL = {
    "max_research_rounds": 4,
    "max_subtasks_per_round": 8,
    "research_agent_max_turns": 20,
    "supervisor_max_active_evidences": 50,
}


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def _register(client, email="cfg@test.dev") -> str:
    response = await client.post("/api/register", json={"email": email, "password": PASSWORD})
    assert response.status_code == 201
    return response.json()["token"]


async def _settle(client, token: str, run_id: str) -> None:
    """取消并等待进入终态：fixture 收尾时不留进行中的 run，避开 aiosqlite 写锁竞态。"""
    await client.post(f"/api/runs/{run_id}/cancel", headers=_auth(token))
    async with asyncio.timeout(5.0):
        while True:
            body = (await client.get(f"/api/runs/{run_id}", headers=_auth(token))).json()
            if body["status"] in {"completed", "failed", "cancelled"}:
                return
            await asyncio.sleep(0.02)


@pytest_asyncio.fixture
async def client(tmp_path):
    def graph_factory(**_kwargs):
        return FakeGraph()

    settings = service_settings(tmp_path)
    config = service_config(tmp_path, worker_poll_seconds=0.01)
    async with worker_lifespan(settings, config, graph_factory=graph_factory):
        app = create_app(settings, config)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as http:
                http.app = app  # type: ignore[attr-defined]
                yield http


def test_clamp_rejects_missing_or_illegal_keys():
    assert clamp_preferences(dict(FULL)) == FULL
    with pytest.raises(ValueError):
        clamp_preferences({**FULL, "max_research_rounds": 99})  # 越枚举
    with pytest.raises(ValueError):
        clamp_preferences({k: v for k, v in FULL.items() if k != "max_research_rounds"})  # 缺键
    with pytest.raises(ValueError):
        clamp_preferences({**FULL, "max_parallel_workers": 5})  # 并发不进用户面


def test_frozen_snapshot_precedence_and_dirty_fallback():
    agent = AgentConfig()
    # 未落项回服务器默认
    snapshot = frozen_snapshot({"max_research_rounds": 1}, agent)
    assert snapshot["max_research_rounds"] == 1
    assert snapshot["max_subtasks_per_round"] == agent.max_subtasks_per_round
    # 脏值(枚举外)按默认，不采纳
    dirty = frozen_snapshot({"max_research_rounds": 99}, agent)
    assert dirty["max_research_rounds"] == agent.max_research_rounds


def test_apply_preferences_overrides_agent_and_ignores_illegal():
    agent = AgentConfig()
    applied = apply_preferences(agent, FULL)
    assert applied.max_research_rounds == 4
    assert applied.supervisor_max_active_evidences == 50
    # 空快照或越枚举项静默回服务器值，绝不让脏数据生效
    assert apply_preferences(agent, None) is agent
    assert apply_preferences(agent, {"max_research_rounds": True, "bogus": 1}) is agent


@pytest.mark.asyncio
async def test_preferences_roundtrip_and_validation(client):
    token = await _register(client)
    initial = await client.get("/api/preferences", headers=_auth(token))
    assert initial.status_code == 200
    body = initial.json()
    assert set(body["choices"]) == set(FULL)
    assert "max_parallel_workers" not in str(body)  # 并发数绝不出现在用户面
    assert body["preferences"] == body["server_defaults"]  # 未改过=服务器默认

    bad = await client.post(
        "/api/preferences",
        json={"preferences": {**FULL, "max_research_rounds": 9}},
        headers=_auth(token),
    )
    assert bad.status_code == 422
    saved = await client.post("/api/preferences", json={"preferences": FULL}, headers=_auth(token))
    assert saved.status_code == 200
    after = await client.get("/api/preferences", headers=_auth(token))
    assert after.json()["preferences"] == FULL


@pytest.mark.asyncio
async def test_run_snapshot_frozen_and_not_retroactive(client):
    token = await _register(client, email="snap@test.dev")
    await client.post("/api/preferences", json={"preferences": FULL}, headers=_auth(token))
    created = await client.post("/api/runs", json={"query": "配置冻结测试"}, headers=_auth(token))
    assert created.status_code == 202
    run_id = created.json()["run_id"]

    session_factory = client.app.state.session_factory  # type: ignore[attr-defined]
    async with session_factory() as session:
        first = await session.get(Run, run_id)
        assert first is not None and first.preferences_snapshot == FULL

    # 之后改用户配置不回溯已建 run:再建一场取新值，旧行仍是 FULL
    changed = {**FULL, "max_research_rounds": 1}
    await client.post("/api/preferences", json={"preferences": changed}, headers=_auth(token))
    second = await client.post("/api/runs", json={"query": "第二场"}, headers=_auth(token))
    second_run_id = second.json()["run_id"]
    async with session_factory() as session:
        old = await session.get(Run, run_id)
        new = await session.get(Run, second_run_id)
        assert old is not None and old.preferences_snapshot == FULL
        assert new is not None and new.preferences_snapshot == changed

    await _settle(client, token, run_id)
    await _settle(client, token, second_run_id)


@pytest.mark.asyncio
async def test_stats_daily_shape_after_run(client):
    token = await _register(client, email="stats@test.dev")
    created = await client.post("/api/runs", json={"query": "统计形状"}, headers=_auth(token))
    assert created.status_code == 202
    stats = await client.get("/api/stats/daily?days=7", headers=_auth(token))
    assert stats.status_code == 200
    payload = stats.json()
    assert payload["days"] == 7
    assert len(payload["series"]) >= 1
    latest = payload["series"][-1]
    assert {"date", "runs", "input_tokens", "output_tokens", "cached_input_tokens"} <= set(latest)
    # days 越界回落默认窗口，不报错
    fallback = await client.get("/api/stats/daily?days=999", headers=_auth(token))
    assert fallback.json()["days"] == 30
    await _settle(client, token, created.json()["run_id"])
