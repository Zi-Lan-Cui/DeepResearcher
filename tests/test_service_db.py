import pytest
import pytest_asyncio
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from deepsearch_agent.service.persistence.database import init_db, make_engine, make_session_factory
from deepsearch_agent.service.persistence.models import Run, RunEvent, User

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def session(tmp_path):
    # 文件库（NullPool）而非 :memory:+StaticPool：与 test_service_runs 同理，
    # 规避 aiosqlite 线程在循环关闭后的迟到回调竞态。
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'db.test.db'}")
    await init_db(engine)
    factory = make_session_factory(engine)
    async with factory() as s:
        yield s
    await engine.dispose()


async def test_unique_email_enforced(session: AsyncSession):
    session.add(User(email="a@test", password_hash="h1"))
    await session.commit()
    session.add(User(email="a@test", password_hash="h2"))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_run_json_roundtrip_and_defaults(session: AsyncSession):
    user = User(email="b@test", password_hash="h")
    session.add(user)
    await session.flush()
    run = Run(id="run-abc", user_id=user.id, query="问题", citations_json=[{"id": "e1"}])
    session.add(run)
    await session.commit()

    loaded = await session.get(Run, "run-abc")
    assert loaded.status == "queued"
    assert loaded.citations_json == [{"id": "e1"}]
    assert loaded.evidence_count == 0


async def test_deleting_run_cascades_events(session: AsyncSession):
    """DB 级 ON DELETE CASCADE 生效（SQLite 需要 make_engine 打开 FK pragma）。"""
    user = User(email="c@test", password_hash="h")
    session.add(user)
    await session.flush()
    run = Run(id="run-casc", user_id=user.id, query="q")
    session.add(run)
    session.add_all(
        [
            RunEvent(run_id="run-casc", seq=1, event_type="node_started", record={"a": 1}),
            RunEvent(run_id="run-casc", seq=2, event_type="node_completed", record={}),
        ]
    )
    await session.commit()

    await session.delete(run)
    await session.commit()

    remaining = await session.scalar(select(func.count()).select_from(RunEvent))
    assert remaining == 0
    # 级联删的孤儿索引不会残留：再插入同 (run_id, seq) 也不冲突，这里仅验证计数为 0。
