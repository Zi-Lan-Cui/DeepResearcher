import pytest
import pytest_asyncio
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from deepresearcher.service.persistence.database import init_db, make_engine, make_session_factory
from deepresearcher.service.persistence.models import Run, RunEvent, User

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def session(tmp_path):
    # 用文件库而非 :memory:：后者每个连接各自一份库，跨 session 读不到同一份数据。
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
    """验证 DB 级 ON DELETE CASCADE 生效;SQLite 下依赖 make_engine 打开 FK pragma。"""
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
