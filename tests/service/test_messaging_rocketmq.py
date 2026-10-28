"""消息队列通道对真实 RocketMQ 的端到端验证。

只在显式配置了接入点时才跑，验证真 Broker 是否按契约收发。

覆盖三条语义：
① 事务消息在本地事务提交后才可见（commit 路径能收到）；
② 本地事务回滚则半消息被丢弃（rollback 路径收不到，且不等可见性超时）；
③ 补偿用的普通消息不经过回查，直接可消费。

消费组每次随机：RocketMQ 的新消费组默认从最新位点开始，复用组名会把上一次
用例的位点带进来，断言就不可信了。
"""

from __future__ import annotations

import asyncio
import os

import pytest
from sqlalchemy import delete

from deepresearcher.observability.span import new_id
from deepresearcher.service.messaging.publisher import RocketMQDispatchPublisher
from deepresearcher.service.messaging.receiver import RocketMQDispatchReceiver
from deepresearcher.service.settings import ServiceConfig
from deepresearcher.service.storage.engine import make_engine, make_session_factory
from deepresearcher.service.storage.tables import Run, User
from pg_backend import database_url_for

ENDPOINTS_ENV = "SERVICE_TEST_ROCKETMQ_ENDPOINTS"

pytestmark = pytest.mark.asyncio

#: 收消息的等待上限：Broker 侧最长可见性/长轮询都在秒级，留足余量但不无限等。
RECEIVE_TIMEOUT_SECONDS = 20.0

#: 单次取消息的等待上限。入站实现是预取队列，取不到就阻塞，所以每次取都得带超时，
#: 否则「沉默窗口」这种「等到时间就该下结论」的断言会永远等下去。
RECEIVE_POLL_SECONDS = 2.0

#: 「不该收到」的观察窗口：足够覆盖一次正常投递延迟。
SILENCE_SECONDS = 8.0


def _endpoints() -> str:
    endpoints = os.getenv(ENDPOINTS_ENV, "").strip()
    if not endpoints:
        pytest.skip(f"需要 RocketMQ：把 {ENDPOINTS_ENV} 指向 proxy 接入点")
    return endpoints


def _config(endpoints: str, database_url: str, *, group_suffix: str) -> ServiceConfig:
    return ServiceConfig(
        database_url=database_url,
        jwt_secret="s" * 40,
        rocketmq_endpoints=endpoints,
        rocketmq_consumer_group=f"deepresearcher-worker-{group_suffix}",
        # 用例要求「收不到」时能快速判定：不可见时长与长轮询都取小值。
        # 长轮询直接决定单次 receive 的阻塞上限，也是「沉默窗口」的粒度。
        rocketmq_invisible_seconds=30,
        rocketmq_visibility_renew_seconds=10,
        rocketmq_await_duration_seconds=2,
    )


async def _receive_one(receiver: RocketMQDispatchReceiver, run_id: str) -> bool:
    """在超时内捞到这条 run 的分派消息并 ACK 掉。

    长轮询可能以 RPC 错误结束（DEADLINE_EXCEEDED 就是空轮询被 proxy 移除流时的
    表现），所以这里按「本轮没取到」处理；真正的健康问题会让下面的断言失败。
    """
    deadline = asyncio.get_running_loop().time() + RECEIVE_TIMEOUT_SECONDS
    while asyncio.get_running_loop().time() < deadline:
        delivery = await _receive_quietly(receiver)
        if delivery is None:
            continue
        # 别的 run 的消息（同组并发用例遗留）也确认掉，避免卡在队首。
        await receiver.ack(delivery)
        if delivery.message.run_id == run_id:
            return True
    return False


async def _receive_quietly(
    receiver: RocketMQDispatchReceiver, *, timeout: float = RECEIVE_POLL_SECONDS
):
    """取一条；等不到或瞬时 RPC 错误都当作没取到。"""
    try:
        return await asyncio.wait_for(receiver.receive(), timeout=timeout)
    except Exception:  # noqa: BLE001 - 超时、DEADLINE_EXCEEDED、网络抖动都是「这轮没有」
        return None


async def test_commit_makes_dispatch_visible_and_rollback_drops_it(tmp_path):
    endpoints = _endpoints()
    database_url = database_url_for(tmp_path)
    engine = make_engine(database_url)
    factory = make_session_factory(engine)
    async with factory() as session:
        # id 交给序列：写死 id=1 会把 users 的序列留在后面，同库上跑真实进程
        # 验收时下一次注册就会撞主键，报成「邮箱已注册」。
        user = User(email="messaging@test.invalid", password_hash="h")
        session.add(user)
        await session.commit()
        user_id = user.id

    config = _config(endpoints, database_url, group_suffix=new_id("g"))
    publisher = RocketMQDispatchPublisher(config=config, session_factory=factory)
    receiver = RocketMQDispatchReceiver(config=config)
    committed_run = new_id("run")
    rolled_back_run = new_id("run")
    try:
        await publisher.start()
        # 消费者先起：新消费组从最新位点开始，先发后订阅会漏掉消息。
        await receiver.start()

        async def commit_run() -> None:
            async with factory() as session:
                async with session.begin():
                    session.add(
                        Run(id=committed_run, user_id=user_id, query="e2e", status="queued")
                    )

        await publisher.dispatch(committed_run, commit_run)
        assert await _receive_one(receiver, committed_run), "本地事务已提交，消息必须可见"

        async def rollback_run() -> None:
            raise RuntimeError("local transaction failed")

        with pytest.raises(RuntimeError, match="local transaction failed"):
            await publisher.dispatch(rolled_back_run, rollback_run)
        # 回滚：半消息不该转正。等满一个观察窗口仍收不到才算数。
        deadline = asyncio.get_running_loop().time() + SILENCE_SECONDS
        seen = False
        while asyncio.get_running_loop().time() < deadline:
            delivery = await _receive_quietly(receiver)
            if delivery is None:
                continue
            await receiver.ack(delivery)
            if delivery.message.run_id == rolled_back_run:
                seen = True
                break
        assert not seen, "本地事务回滚了，这条分派不该被投递"

        # 补偿路径用普通消息：不经过回查，直接可消费。
        await publisher.redispatch(committed_run)
        assert await _receive_one(receiver, committed_run), "补发的普通消息必须可见"
    finally:
        await receiver.close()
        await publisher.close()
        # 这条用例往共享库里写过行，自己收拾干净：别的用例按序列取 id 时不该被污染。
        async with factory() as session:
            await session.execute(delete(Run).where(Run.user_id == user_id))
            await session.execute(delete(User).where(User.id == user_id))
            await session.commit()
        await engine.dispose()
