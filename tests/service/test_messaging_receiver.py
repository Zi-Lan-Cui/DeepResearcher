"""拉取侧对长轮询「没消息」两种收尾的处理。

proxy 到期结束流时 SDK 抛 DEADLINE_EXCEEDED，剩余 deadline 不够跑满一轮时退回并抛
40018；两者都必须当空轮询，其余错误仍要上抛给调用方记录。
"""

from __future__ import annotations

from concurrent.futures import Future

import grpc
import pytest
from rocketmq.v5.exception import BadRequestException

from deepresearcher.service.messaging.receiver import RocketMQDispatchReceiver
from deepresearcher.service.settings import ServiceConfig

pytestmark = pytest.mark.asyncio


class _RpcError(grpc.RpcError):
    def __init__(self, code: grpc.StatusCode) -> None:
        self._code = code

    def code(self) -> grpc.StatusCode:
        return self._code


class _FailingConsumer:
    """receive_async 立即返回一个已带异常的 Future，复现 SDK 的上抛路径。"""

    def __init__(self, error: Exception) -> None:
        self._error = error

    def receive_async(self, max_message_num: int, invisible_duration: int) -> Future[list[object]]:
        future: Future[list[object]] = Future()
        future.set_exception(self._error)
        return future


def _receiver(consumer: _FailingConsumer) -> RocketMQDispatchReceiver:
    config = ServiceConfig(
        database_url="postgresql+asyncpg://postgres:@127.0.0.1:1/unused",
        jwt_secret="s" * 40,
    )
    receiver = RocketMQDispatchReceiver(config=config)
    receiver._consumer = consumer
    return receiver


async def test_deadline_exceeded_is_empty_poll():
    receiver = _receiver(_FailingConsumer(_RpcError(grpc.StatusCode.DEADLINE_EXCEEDED)))

    assert await receiver._pull_once() is None


async def test_other_rpc_error_propagates():
    receiver = _receiver(_FailingConsumer(_RpcError(grpc.StatusCode.UNAVAILABLE)))

    with pytest.raises(grpc.RpcError):
        await receiver._pull_once()


async def test_polling_window_rejection_is_empty_poll():
    """40018 是 proxy 退回这次长轮询，消息没被取走，等同于本轮没有消息。"""
    error = BadRequestException("The deadline time remaining is not enough for polling", 40018)
    receiver = _receiver(_FailingConsumer(error))

    assert await receiver._pull_once() is None


async def test_other_bad_request_propagates():
    """其余 4xxxx 是真正的请求缺陷，不能当空轮询吞掉。"""
    receiver = _receiver(_FailingConsumer(BadRequestException("illegal topic", 40003)))

    with pytest.raises(BadRequestException):
        await receiver._pull_once()
