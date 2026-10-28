"""发布侧单元测试，不连 broker。

只钉住把哪条回报发给 broker：顺序、异常归属、未启动时的拒绝。真 broker 语义见
test_messaging_rocketmq。
"""

from __future__ import annotations

import asyncio

import pytest
from rocketmq.grpc_protocol import TransactionResolution
from rocketmq.v5.model.message import Message

from deepresearcher.service.messaging import publisher as publisher_module
from deepresearcher.service.messaging.dispatch import encode_dispatch, new_dispatch
from deepresearcher.service.messaging.publisher import (
    RocketMQDispatchPublisher,
    RunExistsChecker,
)
from deepresearcher.service.settings import ServiceConfig

pytestmark = pytest.mark.asyncio

CONFIG = ServiceConfig(database_url="postgresql+asyncpg://x/y", jwt_secret="s" * 40)


class _FakeTransaction:
    """记录回报次数；可指定某一侧回报时抛错。"""

    def __init__(self, *, rollback_error=None, commit_error=None) -> None:
        self.committed = False
        self.rolled_back = False
        self._rollback_error = rollback_error
        self._commit_error = commit_error

    def commit(self) -> None:
        if self._commit_error is not None:
            raise self._commit_error
        self.committed = True

    def rollback(self) -> None:
        if self._rollback_error is not None:
            raise self._rollback_error
        self.rolled_back = True


class _FakeProducer:
    def __init__(self, *, send_error=None, transaction=None) -> None:
        self.sent: list[tuple[Message, object]] = []
        self.started = False
        self.shut_down = 0
        self._send_error = send_error
        self._transaction = transaction or _FakeTransaction()

    def send(self, message: Message, transaction=None) -> None:
        if self._send_error is not None:
            raise self._send_error
        self.sent.append((message, transaction))

    def begin_transaction(self) -> _FakeTransaction:
        return self._transaction

    def startup(self) -> None:
        self.started = True

    def shutdown(self) -> None:
        self.shut_down += 1


def _started(producer: _FakeProducer) -> RocketMQDispatchPublisher:
    publisher = RocketMQDispatchPublisher(config=CONFIG, session_factory=lambda: None)
    publisher._producer = producer  # type: ignore[assignment]
    return publisher


async def test_dispatch_sends_half_message_before_local_transaction():
    order: list[str] = []
    producer = _FakeProducer()
    original_send = producer.send

    def spy_send(message: Message, transaction=None) -> None:
        order.append("send")
        original_send(message, transaction)

    producer.send = spy_send  # type: ignore[method-assign]
    publisher = _started(producer)

    async def local_transaction() -> None:
        order.append("local")

    await publisher.dispatch("run-1", local_transaction)

    assert order == ["send", "local"]
    message, transaction = producer.sent[0]
    assert message.topic == CONFIG.rocketmq_dispatch_topic
    assert message.keys == {"run-1"}
    assert encode_dispatch(new_dispatch("run-1")) == message.body
    assert transaction is producer._transaction
    assert producer._transaction.committed is True


async def test_dispatch_rolls_back_and_reraises_local_error():
    transaction = _FakeTransaction()
    publisher = _started(_FakeProducer(transaction=transaction))

    async def local_transaction() -> None:
        raise ValueError("配额超限")

    with pytest.raises(ValueError, match="配额超限"):
        await publisher.dispatch("run-1", local_transaction)

    assert transaction.rolled_back is True
    assert transaction.committed is False


async def test_dispatch_keeps_local_error_when_rollback_report_fails():
    transaction = _FakeTransaction(rollback_error=RuntimeError("回报失败"))
    publisher = _started(_FakeProducer(transaction=transaction))

    async def local_transaction() -> None:
        raise ValueError("配额超限")

    with pytest.raises(ValueError, match="配额超限"):
        await publisher.dispatch("run-1", local_transaction)


async def test_dispatch_swallows_commit_report_failure():
    transaction = _FakeTransaction(commit_error=RuntimeError("回报失败"))
    publisher = _started(_FakeProducer(transaction=transaction))
    called = False

    async def local_transaction() -> None:
        nonlocal called
        called = True

    await publisher.dispatch("run-1", local_transaction)

    assert called is True
    assert transaction.committed is False


async def test_dispatch_skips_local_transaction_when_half_message_fails():
    publisher = _started(_FakeProducer(send_error=RuntimeError("半消息写入失败")))
    called = False

    async def local_transaction() -> None:
        nonlocal called
        called = True

    with pytest.raises(RuntimeError, match="半消息写入失败"):
        await publisher.dispatch("run-1", local_transaction)

    assert called is False


async def test_redispatch_uses_plain_topic_without_transaction():
    producer = _FakeProducer()
    publisher = _started(producer)

    await publisher.redispatch("run-1")

    message, transaction = producer.sent[0]
    assert message.topic == CONFIG.rocketmq_redispatch_topic
    assert transaction is None
    assert message.keys == {"run-1"}


async def test_unstarted_publisher_rejects_both_commands():
    publisher = RocketMQDispatchPublisher(config=CONFIG, session_factory=lambda: None)

    async def local_transaction() -> None:
        return None

    with pytest.raises(RuntimeError, match="尚未启动"):
        await publisher.dispatch("run-1", local_transaction)
    with pytest.raises(RuntimeError, match="尚未启动"):
        await publisher.redispatch("run-1")


async def test_close_shuts_producer_down_once():
    producer = _FakeProducer()
    publisher = _started(producer)

    await publisher.close()
    await publisher.close()

    assert producer.shut_down == 1
    assert publisher._producer is None


async def test_close_without_start_is_noop():
    publisher = RocketMQDispatchPublisher(config=CONFIG, session_factory=lambda: None)

    await publisher.close()


async def test_start_registers_both_topics_with_checker(monkeypatch):
    captured: dict[str, object] = {}

    class _RecordingProducer:
        def __init__(self, client_configuration, *, topics, checker) -> None:
            captured["topics"] = topics
            captured["checker"] = checker

        def startup(self) -> None:
            captured["started"] = True

    monkeypatch.setattr(publisher_module, "Producer", _RecordingProducer)
    publisher = RocketMQDispatchPublisher(config=CONFIG, session_factory=lambda: None)

    await publisher.start()

    assert captured["started"] is True
    assert set(captured["topics"]) == {
        CONFIG.rocketmq_dispatch_topic,
        CONFIG.rocketmq_redispatch_topic,
    }
    assert isinstance(captured["checker"], RunExistsChecker)


class _FakeSession:
    def __init__(self, *, value=None, error=None) -> None:
        self._value = value
        self._error = error

    async def __aenter__(self) -> _FakeSession:
        if self._error is not None:
            raise self._error
        return self

    async def __aexit__(self, *exc_info) -> bool:
        return False

    async def scalar(self, statement) -> object:
        return self._value


def _message(run_id: str = "run-1") -> Message:
    message = Message()
    message.body = encode_dispatch(new_dispatch(run_id))
    return message


async def test_check_commits_when_run_row_exists():
    checker = RunExistsChecker(lambda: _FakeSession(value="run-1"), asyncio.get_running_loop())

    # check 内部阻塞等 future.result，直接同步调用会把承载 _exists 的同一个循环堵死。
    assert await asyncio.to_thread(checker.check, _message()) == TransactionResolution.COMMIT


async def test_check_rolls_back_when_run_row_missing():
    checker = RunExistsChecker(lambda: _FakeSession(value=None), asyncio.get_running_loop())

    assert await asyncio.to_thread(checker.check, _message()) == TransactionResolution.ROLLBACK


async def test_check_rolls_back_empty_body():
    checker = RunExistsChecker(lambda: _FakeSession(value="run-1"), asyncio.get_running_loop())

    assert await asyncio.to_thread(checker.check, Message()) == TransactionResolution.ROLLBACK


async def test_check_rolls_back_undecodable_body():
    checker = RunExistsChecker(lambda: _FakeSession(value="run-1"), asyncio.get_running_loop())
    message = Message()
    message.body = b"not-json"

    assert await asyncio.to_thread(checker.check, message) == TransactionResolution.ROLLBACK


async def test_check_answers_unspecified_when_loop_closed():
    loop = asyncio.new_event_loop()
    loop.close()
    checker = RunExistsChecker(lambda: _FakeSession(value="run-1"), loop)

    assert checker.check(_message()) == TransactionResolution.TRANSACTION_RESOLUTION_UNSPECIFIED


async def test_check_answers_unspecified_when_database_unavailable():
    checker = RunExistsChecker(
        lambda: _FakeSession(error=RuntimeError("库连不上")),
        asyncio.get_running_loop(),
    )

    assert (
        await asyncio.to_thread(checker.check, _message())
        == TransactionResolution.TRANSACTION_RESOLUTION_UNSPECIFIED
    )
