import pytest

from deepresearcher.service.settings import (
    ServiceConfig,
    clear_service_config_cache,
    get_service_config,
)


@pytest.fixture
def _clean_env(monkeypatch, tmp_path):
    """隔离真实 env/.env：指向不存在的 env 文件并清掉相关环境变量。"""
    monkeypatch.setenv("DEEPRESEARCHER_ENV_FILE", str(tmp_path / "absent.env"))
    for name in (
        "SERVICE_DATABASE_URL",
        "SERVICE_JWT_SECRET",
        "SERVICE_TOKEN_TTL_HOURS",
        "SERVICE_MAX_CONCURRENT_RUNS_PER_USER",
        "SERVICE_MAX_GLOBAL_QUEUED_RUNS",
        "SERVICE_WORKER_MAX_RUNNING_RUNS",
        "SERVICE_WORKER_HEARTBEAT_SECONDS",
        "SERVICE_REDIS_PREVIEW_ENABLED",
        "SERVICE_REDIS_URL",
        "SERVICE_REDIS_CHANNEL_PREFIX",
        "SERVICE_REDIS_PREVIEW_QUEUE_SIZE",
        "SERVICE_LOGIN_ACCOUNT_ATTEMPTS",
        "SERVICE_LOGIN_IP_ATTEMPTS",
        "SERVICE_LOGIN_RATE_WINDOW_SECONDS",
        "SERVICE_LOGIN_BLOCK_SECONDS",
        "SERVICE_FORWARDED_ALLOW_IPS",
        "SERVICE_PORT",
        "SERVICE_RUN_TRACE_RETENTION_DAYS",
        "SERVICE_RETENTION_SWEEP_INTERVAL_SECONDS",
        "APP_ENV",
    ):
        monkeypatch.delenv(name, raising=False)
    clear_service_config_cache()
    yield
    clear_service_config_cache()


def _long_secret() -> str:
    return "s" * 40


def test_missing_database_url_raises_with_compose_hint(_clean_env, monkeypatch):
    monkeypatch.setenv("SERVICE_JWT_SECRET", _long_secret())
    with pytest.raises(ValueError, match="postgresql\\+asyncpg"):
        get_service_config()


def test_development_falls_back_to_ephemeral_secret(_clean_env, monkeypatch):
    monkeypatch.setenv("SERVICE_DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    config = get_service_config()  # APP_ENV 默认 development
    assert len(config.jwt_secret) >= 32
    assert config.token_ttl_hours == 12
    assert config.max_concurrent_runs_per_user == 2
    assert config.worker_max_running_runs == 3
    assert config.max_global_queued_runs == 100
    assert config.worker_lease_check_seconds == 15
    assert config.jsonl_retention_days == 14
    assert config.run_trace_retention_days == 14
    assert config.retention_sweep_interval_seconds == 21_600
    # 唯一租约：不可见时长就是接管时延，续期必须明显更密，容得下连续错过的续期。
    assert config.rocketmq_invisible_seconds == 90
    assert config.rocketmq_visibility_renew_seconds == 30
    assert config.rocketmq_invisible_seconds >= config.rocketmq_visibility_renew_seconds * 2
    # 租约确认比续租更勤，否则一次读库抖动要把租约拖过期。
    assert config.worker_lease_check_seconds <= config.rocketmq_visibility_renew_seconds // 2
    assert config.rocketmq_consumer_group == "deepresearcher-worker"
    # 死信判据：下限 2 在代码里钳（上限为 1 时第一条消息就可判死信，等于关掉入站）；
    # 默认值必须小于 proxy 的 maxAttempts（消费组 retryMaxTimes + 1，默认 17），
    # 否则消息先被转进 %DLQ%，结算分支永远触发不到。
    assert config.rocketmq_dispatch_max_attempts == 15
    assert config.rocketmq_dispatch_max_attempts < 17
    # 简单消费者一次只轮询一个 topic，长轮询时长直接变成另一个 topic 上的分派延迟。
    assert config.rocketmq_await_duration_seconds <= 5
    assert config.redis_preview_enabled is False
    assert config.redis_url == "redis://127.0.0.1:6379/0"
    assert config.redis_preview_queue_size == 128
    assert config.login_account_attempts == 5
    assert config.login_ip_attempts == 20
    assert config.login_rate_window_seconds == 300
    assert config.login_block_seconds == 900
    assert config.forwarded_allow_ips == "127.0.0.1"
    assert config.port == 8080
    assert config.jsonl_events is True


def test_production_rejects_short_secret(_clean_env, monkeypatch):
    monkeypatch.setenv("SERVICE_DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("SERVICE_JWT_SECRET", "tooshort")
    with pytest.raises(ValueError, match="SERVICE_JWT_SECRET"):
        get_service_config()


def test_env_overrides_are_clamped(_clean_env, monkeypatch):
    monkeypatch.setenv("SERVICE_DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    monkeypatch.setenv("SERVICE_JWT_SECRET", _long_secret())
    monkeypatch.setenv("SERVICE_TOKEN_TTL_HOURS", "0")  # 非法 → 夹到最小 1
    monkeypatch.setenv("SERVICE_MAX_CONCURRENT_RUNS_PER_USER", "not-a-number")  # 回落默认
    monkeypatch.setenv("SERVICE_REDIS_PREVIEW_ENABLED", "true")
    monkeypatch.setenv("SERVICE_REDIS_PREVIEW_QUEUE_SIZE", "0")
    monkeypatch.setenv("SERVICE_RUN_TRACE_RETENTION_DAYS", "-5")
    monkeypatch.setenv("SERVICE_RETENTION_SWEEP_INTERVAL_SECONDS", "1")
    config = get_service_config()
    assert config.token_ttl_hours == 1
    assert config.run_trace_retention_days == 0
    assert config.retention_sweep_interval_seconds == 60
    assert config.max_concurrent_runs_per_user == 2
    assert config.redis_preview_enabled is True
    assert config.redis_preview_queue_size == 1


def test_explicit_config_usable_without_env(_clean_env):
    config = ServiceConfig(database_url="x", jwt_secret="y")
    assert config.token_ttl_hours == 12


def test_checkpoint_dsn_derivation():
    from deepresearcher.service.settings import checkpoint_dsn

    assert (
        checkpoint_dsn("postgresql+asyncpg://u:p@localhost:5432/db")
        == "postgresql://u:p@localhost:5432/db"
    )
    assert checkpoint_dsn("postgresql://plain/url") == "postgresql://plain/url"
    # 非 PG 直接拒绝，不伪造一个看似可用的 DSN
    with pytest.raises(ValueError, match="只支持 PostgreSQL"):
        checkpoint_dsn("sqlite+aiosqlite:///x.db")
