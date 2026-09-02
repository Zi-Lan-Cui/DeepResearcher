import pytest

from deepsearch_agent.service.settings import (
    ServiceConfig,
    clear_service_config_cache,
    get_service_config,
)


@pytest.fixture
def _clean_env(monkeypatch, tmp_path):
    """隔离真实 env/.env：指向不存在的 env 文件并清掉相关环境变量。"""
    monkeypatch.setenv("DEEPSEARCH_ENV_FILE", str(tmp_path / "absent.env"))
    for name in (
        "SERVICE_DATABASE_URL",
        "SERVICE_JWT_SECRET",
        "SERVICE_TOKEN_TTL_HOURS",
        "SERVICE_MAX_CONCURRENT_RUNS_PER_USER",
        "SERVICE_MAX_GLOBAL_RUNNING_RUNS",
        "SERVICE_MAX_GLOBAL_QUEUED_RUNS",
        "SERVICE_WORKER_LEASE_SECONDS",
        "SERVICE_WORKER_HEARTBEAT_SECONDS",
        "SERVICE_PORT",
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
    monkeypatch.setenv("SERVICE_DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    config = get_service_config()  # APP_ENV 默认 development
    assert len(config.jwt_secret) >= 32
    assert config.token_ttl_hours == 12
    assert config.max_concurrent_runs_per_user == 2
    assert config.max_global_running_runs == 3
    assert config.max_global_queued_runs == 100
    assert config.worker_lease_seconds == 60
    assert config.worker_heartbeat_seconds == 20
    assert config.port == 8080
    assert config.jsonl_events is True


def test_production_rejects_short_secret(_clean_env, monkeypatch):
    monkeypatch.setenv("SERVICE_DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("SERVICE_JWT_SECRET", "tooshort")
    with pytest.raises(ValueError, match="SERVICE_JWT_SECRET"):
        get_service_config()


def test_env_overrides_are_clamped(_clean_env, monkeypatch):
    monkeypatch.setenv("SERVICE_DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    monkeypatch.setenv("SERVICE_JWT_SECRET", _long_secret())
    monkeypatch.setenv("SERVICE_TOKEN_TTL_HOURS", "0")  # 非法 → 夹到最小 1
    monkeypatch.setenv("SERVICE_MAX_CONCURRENT_RUNS_PER_USER", "not-a-number")  # 回落默认
    config = get_service_config()
    assert config.token_ttl_hours == 1
    assert config.max_concurrent_runs_per_user == 2


def test_explicit_config_usable_without_env(_clean_env):
    config = ServiceConfig(database_url="x", jwt_secret="y")
    assert config.token_ttl_hours == 12


def test_checkpoint_dsn_derivation():
    from deepsearch_agent.service.settings import checkpoint_dsn

    assert (
        checkpoint_dsn("postgresql+asyncpg://u:p@localhost:5432/db")
        == "postgresql://u:p@localhost:5432/db"
    )
    assert checkpoint_dsn("postgresql://plain/url") == "postgresql://plain/url"
    # 非 PG 部署（测试 SQLite）不假装支持恢复
    assert checkpoint_dsn("sqlite+aiosqlite:///x.db") is None
