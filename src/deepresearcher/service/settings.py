"""服务层配置：数据库、JWT、监听与配额。

引擎配置（LLM、搜索、agent 预算）仍由 deepresearcher.config.get_settings() 提供，
本模块只定义把它部署成多用户服务所需的新键。env 文件沿用同一约定：
DEEPRESEARCHER_ENV_FILE，默认 env/.env，load_dotenv(override=False)
不会与引擎侧的加载互相覆盖；环境判别复用 APP_ENV。
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from deepresearcher.config import project_path_env
from deepresearcher.env import env_bool as _bool_env
from deepresearcher.env import env_float as _float_env
from deepresearcher.env import env_int as _int_env
from deepresearcher.env import env_str as _env
from deepresearcher.observability.logging_config import get_logger

# service/settings.py 位于 src/deepresearcher/service/ 下，比 config.py 深一层。
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

logger = get_logger("deepresearcher.service.settings")

# 短于此长度的 secret 会被视为未正确配置：HS256 暴力成本与密钥熵直接挂钩。
_MIN_SECRET_CHARS = 32


@dataclass(frozen=True)
class ServiceConfig:
    database_url: str
    jwt_secret: str
    token_ttl_hours: int = 12
    max_concurrent_runs_per_user: int = 2
    max_global_queued_runs: int = 100
    # 单个 worker 进程的同时执行槽位；多副本部署时全局总量等于该值乘以副本数，
    # 对上游供应商的总量保护由每进程的 LLM_PROVIDER_RPM/TPM 按副本数折算承担。
    # 与 max_global_queued_runs 是两件事：那是准入侧排队长度上限，这是执行侧并发数。
    worker_max_running_runs: int = 3
    worker_lease_seconds: int = 60
    worker_heartbeat_seconds: int = 20
    worker_poll_seconds: float = 15.0
    redis_preview_enabled: bool = False
    redis_url: str = "redis://127.0.0.1:6379/0"
    redis_channel_prefix: str = "deepresearcher"
    redis_preview_queue_size: int = 128
    material_store_backend: str = "memory"
    material_redis_url: str = "redis://127.0.0.1:6379/0"
    material_key_prefix: str = "deepresearcher:material"
    search_material_ttl_seconds: int = 6 * 60 * 60
    document_material_ttl_seconds: int = 24 * 60 * 60
    login_account_attempts: int = 5
    login_ip_attempts: int = 20
    login_rate_window_seconds: int = 5 * 60
    login_block_seconds: int = 15 * 60
    forwarded_allow_ips: str = "127.0.0.1"
    host: str = "127.0.0.1"
    port: int = 8080
    service_log_dir: Path = _PROJECT_ROOT / "var" / "service"
    jsonl_events: bool = True
    jsonl_retention_days: int = 14
    # 终态 run 过程数据(events/usage 行 + checkpoint 线程)的保留窗口与清扫间隔；
    # 报告与 runs 行不在清除范围。retention_days=0 关闭清扫。
    run_trace_retention_days: int = 14
    retention_sweep_interval_seconds: int = 21_600
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = ""
    otel_exporter: str = "none"


@lru_cache(maxsize=1)
def get_service_config() -> ServiceConfig:
    """加载并缓存服务配置；测试请显式构造 :class:`ServiceConfig` 或先清缓存。"""
    env_file = Path(_env("DEEPRESEARCHER_ENV_FILE", str(_PROJECT_ROOT / "env" / ".env")))
    load_dotenv(env_file, override=False)
    database_url = _env("SERVICE_DATABASE_URL")
    if not database_url:
        raise ValueError(
            "缺少 SERVICE_DATABASE_URL。本地 docker-compose 对应值："
            "postgresql+asyncpg://deepsearch:deepsearch@localhost:5432/deepsearch"
        )
    environment = _env("APP_ENV", "development").lower()
    jwt_secret = _env("SERVICE_JWT_SECRET")
    if len(jwt_secret) < _MIN_SECRET_CHARS:
        if environment == "production":
            raise ValueError(f"生产环境必须提供长度 ≥{_MIN_SECRET_CHARS} 的 SERVICE_JWT_SECRET。")
        # 开发环境缺省时每个进程随机生成一份，代价是重启后全部旧 token 失效。
        jwt_secret = secrets.token_urlsafe(_MIN_SECRET_CHARS)
        logger.warning(
            "SERVICE_JWT_SECRET 缺失或过短（%s 环境），已生成一次性密钥；进程重启后登录态失效。",
            environment,
        )
    redis_url = _env("SERVICE_REDIS_URL", "redis://127.0.0.1:6379/0")
    material_backend = _env("SERVICE_MATERIAL_STORE_BACKEND", "memory").strip().lower()
    if material_backend not in {"memory", "redis"}:
        raise ValueError("SERVICE_MATERIAL_STORE_BACKEND 只能是 memory 或 redis。")
    return ServiceConfig(
        database_url=database_url,
        jwt_secret=jwt_secret,
        token_ttl_hours=max(1, _int_env("SERVICE_TOKEN_TTL_HOURS", 12)),
        max_concurrent_runs_per_user=max(1, _int_env("SERVICE_MAX_CONCURRENT_RUNS_PER_USER", 2)),
        max_global_queued_runs=max(1, _int_env("SERVICE_MAX_GLOBAL_QUEUED_RUNS", 100)),
        worker_max_running_runs=max(1, _int_env("SERVICE_WORKER_MAX_RUNNING_RUNS", 3)),
        worker_lease_seconds=max(10, _int_env("SERVICE_WORKER_LEASE_SECONDS", 60)),
        worker_heartbeat_seconds=max(1, _int_env("SERVICE_WORKER_HEARTBEAT_SECONDS", 20)),
        worker_poll_seconds=max(0.05, _float_env("SERVICE_WORKER_POLL_SECONDS", 15.0)),
        redis_preview_enabled=_bool_env("SERVICE_REDIS_PREVIEW_ENABLED", False),
        redis_url=redis_url,
        redis_channel_prefix=_env("SERVICE_REDIS_CHANNEL_PREFIX", "deepresearcher"),
        redis_preview_queue_size=max(1, _int_env("SERVICE_REDIS_PREVIEW_QUEUE_SIZE", 128)),
        material_store_backend=material_backend,
        material_redis_url=_env("SERVICE_MATERIAL_REDIS_URL", redis_url),
        material_key_prefix=(
            _env("SERVICE_MATERIAL_KEY_PREFIX", "deepresearcher:material")
            or "deepresearcher:material"
        ),
        search_material_ttl_seconds=max(1, _int_env("SERVICE_SEARCH_MATERIAL_TTL_SECONDS", 21_600)),
        document_material_ttl_seconds=max(
            1, _int_env("SERVICE_DOCUMENT_MATERIAL_TTL_SECONDS", 86_400)
        ),
        login_account_attempts=max(1, _int_env("SERVICE_LOGIN_ACCOUNT_ATTEMPTS", 5)),
        login_ip_attempts=max(1, _int_env("SERVICE_LOGIN_IP_ATTEMPTS", 20)),
        login_rate_window_seconds=max(1, _int_env("SERVICE_LOGIN_RATE_WINDOW_SECONDS", 300)),
        login_block_seconds=max(1, _int_env("SERVICE_LOGIN_BLOCK_SECONDS", 900)),
        forwarded_allow_ips=_env("SERVICE_FORWARDED_ALLOW_IPS", "127.0.0.1"),
        host=_env("SERVICE_HOST", "127.0.0.1"),
        port=_int_env("SERVICE_PORT", 8080),
        service_log_dir=project_path_env("SERVICE_LOG_DIR", _PROJECT_ROOT / "var" / "service"),
        jsonl_events=_bool_env("SERVICE_JSONL_EVENTS", True),
        jsonl_retention_days=max(0, _int_env("SERVICE_JSONL_RETENTION_DAYS", 14)),
        run_trace_retention_days=max(0, _int_env("SERVICE_RUN_TRACE_RETENTION_DAYS", 14)),
        retention_sweep_interval_seconds=max(
            60, _int_env("SERVICE_RETENTION_SWEEP_INTERVAL_SECONDS", 21_600)
        ),
        langfuse_public_key=_env("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=_env("LANGFUSE_SECRET_KEY"),
        langfuse_base_url=_env("LANGFUSE_BASE_URL"),
        otel_exporter=_env("SERVICE_OTEL_EXPORTER", "none").strip().lower(),
    )


def clear_service_config_cache() -> None:
    """仅供测试或显式热加载配置使用。"""
    get_service_config.cache_clear()


def checkpoint_dsn(database_url: str) -> str | None:
    """由业务库 URL 派生 checkpointer 的 psycopg DSN。

    AsyncPostgresSaver 走 psycopg，不认 SQLAlchemy 的 +asyncpg 方言，需改写协议前缀。
    非 postgresql URL（测试用 SQLite）返回 None，装配方据此跳过 checkpointer。
    """
    if not database_url.startswith("postgresql"):
        return None
    return database_url.replace("postgresql+asyncpg://", "postgresql://", 1)
