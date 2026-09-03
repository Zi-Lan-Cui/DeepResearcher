"""服务层配置：数据库、JWT、监听与配额。

引擎配置（LLM/搜索/agent 预算）仍由 ``deepsearch_agent.config.get_settings()`` 负责；
本模块只补“把它跑成多用户服务”所需的新键。沿用同一 env 文件约定
（``DEEPSEARCH_ENV_FILE``，默认 ``env/.env``；``load_dotenv(override=False)``
不会与引擎的加载互相覆盖），环境判别复用 ``APP_ENV``。
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from functools import lru_cache
from os import getenv
from pathlib import Path

from dotenv import load_dotenv

from deepsearch_agent.observability.logger import get_logger

# service/settings.py 位于 src/deepsearch_agent/service/ 下，比 config.py 深一层。
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

logger = get_logger("deepsearch_agent.service.settings")

# 短于此长度的 secret 会被视为未正确配置：HS256 暴力成本与密钥熵直接挂钩。
_MIN_SECRET_CHARS = 32


def _env(name: str, default: str = "") -> str:
    return getenv(name, default).strip()


def _int_env(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = _env(name, "true" if default else "false").lower()
    if raw in {"1", "true", "yes"}:
        return True
    if raw in {"0", "false", "no"}:
        return False
    return default


@dataclass(frozen=True)
class ServiceConfig:
    database_url: str
    jwt_secret: str
    token_ttl_hours: int = 12
    max_concurrent_runs_per_user: int = 2
    max_global_running_runs: int = 3
    max_global_queued_runs: int = 100
    worker_lease_seconds: int = 60
    worker_heartbeat_seconds: int = 20
    worker_poll_seconds: float = 1.0
    api_embedded_worker: bool = False
    host: str = "127.0.0.1"
    port: int = 8080
    service_log_dir: Path = _PROJECT_ROOT / "var" / "service"
    jsonl_events: bool = True


@lru_cache(maxsize=1)
def get_service_config() -> ServiceConfig:
    """加载并缓存服务配置；测试请显式构造 :class:`ServiceConfig` 或先清缓存。"""
    env_file = Path(_env("DEEPSEARCH_ENV_FILE", str(_PROJECT_ROOT / "env" / ".env")))
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
        # 开发期缺省 → 每次进程随机一份：能用，但重启即令全部旧 token 失效。
        jwt_secret = secrets.token_urlsafe(_MIN_SECRET_CHARS)
        logger.warning(
            "SERVICE_JWT_SECRET 缺失或过短（%s 环境），已生成一次性密钥；进程重启后登录态失效。",
            environment,
        )
    return ServiceConfig(
        database_url=database_url,
        jwt_secret=jwt_secret,
        token_ttl_hours=max(1, _int_env("SERVICE_TOKEN_TTL_HOURS", 12)),
        max_concurrent_runs_per_user=max(1, _int_env("SERVICE_MAX_CONCURRENT_RUNS_PER_USER", 2)),
        max_global_running_runs=max(1, _int_env("SERVICE_MAX_GLOBAL_RUNNING_RUNS", 3)),
        max_global_queued_runs=max(1, _int_env("SERVICE_MAX_GLOBAL_QUEUED_RUNS", 100)),
        worker_lease_seconds=max(10, _int_env("SERVICE_WORKER_LEASE_SECONDS", 60)),
        worker_heartbeat_seconds=max(1, _int_env("SERVICE_WORKER_HEARTBEAT_SECONDS", 20)),
        worker_poll_seconds=max(0.05, _float_env("SERVICE_WORKER_POLL_SECONDS", 1.0)),
        api_embedded_worker=_bool_env("SERVICE_API_EMBEDDED_WORKER", False),
        host=_env("SERVICE_HOST", "127.0.0.1"),
        port=_int_env("SERVICE_PORT", 8080),
        service_log_dir=Path(_env("SERVICE_LOG_DIR", str(_PROJECT_ROOT / "var" / "service"))),
        jsonl_events=_bool_env("SERVICE_JSONL_EVENTS", True),
    )


def clear_service_config_cache() -> None:
    """仅供测试或显式热加载配置使用。"""
    get_service_config.cache_clear()


def checkpoint_dsn(database_url: str) -> str | None:
    """由业务库 URL 派生 checkpointer 的 psycopg DSN。

    langgraph 的 AsyncPostgresSaver 走 psycopg（不认 SQLAlchemy 的 +asyncpg 方言），
    两者共享同一个 PG 实例但连接层独立；非 postgresql URL（测试用 SQLite）
    返回 None，服务自动跳过 checkpointer——恢复能力随部署形态降级，不假装有。
    """
    if not database_url.startswith("postgresql"):
        return None
    return database_url.replace("postgresql+asyncpg://", "postgresql://", 1)
