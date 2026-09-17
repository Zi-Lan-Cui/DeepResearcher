from __future__ import annotations

import asyncio

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from deepresearcher.service.persistence.models import Base

config = context.config
# 故意不调用 fileConfig：Alembic 跑在 API/Worker 进程内、在 configure_logging 之后。
# fileConfig(alembic.ini) 会按 [logger_root] handlers=console 覆盖 root 的 handlers，
# 把应用刚装好的 RotatingFileHandler 抹掉（disable_existing_loggers=False 只保活子 logger、
# 挡不住覆盖 root.handlers）→ agent.log 停写。这里让 Alembic 直接沿用进程已有日志配置，
# 迁移日志反而一并进入 agent.log。
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def _run_sync_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(_run_sync_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_async_migrations())
