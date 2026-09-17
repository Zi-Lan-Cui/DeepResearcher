"""回归守卫：Alembic 迁移不得抹掉应用的 root 日志 handler（agent.log 停写的根因）。"""

import logging
from logging.handlers import RotatingFileHandler

import pytest

from deepresearcher.observability.logging_config import configure_logging
from deepresearcher.service.persistence.database import migrate_database

pytestmark = pytest.mark.asyncio


async def test_migrate_keeps_application_file_handler(tmp_path, caplog):
    log_file = tmp_path / "agent.log"
    configure_logging("INFO", log_path=log_file)
    root = logging.getLogger()
    assert any(isinstance(h, RotatingFileHandler) for h in root.handlers), (
        "configure_logging 应先挂上文件 handler"
    )

    # 跑一次真实 Alembic 升级（临时 SQLite 库）——旧实现里 env.py 的 fileConfig
    # 会按 alembic.ini 用 console handler 覆盖 root.handlers，把文件 handler 冲掉。
    url = f"sqlite+aiosqlite:///{tmp_path / 'db.sqlite'}"
    await migrate_database(url)

    assert any(isinstance(h, RotatingFileHandler) for h in root.handlers), (
        "migrate 后文件 handler 不应被 Alembic 的 fileConfig 抹掉（agent.log 停写回归）"
    )
    logging.getLogger("deepresearcher.test").warning("sentinel-line-for-agent-log")
    for h in root.handlers:
        h.flush()
    assert "sentinel-line-for-agent-log" in log_file.read_text("utf-8")
