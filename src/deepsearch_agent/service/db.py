"""Compatibility imports; prefer ``service.persistence.database``."""

from deepsearch_agent.service.persistence.database import (
    init_db,
    make_engine,
    make_session_factory,
    migrate_database,
)

__all__ = ["init_db", "make_engine", "make_session_factory", "migrate_database"]
