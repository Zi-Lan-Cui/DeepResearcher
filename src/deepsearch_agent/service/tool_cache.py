"""Compatibility import; prefer ``service.persistence.tool_cache``."""

from deepsearch_agent.service.persistence.tool_cache import PostgresToolCache

__all__ = ["PostgresToolCache"]
