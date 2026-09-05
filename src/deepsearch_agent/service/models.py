"""Compatibility imports; prefer ``service.persistence.models``."""

from deepsearch_agent.service.persistence.models import (
    Base,
    Run,
    RunEvent,
    RunUsage,
    ToolCacheEntry,
    User,
)

__all__ = ["Base", "Run", "RunEvent", "RunUsage", "ToolCacheEntry", "User"]
