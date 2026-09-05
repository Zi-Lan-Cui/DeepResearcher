"""Compatibility imports; prefer ``service.runs.service``."""

from deepsearch_agent.service.runs.service import QuotaExceededError, RunService

__all__ = ["QuotaExceededError", "RunService"]
