"""Compatibility imports; prefer ``service.runs.queue``."""

from deepsearch_agent.service.runs.queue import PostgresRunQueue, RunWork

__all__ = ["PostgresRunQueue", "RunWork"]
