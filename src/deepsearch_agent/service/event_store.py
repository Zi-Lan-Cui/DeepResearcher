"""Compatibility import for the event store; prefer ``service.events.store``."""

from deepsearch_agent.service.events.store import RunEventStore

__all__ = ["RunEventStore"]
