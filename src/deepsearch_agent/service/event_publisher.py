"""Compatibility import for the publisher; prefer ``service.events.publisher``."""

from deepsearch_agent.service.events.publisher import RunEventPublisher

__all__ = ["RunEventPublisher"]
