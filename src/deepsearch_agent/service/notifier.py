"""Compatibility import for the notifier; prefer ``service.events.notifier``."""

from deepsearch_agent.service.events.notifier import EventNotifier

__all__ = ["EventNotifier"]
