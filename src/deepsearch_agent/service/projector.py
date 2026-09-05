"""Compatibility import for SSE projection; prefer ``service.events.projector``."""

from deepsearch_agent.service.events.projector import SseFrame, project

__all__ = ["SseFrame", "project"]
