import logging
import logging.config
from pathlib import Path

from deepsearch_agent.observability.tracing.context import current_context


class CompactFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        node = getattr(record, "node", None)
        context = current_context()
        correlation = []
        for label, value in (
            ("run", getattr(record, "run_id", None) or (context.run_id if context else None)),
            (
                "session",
                getattr(record, "session_id", None) or (context.session_id if context else None),
            ),
            ("node", node or (context.node_id if context else None)),
        ):
            if value:
                correlation.append(f"{label}={value}")
        prefix = f"[{', '.join(correlation)}] " if correlation else ""
        return prefix + base


def configure_logging(level: str = "INFO", *, log_path: Path | None = None) -> None:
    """配置进程级 Logger；库被导入时不产生全局副作用。"""
    handlers: dict[str, dict[str, object]] = {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "compact",
            "stream": "ext://sys.stderr",
        },
    }
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "compact",
            "filename": str(log_path),
            "maxBytes": 10_000_000,
            "backupCount": 5,
            "encoding": "utf-8",
        }
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "compact": {
                    "()": CompactFormatter,
                    "format": "%(asctime)s %(levelname)s %(name)s: %(message)s",
                    "datefmt": "%H:%M:%S",
                }
            },
            "handlers": handlers,
            "root": {"level": level.upper(), "handlers": list(handlers)},
            "loggers": {
                "openai": {"level": "WARNING", "handlers": list(handlers), "propagate": False},
                "httpcore": {"level": "WARNING", "handlers": list(handlers), "propagate": False},
            },
        }
    )
