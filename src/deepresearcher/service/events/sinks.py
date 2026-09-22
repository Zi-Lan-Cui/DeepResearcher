"""通用扇出 sink：把同一条记录分给多个接收者。"""

from deepresearcher.observability.logging_config import get_logger

logger = get_logger("deepresearcher.service.events")


class CompositeSink:
    """把同一条记录扇出到多个 sink；任何一路失败都不拖垮其它路。"""

    def __init__(self, *sinks):
        self._sinks = sinks

    def write(self, record) -> None:
        for sink in self._sinks:
            try:
                sink.write(record)
            except Exception:  # noqa: BLE001 - sink 之间必须互相隔离
                logger.warning(
                    "composite_sink_member_failed sink=%s", type(sink).__name__, exc_info=True
                )
