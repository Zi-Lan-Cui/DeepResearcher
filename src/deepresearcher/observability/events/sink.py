import json
from pathlib import Path
from threading import Lock
from typing import Any, Mapping

from pydantic import BaseModel


class JsonlSink:
    """追加写入机器可读记录；一行一个 JSON 对象。"""

    def __init__(self, path: Path):
        self.path = path
        self._lock = Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: Mapping[str, Any] | Any) -> None:
        if isinstance(record, BaseModel):
            record = record.model_dump(exclude_none=True)
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
