"""Append-only JSONL event logging."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any


class JsonlEventLogger:
    """Thread-safe append-only event writer."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def log(self, event_type: str, *, sim_time_s: float, details: dict[str, Any]) -> None:
        """Persist one event with stable field names."""

        record = {"event_type": event_type, "sim_time_s": sim_time_s, "details": details}
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
