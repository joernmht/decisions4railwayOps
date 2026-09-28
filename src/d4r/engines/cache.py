"""Append-only JSONL response cache for paid API calls.

Every live call is stored under a key derived from (engine id, model, canonical request payload).
Re-running an experiment replays the cached answers, so results are reproducible even though the
services are not bit-deterministic and model aliases move. The cache stores requests and answers,
never credentials.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

__all__ = ["ResponseCache", "request_key"]


def request_key(engine_id: str, model: str, payload: Any) -> str:
    """Stable key for a request."""
    blob = json.dumps(
        {"e": engine_id, "m": model, "p": payload}, sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(blob.encode()).hexdigest()


class ResponseCache:
    """A dict-like cache backed by one JSONL file (one record per line, last write wins)."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self._mem: dict[str, dict[str, Any]] = {}
        if self.path and self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        self._mem[rec["key"]] = rec["value"]

    def get(self, key: str) -> dict[str, Any] | None:
        return self._mem.get(key)

    def put(self, key: str, value: dict[str, Any]) -> None:
        self._mem[key] = value
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self._mem)
