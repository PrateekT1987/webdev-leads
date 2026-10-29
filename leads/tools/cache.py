"""SQLite-backed response cache.

Tavily search and Overpass are the two things a run repeats most (a resumed run
re-verifies the same businesses). Caching them by request signature makes runs
cheap, replayable and deterministic during development.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ..store import Store


class HttpCache:
    def __init__(self, store: Store, ttl_hours: int = 168) -> None:
        self.store = store
        self.ttl_hours = ttl_hours
        self.hits = 0
        self.misses = 0
        self._run_scope: str | None = None
        self._memory: dict[str, Any] = {}

    def set_scope(self, run_id: str | None) -> None:
        """Namespace cache keys per run when a fresh run is requested."""
        self._run_scope = run_id

    @staticmethod
    def make_key(provider: str, payload: dict[str, Any]) -> str:
        blob = json.dumps(payload, sort_keys=True, default=str)
        digest = hashlib.sha256(blob.encode()).hexdigest()[:32]
        return f"{provider}:{digest}"

    def get(self, provider: str, payload: dict[str, Any]) -> Any | None:
        key = self.make_key(provider, payload)
        if key in self._memory:
            self.hits += 1
            return self._memory[key]
        cached = self.store.cache_get(key)
        if cached is not None:
            self.hits += 1
            self._memory[key] = cached
            return cached
        self.misses += 1
        return None

    def put(self, provider: str, payload: dict[str, Any], value: Any) -> None:
        key = self.make_key(provider, payload)
        self._memory[key] = value
        self.store.cache_put(key, provider, json.dumps(payload, default=str)[:500], value, self.ttl_hours)

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}
