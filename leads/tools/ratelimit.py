"""Thread-safe rate limiting.

Each upstream gets its own limiter so a slow Overpass query never starves the
LLM queue. Token-bucket + minimum interval between requests.
"""

from __future__ import annotations

import threading
import time


class RateLimiter:
    """Simple thread-safe token bucket with a minimum spacing between calls."""

    def __init__(self, rate_per_minute: float, burst: int = 1) -> None:
        self.rate = max(0.1, rate_per_minute) / 60.0  # tokens per second
        self.capacity = max(1, burst)
        self._tokens = float(self.capacity)
        self._last = time.monotonic()
        self._lock = threading.Lock()
        self._next_allowed = 0.0
        self.min_interval = 0.0

    def acquire(self, tokens: float = 1.0, block: bool = True) -> bool:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._last = now
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)

            wait = 0.0
            if self._tokens < tokens:
                wait = (tokens - self._tokens) / self.rate
            if now < self._next_allowed:
                wait = max(wait, self._next_allowed - now)

            if wait > 0 and not block:
                return False

            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
                self._last = now
                self._tokens = min(self.capacity, self._tokens + wait * self.rate)

            self._tokens -= tokens
            if self.min_interval:
                self._next_allowed = now + self.min_interval
            return True

    def penalize(self, seconds: float) -> None:
        """Back off globally after a 429."""
        with self._lock:
            self._next_allowed = max(self._next_allowed, time.monotonic() + seconds)
