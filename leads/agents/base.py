"""Agent base class: shared context, telemetry and graceful degradation.

Every agent has the same shape: a narrow ``run``/``handle`` method that either
returns a typed result or raises. The orchestrator owns retries, concurrency and
checkpointing; agents stay small and independently testable.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..config import Settings
from ..errors import JSONParseError, LLMError, LLMUnavailable
from ..logging_setup import AgentLogger, run_id_var
from ..models import utcnow
from ..store import Store
from ..tools.overpass import OverpassClient
from ..tools.tavily import TavilyClient
from ..tools.webprobe import WebProbe
from ..llm.client import LLMClient


@dataclass
class AgentContext:
    """Everything an agent is allowed to touch."""

    run_id: str
    settings: Settings
    store: Store
    llm: LLMClient
    tavily: TavilyClient | None = None
    overpass: OverpassClient | None = None
    probe: WebProbe | None = None
    cache: Any = None
    counters: dict[str, int] = field(default_factory=dict)
    _counter_lock: Any = field(default_factory=threading.Lock, repr=False)

    def bump(self, key: str, amount: int = 1) -> None:
        with self._counter_lock:
            self.counters[key] = self.counters.get(key, 0) + amount

    def log_event(self, level: str, agent: str, message: str, data: dict | None = None) -> None:
        self.store.log_event(self.run_id, level, agent, message, data)


class Agent(Protocol):
    name: str


class BaseAgent:
    """Common behaviour: named logger, durable events, LLM-or-fallback helper."""

    name = "agent"

    def __init__(self, ctx: AgentContext) -> None:
        self.ctx = ctx
        self.log = AgentLogger(self.name)
        self._run_token = run_id_var.set(ctx.run_id)

    def close(self) -> None:
        try:
            run_id_var.reset(self._run_token)
        except ValueError:
            pass

    def event(self, level: str, message: str, data: dict[str, Any] | None = None, **extra: Any) -> None:
        payload = {**(data or {}), **extra}
        self.ctx.log_event(level, self.name, message, payload or None)

    def ask_llm_json(self, *, system: str, user: str, max_tokens: int | None = None) -> dict[str, Any]:
        """Call the model, converting failures into a single typed exception."""
        return self.ctx.llm.chat_json(
            system=system, user=user, agent=self.name, max_tokens=max_tokens
        )

    @property
    def llm_available(self) -> bool:
        return self.ctx.llm.enabled and not self.ctx.llm.breaker_open


__all__ = [
    "AgentContext",
    "Agent",
    "BaseAgent",
    "JSONParseError",
    "LLMError",
    "LLMUnavailable",
    "utcnow",
]
