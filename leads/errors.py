"""Typed exception hierarchy.

Every failure mode in this project is expected, classified, and either retried
or degraded deterministically. Nothing is allowed to raise a bare ``Exception``
into the orchestrator.
"""

from __future__ import annotations


class LeadsError(Exception):
    """Base class for every error this project raises on purpose."""


class ConfigError(LeadsError):
    """Configuration is missing or invalid. Raised before any network call."""


class BudgetExceeded(LeadsError):
    """A hard spending/usage cap was hit. The run must stop cleanly."""


class RateLimited(LeadsError):
    """Upstream asked us to slow down."""

    def __init__(self, provider: str, retry_after: float = 5.0) -> None:
        super().__init__(f"{provider} rate limited (retry after {retry_after:.1f}s)")
        self.provider = provider
        self.retry_after = retry_after


class ProviderError(LeadsError):
    """An upstream tool returned an error we cannot recover from locally."""

    def __init__(self, provider: str, detail: str, status: int | None = None) -> None:
        super().__init__(f"{provider}: {detail}")
        self.provider = provider
        self.detail = detail
        self.status = status


class LLMError(LeadsError):
    """The model call failed, timed out, or returned unusable output."""


class LLMUnavailable(LLMError):
    """Circuit breaker is open: too many consecutive model failures.

    Agents that can degrade deterministically catch this and continue.
    """


class JSONParseError(LLMError):
    """Model output could not be recovered into JSON, even after repair."""


class ToolUnavailable(LeadsError):
    """An optional data source is unavailable; the pipeline should fall back."""

    def __init__(self, tool: str, detail: str) -> None:
        super().__init__(f"{tool} unavailable: {detail}")
        self.tool = tool
        self.detail = detail
