"""OpenRouter chat client with budget, retries, telemetry and a circuit breaker.

Deliberate design choices for reliability on a free reasoning model:

* **Small outputs only.** Every prompt asks for one decision or a handful of
  records, never a large document. Long generations get truncated mid-JSON.
* **JSON repair.** Truncated output is salvaged rather than discarded.
* **Circuit breaker.** After N consecutive failures the breaker opens and
  agents fall back to deterministic rules instead of burning the run.
* **Budget.** Hard caps on calls and tokens, checked before every request.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from ..errors import BudgetExceeded, JSONParseError, LLMError, LLMUnavailable, RateLimited
from .json_repair import parse_json_object

log = logging.getLogger(__name__)


@dataclass
class LLMBudget:
    max_calls: int = 500
    max_tokens: int = 3_000_000
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def check(self) -> None:
        if self.max_calls and self.calls >= self.max_calls:
            raise BudgetExceeded(f"LLM call budget reached ({self.calls}/{self.max_calls})")
        if self.max_tokens and (self.prompt_tokens + self.completion_tokens) >= self.max_tokens:
            raise BudgetExceeded(
                f"LLM token budget reached ({self.prompt_tokens + self.completion_tokens})"
            )

    def record(self, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self.calls += 1
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient:
    """Thin, hardened wrapper over the OpenRouter chat-completions API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 180.0,
        max_tokens: int = 1500,
        budget: LLMBudget | None = None,
        telemetry=None,
        failure_threshold: int = 6,
        enabled: bool = True,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.budget = budget or LLMBudget()
        self.telemetry = telemetry
        self.failure_threshold = failure_threshold
        self.enabled = enabled and bool(api_key and model)

        self._consecutive_failures = 0
        self._lock = threading.Lock()
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://localhost/webdev-leads",
                "X-Title": "webdev-leads",
            }
        )

    # ------------------------------------------------------------- breaker
    @property
    def breaker_open(self) -> bool:
        return self._consecutive_failures >= self.failure_threshold

    def _record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures == self.failure_threshold:
                log.warning("LLM circuit breaker opened after %d failures", self._consecutive_failures)

    def _record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0

    def reset_breaker(self) -> None:
        with self._lock:
            self._consecutive_failures = 0

    # ---------------------------------------------------------------- core
    def _chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        json_mode: bool,
        agent: str,
    ) -> tuple[str, dict[str, Any]]:
        if not self.enabled:
            raise LLMUnavailable("LLM disabled (no API key/model configured)")
        if self.breaker_open:
            raise LLMUnavailable(
                f"circuit breaker open after {self._consecutive_failures} consecutive failures"
            )
        self.budget.check()

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        started = time.monotonic()
        last_error = ""

        for attempt in range(3):
            try:
                resp = self._session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=(10.0, self.timeout),
                )
            except requests.RequestException as exc:
                last_error = f"transport: {exc}"
                time.sleep(1.5 * (attempt + 1) + random.random())
                continue

            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", 8 + attempt * 6))
                last_error = f"429 rate limited (retry after {retry_after:.0f}s)"
                time.sleep(min(retry_after, 45))
                continue
            if resp.status_code == 402:
                self._record_failure()
                raise BudgetExceeded("OpenRouter credits exhausted (HTTP 402)")
            if resp.status_code in (401, 403):
                self._record_failure()
                raise LLMError(f"OpenRouter auth failed (HTTP {resp.status_code})")
            if resp.status_code >= 500:
                last_error = f"HTTP {resp.status_code}: {resp.text[:160]}"
                time.sleep(2.0 * (attempt + 1) + random.random())
                continue
            if resp.status_code >= 400:
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                break

            try:
                data = resp.json()
            except ValueError:
                last_error = "non-JSON API response"
                continue

            if "choices" not in data or not data["choices"]:
                last_error = f"malformed completion: {str(data)[:200]}"
                continue

            choice = data["choices"][0]
            content = (choice.get("message") or {}).get("content") or ""
            usage = data.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)

            self.budget.record(prompt_tokens, completion_tokens)
            if self.telemetry is not None:
                self.telemetry(
                    agent=agent,
                    model=self.model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=int((time.monotonic() - started) * 1000),
                    ok=True,
                )

            if choice.get("finish_reason") == "length":
                log.debug("[%s] response hit token ceiling; salvaging JSON", agent)

            self._record_success()
            return content, {
                "finish_reason": choice.get("finish_reason"),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }

        # All attempts failed.
        self._record_failure()
        if self.telemetry is not None:
            self.telemetry(
                agent=agent,
                model=self.model,
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=int((time.monotonic() - started) * 1000),
                ok=False,
                error=last_error,
            )
        if "429" in last_error:
            raise RateLimited("openrouter", 20.0)
        raise LLMError(last_error or "unknown LLM failure")

    # ------------------------------------------------------------- public
    def chat_json(
        self,
        *,
        system: str,
        user: str,
        agent: str = "unknown",
        max_tokens: int | None = None,
        temperature: float = 0.0,
        attempts: int = 2,
    ) -> dict[str, Any]:
        """Ask for one small JSON object. Retries once with a brevity nudge."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        budget = max_tokens or self.max_tokens
        last_error: Exception | None = None

        for attempt in range(attempts):
            content, meta = self._chat(
                messages,
                max_tokens=budget,
                temperature=temperature,
                json_mode=True,
                agent=agent,
            )
            try:
                return parse_json_object(content)
            except JSONParseError as exc:
                last_error = exc
                log.debug("[%s] unparseable JSON (attempt %d): %s", agent, attempt + 1, exc)
                if meta.get("finish_reason") == "length":
                    # Cut off: ask for a shorter answer.
                    budget = min(self.max_tokens, int(budget * 1.4))
                    messages = [
                        {"role": "system", "content": system},
                        {
                            "role": "user",
                            "content": user
                            + "\n\nIMPORTANT: your previous answer was cut off. "
                            "Reply with the complete JSON object only, fewer words, "
                            "and no explanation.",
                        },
                    ]
                else:
                    messages.append({"role": "assistant", "content": content[:1500]})
                    messages.append(
                        {
                            "role": "user",
                            "content": "That was not valid JSON. Return ONLY the corrected "
                            "JSON object, nothing else.",
                        }
                    )

        raise JSONParseError(str(last_error)) if last_error else LLMError("no JSON produced")

    def chat_text(
        self, *, system: str, user: str, agent: str = "unknown", max_tokens: int | None = None
    ) -> str:
        content, _ = self._chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens or self.max_tokens,
            temperature=0.2,
            json_mode=False,
            agent=agent,
        )
        return content.strip()

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "calls": self.budget.calls,
            "prompt_tokens": self.budget.prompt_tokens,
            "completion_tokens": self.budget.completion_tokens,
            "total_tokens": self.budget.total_tokens,
            "consecutive_failures": self._consecutive_failures,
            "breaker_open": self.breaker_open,
        }
