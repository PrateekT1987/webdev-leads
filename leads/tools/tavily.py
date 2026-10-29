"""Tavily client: search + extract.

Tavily is a retrieval API built for agents. It returns ranked results with page
content, which is exactly what we need to (a) find directory pages that list
local businesses and (b) check whether a business already owns a website.

It is *not* a Google Maps replacement: no place IDs, no review counts, no
local-pack ordering. See docs/DATA_SOURCES.md.

Credit costs (free tier = 1,000/month):
  basic search    = 1 credit
  advanced search = 2 credits
  basic extract   = 1 credit per 5 URLs
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from ..errors import BudgetExceeded, ProviderError, RateLimited
from .cache import HttpCache
from .ratelimit import RateLimiter

log = logging.getLogger(__name__)

SEARCH_URL = "https://api.tavily.com/search"
EXTRACT_URL = "https://api.tavily.com/extract"


class TavilyClient:
    def __init__(
        self,
        api_key: str,
        *,
        cache: HttpCache | None = None,
        timeout: float = 45.0,
        user_agent: str = "webdev-leads/1.0",
        credit_budget: float = 900,
    ) -> None:
        self.api_key = api_key
        self.cache = cache
        self.timeout = timeout
        self.user_agent = user_agent
        self.credit_budget = credit_budget
        self.credits_used = 0.0
        self.limiter = RateLimiter(rate_per_minute=120, burst=6)
        self._session = requests.Session()
        self._session.headers.update(
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        )

    # ------------------------------------------------------------------ utils
    def _charge(self, credits: float) -> None:
        if self.credits_used + credits > self.credit_budget:
            raise BudgetExceeded(
                f"Tavily credit budget reached ({self.credits_used:.0f}/{self.credit_budget:.0f})"
            )
        self.credits_used += credits

    def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.limiter.acquire()
        for attempt in range(3):
            try:
                resp = self._session.post(url, json=payload, timeout=self.timeout)
            except requests.Timeout as exc:
                if attempt == 2:
                    raise ProviderError("tavily", f"timeout: {exc}") from exc
                time.sleep(1.5 * (attempt + 1))
                continue
            except requests.RequestException as exc:
                if attempt == 2:
                    raise ProviderError("tavily", str(exc)) from exc
                time.sleep(1.5 * (attempt + 1))
                continue

            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", 10 + attempt * 5))
                self.limiter.penalize(retry_after)
                if attempt == 2:
                    raise RateLimited("tavily", retry_after)
                time.sleep(min(retry_after, 30))
                continue

            if resp.status_code in (401, 403):
                raise ProviderError(
                    "tavily", "unauthorized — check TAVILY_API_KEY", resp.status_code
                )
            if resp.status_code == 432:  # Tavily: plan limit exceeded
                raise BudgetExceeded("Tavily plan limit exceeded (HTTP 432)")
            if resp.status_code >= 500:
                if attempt == 2:
                    raise ProviderError("tavily", resp.text[:300], resp.status_code)
                time.sleep(2.0 * (attempt + 1))
                continue
            if resp.status_code >= 400:
                raise ProviderError("tavily", resp.text[:300], resp.status_code)

            try:
                return resp.json()
            except ValueError as exc:
                raise ProviderError("tavily", "non-JSON response") from exc

        raise ProviderError("tavily", "exhausted retries")

    # ----------------------------------------------------------------- search
    def search(
        self,
        query: str,
        *,
        depth: str = "basic",
        max_results: int = 8,
        include_raw_content: bool = False,
        exclude_domains: list[str] | None = None,
        include_domains: list[str] | None = None,
        topic: str = "general",
        days: int | None = None,
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "query": query,
            "search_depth": depth,
            "max_results": max(1, min(int(max_results), 20)),
            "topic": topic,
            "include_raw_content": include_raw_content,
            "include_answer": False,
            "include_images": False,
        }
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains[:50]
        if include_domains:
            payload["include_domains"] = include_domains[:20]
        if days:
            payload["days"] = days

        if self.cache is not None:
            cached = self.cache.get("tavily.search", payload)
            if cached is not None:
                return list(cached.get("results", []))

        self._charge(2.0 if depth == "advanced" else 1.0)
        data = self._post(SEARCH_URL, payload)

        results = []
        for r in data.get("results", []) or []:
            results.append(
                {
                    "url": r.get("url", ""),
                    "title": r.get("title", ""),
                    "content": r.get("content", "") or "",
                    "raw_content": r.get("raw_content", "") or "",
                    "score": r.get("score", 0.0),
                }
            )

        if self.cache is not None:
            self.cache.put("tavily.search", payload, {"results": results})
        return results

    # ---------------------------------------------------------------- extract
    def extract(
        self, urls: list[str], *, depth: str = "basic", max_chars: int = 20000
    ) -> dict[str, str]:
        """Return {url: markdown content}. Failed URLs are simply omitted."""
        urls = [u for u in urls if u][:20]
        if not urls:
            return {}

        payload = {"urls": urls, "extract_depth": depth, "include_images": False}
        if self.cache is not None:
            cached = self.cache.get("tavily.extract", payload)
            if cached is not None:
                return dict(cached.get("pages", {}))

        self._charge(max(1.0, len(urls) / 5.0) * (2.0 if depth == "advanced" else 1.0))
        data = self._post(EXTRACT_URL, payload)

        pages: dict[str, str] = {}
        for item in data.get("results", []) or []:
            url = item.get("url", "")
            content = item.get("raw_content") or item.get("content") or ""
            if url and content:
                pages[url] = content[:max_chars]
        for item in data.get("failed_results", []) or []:
            log.debug("tavily extract failed: %s", item)

        if self.cache is not None:
            self.cache.put("tavily.extract", payload, {"pages": pages})
        return pages

    # ------------------------------------------------------------------ stats
    @property
    def stats(self) -> dict[str, float]:
        return {
            "credits": round(self.credits_used, 2),
            "budget": self.credit_budget,
            "cache": self.cache.stats if self.cache else {},
        }
