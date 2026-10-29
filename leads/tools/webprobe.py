"""Direct web probing: the cheapest way to answer "do they have a website?".

Strategy for a business with no known URL:
  1. guess likely domains from the name and probe them (zero API credits);
  2. a probe only counts when the page really mentions the business;
  3. parked/for-sale/placeholder pages are rejected.

This runs before any paid search, which keeps Tavily credits for the genuinely
ambiguous cases.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from urllib.parse import urlparse

import requests

from .extract import (
    guess_domains,
    is_blocklisted,
    looks_parked,
    page_mentions_business,
    registrable_domain,
)

log = logging.getLogger(__name__)

MAX_BYTES = 220_000


@dataclass
class ProbeResult:
    url: str
    ok: bool
    status: int | None = None
    final_url: str | None = None
    title: str = ""
    text: str = ""
    parked: bool = False
    error: str = ""

    @property
    def title_matches(self) -> bool:
        return bool(self.title)


class WebProbe:
    def __init__(self, *, timeout: float = 12.0, user_agent: str = "webdev-leads/1.0", max_workers: int = 8) -> None:
        self.timeout = timeout
        self.user_agent = user_agent
        self.request_count = 0
        self._lock = threading.Lock()
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )

    # ------------------------------------------------------------------ probe
    def probe(self, url: str) -> ProbeResult:
        if not url:
            return ProbeResult(url=url, ok=False, error="empty url")
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        with self._lock:
            self.request_count += 1

        try:
            resp = self._session.get(
                url, timeout=self.timeout, allow_redirects=True, stream=True
            )
        except requests.RequestException as exc:
            return ProbeResult(url=url, ok=False, error=str(exc)[:200])

        status = resp.status_code
        final_url = resp.url
        ctype = resp.headers.get("Content-Type", "").lower()

        if status >= 400:
            resp.close()
            return ProbeResult(url=url, ok=False, status=status, final_url=final_url)

        if "html" not in ctype and "text" not in ctype:
            resp.close()
            return ProbeResult(
                url=url, ok=False, status=status, final_url=final_url,
                error=f"non-html content-type: {ctype[:60]}",
            )

        chunks: list[bytes] = []
        size = 0
        try:
            for chunk in resp.iter_content(16384):
                chunks.append(chunk)
                size += len(chunk)
                if size >= MAX_BYTES:
                    break
        finally:
            resp.close()

        raw = b"".join(chunks).decode(resp.encoding or "utf-8", errors="ignore")
        title = self._extract_title(raw)
        text = self._html_to_text(raw)
        return ProbeResult(
            url=url,
            ok=True,
            status=status,
            final_url=final_url,
            title=title,
            text=text,
            parked=looks_parked(title, text),
        )

    @staticmethod
    def _extract_title(html: str) -> str:
        m = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
        if not m:
            m = re.search(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)', html, re.I)
        return re.sub(r"\s+", " ", m.group(1)).strip()[:300] if m else ""

    @staticmethod
    def _html_to_text(html: str) -> str:
        html = re.sub(r"(?is)<(script|style|noscript|svg)[^>]*>.*?</\1>", " ", html)
        html = re.sub(r"(?s)<!--.*?-->", " ", html)
        html = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</h[1-6]>|</li>", "\n", html)
        text = re.sub(r"(?s)<[^>]+>", " ", html)
        text = (
            text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&quot;", '"')
            .replace("&#39;", "'").replace("&lt;", "<").replace("&gt;", ">")
        )
        text = re.sub(r"[ \t\u00a0]+", " ", text)
        return re.sub(r"\n\s*\n+", "\n", text).strip()[:40000]

    # --------------------------------------------------------------- verdicts
    def verify_candidate(
        self, name: str, city: str | None, phone: str | None, niche: str = "roofing"
    ) -> tuple[bool, str | None, float, str] | None:
        """Try to find the business's own site without any paid API.

        Returns ``(has_website, url, confidence, evidence)`` when conclusive,
        or ``None`` when we need the search/LLM layer to decide.
        """
        for url in guess_domains(name, niche):
            result = self.probe(url)
            if not result.ok:
                continue
            if result.parked:
                continue
            if is_blocklisted(result.final_url or url):
                continue
            mentioned, score = page_mentions_business(
                result.title, result.text, name, city, phone
            )
            if not mentioned:
                continue
            # Require strong corroboration: exact core name, or city+token.
            if score < 0.55:
                continue
            evidence = (
                f"guessed domain {url} -> {result.final_url} "
                f"(title={result.title[:80]!r}, score={score})"
            )
            return True, result.final_url or url, min(0.95, 0.6 + score / 2), evidence
        return None

    def probe_known_url(
        self, url: str, name: str, city: str | None, phone: str | None
    ) -> tuple[bool, str, float, str] | None:
        """Confirm a URL that some other source claims is the business site."""
        if not url:
            return None
        if is_blocklisted(url):
            return False, url, 0.9, f"{registrable_domain(url)} is a directory/social domain"
        result = self.probe(url)
        if not result.ok:
            return None
        if result.parked:
            return False, url, 0.85, f"parked/placeholder page at {result.final_url}"
        mentioned, score = page_mentions_business(result.title, result.text, name, city, phone)
        if mentioned and score >= 0.5:
            return True, result.final_url or url, min(0.98, 0.65 + score / 2), (
                f"probe ok: {result.final_url} title={result.title[:80]!r}"
            )
        return False, url, 0.5, "URL responded but page does not mention the business"

    @property
    def stats(self) -> dict[str, int]:
        return {"requests": self.request_count}


def domain_of(url: str) -> str:
    return registrable_domain(url) if url else ""


def url_is_business_site(url: str) -> bool:
    return bool(url) and not is_blocklisted(url) and bool(urlparse(url).hostname)
