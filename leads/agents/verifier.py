"""Verifier agent: does this business own a website?

This is the question the whole campaign hinges on, so it is answered in three
escalating steps, cheapest first:

  1. **Known URL** (OSM tag) — probe it. A directory/social URL is not a website.
  2. **Domain guessing** — ``acmeroofing.com`` etc., probed and matched against
     the business name. Zero API cost, resolves most established businesses.
  3. **Tavily search** — find any non-directory domain that mentions the
     business. Probe each hit.
  4. **LLM adjudication** — only when the evidence conflicts, and only for a
     boolean. The model never supplies the URL.

Absence of evidence is *not* proof: a verdict below the confidence threshold is
flagged ``ambiguous`` and sent to human review instead of being exported as a
lead.
"""

from __future__ import annotations

from ..errors import JSONParseError, LLMUnavailable
from ..llm import prompts
from ..models import WebsiteVerdict, VerdictMethod
from ..tools.extract import is_blocklisted, registrable_domain
from .base import BaseAgent

CONFIDENT_NO_WEBSITE = 0.55


class VerifierAgent(BaseAgent):
    name = "verifier"

    def __init__(self, ctx, *, use_search: bool = True) -> None:
        super().__init__(ctx)
        self.use_search = use_search

    # ------------------------------------------------------------------ main
    def verify(
        self,
        *,
        name: str,
        city: str | None,
        state: str | None,
        phone: str | None,
        known_website: str | None = None,
        niche: str = "roofing",
    ) -> WebsiteVerdict:
        probe = self.ctx.probe
        probe_notes: list[str] = []

        # --- 1. a URL another source claims is theirs
        if known_website and probe is not None:
            result = probe.probe_known_url(known_website, name, city, phone)
            if result is not None:
                has_site, url, confidence, evidence = result
                probe_notes.append(evidence)
                if has_site:
                    return WebsiteVerdict(
                        has_website=True,
                        website_url=url,
                        confidence=confidence,
                        method=VerdictMethod.OSM_TAG,
                        evidence=evidence,
                    )
                # The tagged URL was a directory or dead page: keep digging.
                probe_notes.append(f"tagged website rejected: {evidence}")

        # --- 2. free domain guessing
        if probe is not None:
            guessed = probe.verify_candidate(name, city, phone, niche)
            if guessed is not None:
                has_site, url, confidence, evidence = guessed
                return WebsiteVerdict(
                    has_website=has_site,
                    website_url=url,
                    confidence=confidence,
                    method=VerdictMethod.DOMAIN_GUESS,
                    evidence=evidence,
                )

        evidence_lines: list[str] = []
        domain_hits: list[str] = []

        # --- 3. paid search
        if self.use_search and self.ctx.tavily is not None:
            query = " ".join(x for x in (name, city, state) if x)
            try:
                results = self.ctx.tavily.search(query, depth="basic", max_results=4)
            except Exception as exc:
                self.event("warning", f"verifier: search failed for {name}: {exc}")
                results = []

            for r in results:
                url = r.get("url", "")
                if not url:
                    continue
                tag = "directory/social" if is_blocklisted(url) else "business-domain"
                evidence_lines.append(
                    f"{r.get('title', '')[:90]} — {url} ({tag}) — "
                    f"{(r.get('content') or '')[:160]}"
                )
                if not is_blocklisted(url):
                    domain_hits.append(url)

            # Probe every non-directory domain found.
            if probe is not None:
                for url in domain_hits[:3]:
                    result = probe.probe_known_url(url, name, city, phone)
                    if result is None:
                        continue
                    has_site, final_url, confidence, evidence = result
                    probe_notes.append(evidence)
                    if has_site:
                        return WebsiteVerdict(
                            has_website=True,
                            website_url=final_url,
                            confidence=confidence,
                            method=VerdictMethod.DOMAIN_PROBE,
                            evidence=evidence + f" | search: {query}",
                        )

            # Only directories and social pages found, pages probed cleanly.
            if results and not domain_hits:
                return WebsiteVerdict(
                    has_website=False,
                    confidence=0.7,
                    method=VerdictMethod.SEARCH,
                    evidence=(
                        f"search '{query}' returned {len(results)} results, none a "
                        "business-owned domain: "
                        + "; ".join(f"{registrable_domain(r.get('url', ''))}" for r in results[:4])
                    ),
                )

        # --- 4. LLM adjudication of conflicting evidence
        if evidence_lines or probe_notes:
            verdict = self._adjudicate(
                name=name, city=city, state=state, phone=phone,
                evidence_lines=evidence_lines, probe_notes=probe_notes,
            )
            if verdict is not None:
                return verdict

        # --- nothing conclusive
        return WebsiteVerdict(
            has_website=False,
            confidence=0.3,
            method=VerdictMethod.NONE,
            evidence="no evidence gathered (search unavailable or empty)",
            ambiguous=True,
        )

    # ------------------------------------------------------------ LLM step
    def _adjudicate(
        self,
        *,
        name: str,
        city: str | None,
        state: str | None,
        phone: str | None,
        evidence_lines: list[str],
        probe_notes: list[str],
    ) -> WebsiteVerdict | None:
        if not self.llm_available:
            return None

        self.ctx.bump("llm_verify_calls")
        try:
            data = self.ask_llm_json(
                system=prompts.VERIFIER_SYSTEM,
                user=prompts.verifier_user(
                    name=name, city=city, state=state, phone=phone,
                    evidence_lines=evidence_lines, probe_notes=probe_notes,
                ),
                max_tokens=900,
            )
        except (LLMUnavailable, JSONParseError) as exc:
            self.event("warning", f"verifier: LLM adjudication unavailable: {exc}")
            return None
        except Exception as exc:
            self.event("warning", f"verifier: LLM adjudication failed: {exc}")
            return None

        has_site = bool(data.get("has_website"))
        try:
            confidence = float(data.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5

        url = data.get("website_url") or None
        # Never trust a model-supplied URL that is a directory.
        if url and is_blocklisted(str(url)):
            url = None
            if has_site:
                has_site = False

        ambiguous = confidence < CONFIDENT_NO_WEBSITE and not has_site
        return WebsiteVerdict(
            has_website=has_site,
            website_url=str(url)[:300] if url else None,
            confidence=max(0.0, min(1.0, confidence)),
            method=VerdictMethod.LLM,
            evidence=str(data.get("reason", ""))[:500],
            ambiguous=ambiguous,
        )
