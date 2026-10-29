"""Discovery agent: find candidate businesses for one ZIP/city cell.

Discovery is deliberately *mostly deterministic*. It combines:

  * OpenStreetMap via Overpass — free, structured, no ToS risk;
  * Tavily search — finds directory/list pages that name many businesses;
  * Tavily extract — pulls full page text for the best list pages;
  * rule-based parsing (``tools.extract``) to turn text into records.

The LLM is used only as a fallback when a rich page yields zero records, and it
is capped so a bad page cannot burn the budget.

Every candidate carries the URL and a verbatim snippet it came from, which the
critic agent later checks.
"""

from __future__ import annotations

from ..errors import BudgetExceeded, JSONParseError, LLMUnavailable, ProviderError, RateLimited
from ..llm import prompts
from ..models import Candidate, Cell, RunPlan
from ..tools.extract import (
    extract_candidates,
    is_blocklisted,
    is_plausible_business_name,
    page_looks_like_listing,
)
from .base import BaseAgent

# Pages we never mine for listings (search engines, social feeds, marketplaces).
SKIP_DOMAINS = (
    "google.com", "bing.com", "duckduckgo.com", "yahoo.com", "youtube.com",
    "facebook.com", "instagram.com", "x.com", "twitter.com", "tiktok.com",
    "pinterest.com", "reddit.com", "linkedin.com", "amazon.com", "ebay.com",
)


class DiscoveryAgent(BaseAgent):
    name = "discovery"

    def __init__(self, ctx, *, max_queries: int = 3, max_extracts: int = 3) -> None:
        super().__init__(ctx)
        self.max_queries = max_queries
        self.max_extracts = max_extracts

    # ------------------------------------------------------------------ main
    def discover(self, cell: Cell, plan: RunPlan, *, use_osm: bool = True) -> list[Candidate]:
        found: dict[str, Candidate] = {}

        if use_osm and self.ctx.overpass is not None and "overpass" in plan.source_priority:
            for cand in self._from_osm(cell):
                self._add(found, cand)

        for cand in self._from_search(cell, plan):
            self._add(found, cand)

        self.event(
            "info",
            f"discovery: {cell.label} -> {len(found)} candidates",
            {"zip": cell.zip, "city": cell.city, "count": len(found)},
        )
        return list(found.values())

    @staticmethod
    def _add(found: dict[str, Candidate], cand: Candidate) -> None:
        from ..tools.extract import dedupe_key

        if not is_plausible_business_name(cand.name):
            return
        key = dedupe_key(cand.name, cand.city, cand.zip)
        if not key:
            return
        if key in found:
            existing = found[key]
            for f in ("phone", "address", "city", "state", "zip", "category", "rating", "reviews"):
                if getattr(existing, f) in (None, "", 0) and getattr(cand, f) not in (None, "", 0):
                    setattr(existing, f, getattr(cand, f))
            return
        found[key] = cand

    # ------------------------------------------------------------------- OSM
    def _from_osm(self, cell: Cell) -> list[Candidate]:
        """Bounding-box query around the ZIP.

        Area-name queries time out on the public Overpass instances, and a
        ZIP-centered box is both cheaper and more precise. Failure is non-fatal:
        Tavily discovery continues without OSM.
        """
        assert self.ctx.overpass is not None
        try:
            if cell.lat is not None and cell.lon is not None:
                # ~12 km each side: covers a city ZIP and its neighbours.
                cands = self.ctx.overpass.search_bbox(
                    south=cell.lat - 0.12,
                    west=cell.lon - 0.14,
                    north=cell.lat + 0.12,
                    east=cell.lon + 0.14,
                )
            else:
                cands = self.ctx.overpass.search_city(cell.city, cell.state)
        except Exception as exc:
            self.event("warning", f"discovery: overpass unavailable for {cell.label}: {exc}")
            return []

        for c in cands:
            c.zip = cell.zip
            c.city = c.city or cell.city
            c.state = cell.state
        return cands

    # ---------------------------------------------------------------- Tavily
    def _from_search(self, cell: Cell, plan: RunPlan) -> list[Candidate]:
        if self.ctx.tavily is None:
            return []

        place_city = cell.city or cell.zip
        queries: list[str] = []
        for template in plan.queries[: self.max_queries]:
            try:
                q = template.format(
                    city=place_city, zip=cell.zip, state=cell.state, niche="roofing contractor"
                )
            except (KeyError, IndexError):
                q = template
            if q not in queries:
                queries.append(q)

        results: list[dict] = []
        for q in queries:
            try:
                results.extend(
                    self.ctx.tavily.search(
                        q,
                        depth="basic",
                        max_results=self.ctx.settings.max_results_per_query,
                        # Full page text at no extra credit — this is the single
                        # biggest lever on extraction quality.
                        include_raw_content=True,
                    )
                )
            except BudgetExceeded:
                self.event("warning", "discovery: Tavily budget reached; stopping searches")
                break
            except (ProviderError, RateLimited) as exc:
                self.event("warning", f"discovery: search failed for {q!r}: {exc}")
                continue

        if not results:
            return []

        # Deduplicate result URLs, drop obvious non-listing pages.
        seen: set[str] = set()
        pages: list[dict] = []
        for r in results:
            url = r.get("url", "")
            if not url or url in seen:
                continue
            seen.add(url)
            if any(d in url for d in SKIP_DOMAINS):
                continue
            pages.append(r)

        candidates: list[Candidate] = []
        thin_pages: list[str] = []

        for page in pages:
            body = "\n".join(
                part for part in (page.get("content"), page.get("raw_content")) if part
            )
            cands = extract_candidates(
                body,
                source_url=page["url"],
                source="tavily_search",
                default_city=cell.city or None,
                default_state=cell.state or None,
                default_zip=cell.zip,
                expected_state=cell.state,
            )
            if page_looks_like_listing(page["url"], page.get("title", ""), len(cands)):
                candidates.extend(cands)
                if len(cands) < 3 and not page.get("raw_content"):
                    thin_pages.append(page["url"])

        # Only pay for extraction when a real list page returned nothing usable.
        to_extract = thin_pages[: self.max_extracts]
        if to_extract:
            try:
                extracted = self.ctx.tavily.extract(to_extract)
            except BudgetExceeded:
                self.event("warning", "discovery: Tavily budget reached; skipping extract")
                extracted = {}
            except (ProviderError, RateLimited) as exc:
                self.event("warning", f"discovery: extract failed: {exc}")
                extracted = {}

            for url, text in extracted.items():
                cands = extract_candidates(
                    text,
                    source_url=url,
                    source="tavily_extract",
                    default_city=cell.city or None,
                    default_state=cell.state or None,
                    default_zip=cell.zip,
                    expected_state=cell.state,
                )
                if page_looks_like_listing(url, "", len(cands)):
                    candidates.extend(cands)
                elif not cands:
                    candidates.extend(self._llm_extract(text, url, cell))

        return candidates

    # ----------------------------------------------------------- LLM fallback
    def _llm_extract(self, text: str, url: str, cell: Cell) -> list[Candidate]:
        """Last resort for a rich page that rules could not parse."""
        if not self.llm_available or len(text) < 800:
            return []

        self.ctx.bump("llm_extract_calls")
        try:
            data = self.ask_llm_json(
                system=prompts.EXTRACTOR_SYSTEM,
                user=prompts.extractor_user(text, cell.city or cell.zip, cell.state),
                max_tokens=1200,
            )
        except (LLMUnavailable, BudgetExceeded) as exc:
            self.event("warning", f"discovery: LLM extract unavailable: {exc}")
            return []
        except Exception as exc:
            self.event("warning", f"discovery: LLM extract failed: {exc}")
            return []

        out: list[Candidate] = []
        for item in (data.get("businesses") or [])[:8]:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            try:
                cand = Candidate(
                    name=str(item["name"]),
                    phone=item.get("phone"),
                    address=item.get("address"),
                    city=item.get("city") or cell.city or None,
                    state=cell.state,  # cell's state is authoritative; don't trust LLM's
                    zip=item.get("zip") or cell.zip,
                    rating=item.get("rating"),
                    reviews=item.get("reviews"),
                    source="llm_extract",
                    source_url=url,
                    evidence=str(item.get("name"))[:200],
                )
            except Exception:
                continue
            out.append(cand)

        if out:
            self.event("info", f"discovery: LLM recovered {len(out)} records from {url}")
            self.ctx.log_event(
                self.ctx.run_id, "warning", self.name,
                "candidates recovered by LLM (not rule-verified)",
                {"url": url, "count": len(out)},
            )
        return out
