"""OpenStreetMap discovery via Overpass (no API key, no ToS risk).

OSM is the free backbone of discovery: ``craft=roofer``,
``shop=roofing``, ``office=company`` + ``building=*`` nodes carry name, phone,
website and address tags. Critically, **the absence of a ``website`` tag is weak
evidence of no website** — treated as a hint, never as proof.

Overpass is slow and rate-limited; endpoints are tried in order with a short
timeout, and results are cached per bounding box.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from ..errors import ProviderError, RateLimited
from ..models import Candidate
from .cache import HttpCache
from .extract import canonical_phone
from .ratelimit import RateLimiter

log = logging.getLogger(__name__)

ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
)

# Tags that identify roofing / exterior contractors in OSM.
DEFAULT_TAGS = (
    '["craft"="roofer"]',
    '["shop"="roofing"]',
    '["office"="roofing"]',
    '["construction"="roofing"]',
)


class OverpassClient:
    def __init__(
        self,
        *,
        cache: HttpCache | None = None,
        timeout: float = 60.0,
        user_agent: str = "webdev-leads/1.0",
    ) -> None:
        self.cache = cache
        self.timeout = timeout
        self.user_agent = user_agent
        self.limiter = RateLimiter(rate_per_minute=30, burst=2)
        self.failures = 0
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent})

    def _query(self, ql: str) -> dict[str, Any]:
        self.limiter.acquire()
        last_err = ""
        for endpoint in ENDPOINTS:
            for attempt in range(2):
                try:
                    resp = self._session.post(
                        endpoint, data={"data": ql}, timeout=self.timeout
                    )
                    if resp.status_code == 200:
                        return resp.json()
                    if resp.status_code in (429, 504):
                        last_err = f"{endpoint} HTTP {resp.status_code}"
                        time.sleep(3 * (attempt + 1))
                        continue
                    last_err = f"{endpoint} HTTP {resp.status_code}: {resp.text[:120]}"
                    break
                except requests.RequestException as exc:
                    last_err = f"{endpoint}: {exc}"
                    time.sleep(2 * (attempt + 1))
            log.debug("overpass endpoint failed: %s", last_err)
        self.failures += 1
        raise ProviderError("overpass", last_err or "no endpoint answered")

    def search_bbox(
        self,
        *,
        south: float,
        west: float,
        north: float,
        east: float,
        tags: tuple[str, ...] = DEFAULT_TAGS,
        limit: int = 60,
    ) -> list[Candidate]:
        cache_payload = {
            "bbox": [round(south, 4), round(west, 4), round(north, 4), round(east, 4)],
            "tags": list(tags),
            "limit": limit,
        }
        if self.cache is not None:
            cached = self.cache.get("overpass.bbox", cache_payload)
            if cached is not None:
                return [Candidate(**c) for c in cached.get("candidates", [])]

        parts = "".join(
            f'nwr{t}({south},{west},{north},{east});' for t in tags
        )
        ql = f"[out:json][timeout:{int(self.timeout)}];({parts});out tags center {limit};"

        try:
            data = self._query(ql)
        except ProviderError:
            raise

        candidates: list[Candidate] = []
        for el in data.get("elements", []) or []:
            tags_ = el.get("tags") or {}
            name = tags_.get("name") or tags_.get("operator")
            if not name:
                continue
            addr_city = tags_.get("addr:city")
            addr_state = tags_.get("addr:state")
            addr_postcode = tags_.get("addr:postcode")
            street = " ".join(
                p for p in (tags_.get("addr:housenumber"), tags_.get("addr:street")) if p
            )
            address = ", ".join(
                p for p in (street, addr_city, " ".join(x for x in (addr_state, addr_postcode) if x)) if p
            )
            website = tags_.get("website") or tags_.get("contact:website") or tags_.get("url")
            category = (
                tags_.get("craft")
                or tags_.get("shop")
                or tags_.get("office")
                or tags_.get("construction")
                or "roofer"
            )
            candidates.append(
                Candidate(
                    name=name,
                    city=addr_city,
                    state=(addr_state or "")[:2] or None,
                    zip=addr_postcode,
                    phone=canonical_phone(tags_.get("phone") or tags_.get("contact:phone")),
                    address=address or None,
                    category=f"osm:{category}",
                    website=website,
                    source="overpass",
                    source_url=f"https://www.openstreetmap.org/{el.get('type','node')}/{el.get('id','')}",
                    evidence="osm tags: " + ", ".join(
                        f"{k}={v}"
                        for k, v in tags_.items()
                        if k in {"name", "website", "phone", "contact:phone", "addr:city", "craft", "shop"}
                    ),
                )
            )

        if self.cache is not None:
            self.cache.put(
                "overpass.bbox",
                cache_payload,
                {"candidates": [c.model_dump() for c in candidates]},
            )
        return candidates

    def search_city(self, city: str, state: str | None, **kwargs: Any) -> list[Candidate]:
        """Area-name query, used when we have no coordinates."""
        area = f'area["name"="{city}"]' + (f'["is_in:state"="{state}"]' if state else "")
        cache_payload = {"city": city, "state": state, **{k: v for k, v in kwargs.items() if k != "tags"}}
        if self.cache is not None:
            cached = self.cache.get("overpass.city", cache_payload)
            if cached is not None:
                return [Candidate(**c) for c in cached.get("candidates", [])]

        tags = kwargs.get("tags", DEFAULT_TAGS)
        parts = "".join(f"nwr{t}(area.searchArea);" for t in tags)
        ql = (
            f"[out:json][timeout:{int(self.timeout)}];"
            f'{area}->.searchArea;({parts});out tags center 60;'
        )
        data = self._query(ql)
        candidates: list[Candidate] = []
        for el in data.get("elements", []) or []:
            tags_ = el.get("tags") or {}
            name = tags_.get("name")
            if not name:
                continue
            candidates.append(
                Candidate(
                    name=name,
                    city=tags_.get("addr:city") or city,
                    state=(tags_.get("addr:state") or state or "")[:2] or None,
                    zip=tags_.get("addr:postcode"),
                    phone=canonical_phone(tags_.get("phone")),
                    address=tags_.get("addr:street"),
                    category="osm:roofer",
                    website=tags_.get("website"),
                    source="overpass",
                    source_url=f"https://www.openstreetmap.org/{el.get('type','node')}/{el.get('id','')}",
                    evidence=f"osm: {name}",
                )
            )
        if self.cache is not None:
            self.cache.put(
                "overpass.city", cache_payload, {"candidates": [c.model_dump() for c in candidates]}
            )
        return candidates
