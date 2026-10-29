"""Typed domain models shared by every agent.

These are the contracts between agents. If it is not in this file, an agent may
not invent it — that rule is what keeps the LLM from fabricating lead data.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Tier(str, Enum):
    A = "A"  # best fit, call first
    B = "B"  # good fit
    C = "C"  # marginal
    REJECT = "reject"

    @classmethod
    def parse(cls, value: Any, default: "Tier" = None) -> "Tier":  # type: ignore[assignment]
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            v = value.strip().lower()
            for member in cls:
                if member.value == v:
                    return member
        return default if default is not None else cls.C


class LeadStatus(str, Enum):
    DISCOVERED = "discovered"
    WEBSITE_FOUND = "website_found"  # disqualified: has a website
    NO_WEBSITE = "no_website"  # passed the core filter
    QUALIFIED = "qualified"  # scored and accepted
    REJECTED = "rejected"  # failed qualification or critic review
    NEEDS_REVIEW = "needs_review"  # ambiguous, human should look


class VerdictMethod(str, Enum):
    OSM_TAG = "osm_tag"
    DOMAIN_GUESS = "domain_guess"
    DOMAIN_PROBE = "domain_probe"
    SEARCH = "search"
    LLM = "llm"
    NONE = "none"


class ICP(BaseModel):
    """Ideal customer profile. Numeric gates are enforced in code, not by the LLM."""

    model_config = ConfigDict(extra="ignore")

    niche: str = "Roofing Contractor"
    min_rating: float = 4.5
    min_reviews: int = 10
    must_be_contractor: bool = True
    exclude_categories: list[str] = Field(
        default_factory=lambda: [
            "supplier",
            "supplies",
            "materials",
            "manufacturer",
            "distributor",
            "wholesale",
        ]
    )
    target_traits: list[str] = Field(
        default_factory=lambda: [
            "established local operator",
            "residential and/or commercial installation",
            "owner-operated or small team",
            "actively taking new jobs",
        ]
    )


class Campaign(BaseModel):
    """What the user asked for."""

    niche: str = "Roofing Contractor"
    state: str = "TX"
    zips: list[str] = Field(default_factory=list)
    max_zips: int = 25
    icp: ICP = Field(default_factory=ICP)

    @field_validator("state")
    @classmethod
    def _upper_state(cls, v: str) -> str:
        v = v.strip().upper()
        if not re.fullmatch(r"[A-Z]{2}", v):
            raise ValueError(f"state must be a 2-letter code, got {v!r}")
        return v

    @field_validator("zips")
    @classmethod
    def _clean_zips(cls, v: list[str]) -> list[str]:
        out: list[str] = []
        for z in v:
            z = str(z).strip().zfill(5)
            if re.fullmatch(r"\d{5}", z) and z not in out:
                out.append(z)
        return out


class RunPlan(BaseModel):
    """Output of the planner agent: how the run will be executed."""

    queries: list[str] = Field(default_factory=list)
    icp: ICP = Field(default_factory=ICP)
    source_priority: list[str] = Field(default_factory=lambda: ["overpass", "tavily"])
    notes: str = ""
    llm_generated: bool = False

    @field_validator("queries")
    @classmethod
    def _require_placeholder(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("plan must contain at least one query")
        cleaned: list[str] = []
        for q in v:
            q = " ".join(str(q).split())
            if "{city}" not in q and "{zip}" not in q:
                continue
            if q not in cleaned:
                cleaned.append(q)
        if not cleaned:
            raise ValueError("no query contained {city} or {zip} placeholder")
        return cleaned


class Candidate(BaseModel):
    """A business we found but have not yet verified."""

    model_config = ConfigDict(extra="ignore")

    name: str
    city: str | None = None
    state: str | None = None
    zip: str | None = None
    phone: str | None = None
    address: str | None = None
    category: str | None = None
    rating: float | None = None
    reviews: int | None = None
    website: str | None = None
    source: str = "unknown"
    source_url: str = ""
    evidence: str = ""

    @field_validator("rating", mode="before")
    @classmethod
    def _coerce_rating(cls, v: Any) -> float | None:
        if v is None or v == "":
            return None
        try:
            f = float(str(v).replace(",", "."))
        except (TypeError, ValueError):
            return None
        return f if 0.0 <= f <= 5.0 else None

    @field_validator("reviews", mode="before")
    @classmethod
    def _coerce_reviews(cls, v: Any) -> int | None:
        if v is None or v == "":
            return None
        m = re.search(r"\d[\d,]*", str(v))
        if not m:
            return None
        return int(m.group(0).replace(",", ""))

    @field_validator("name")
    @classmethod
    def _clean_name(cls, v: str) -> str:
        v = " ".join(str(v).split()).strip(" -|,.;:")
        if not v:
            raise ValueError("candidate name is empty")
        return v[:120]


class WebsiteVerdict(BaseModel):
    """The deciding question of the whole project: do they own a website?"""

    has_website: bool
    website_url: str | None = None
    confidence: float = 0.5
    method: VerdictMethod = VerdictMethod.NONE
    evidence: str = ""
    ambiguous: bool = False

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


class Qualification(BaseModel):
    """Fit scoring for a website-less business."""

    fit: int = 0
    tier: Tier = Tier.C
    reason: str = ""
    signals: dict[str, Any] = Field(default_factory=dict)
    llm_generated: bool = False


class CriticReview(BaseModel):
    """Independent audit of a lead before it reaches the customer."""

    accepted: bool = True
    issues: list[str] = Field(default_factory=list)
    corrected_name: str | None = None
    rationale: str = ""
    llm_generated: bool = False


class Lead(BaseModel):
    """Fully assembled lead, as exported."""

    run_id: str
    candidate_id: int
    name: str
    city: str | None = None
    state: str | None = None
    zip: str | None = None
    phone: str | None = None
    address: str | None = None
    category: str | None = None
    rating: float | None = None
    reviews: int | None = None
    has_website: bool | None = None
    website_url: str | None = None
    website_confidence: float | None = None
    website_method: str | None = None
    fit: int | None = None
    tier: str | None = None
    qualifier_reason: str = ""
    critic_status: str | None = None
    critic_notes: str = ""
    status: str = LeadStatus.DISCOVERED.value
    source: str = ""
    source_url: str = ""


class Cell(BaseModel):
    """One unit of discovery work: a ZIP code and the city it belongs to."""

    zip: str
    city: str = ""
    state: str = ""
    lat: float | None = None
    lon: float | None = None

    @property
    def key(self) -> str:
        return f"{self.zip}"

    @property
    def label(self) -> str:
        return f"{self.city}, {self.state} {self.zip}".strip(", ")


class RunStats(BaseModel):
    """Counters surfaced in the run report and CLI status output."""

    cells_total: int = 0
    cells_done: int = 0
    cells_failed: int = 0
    candidates: int = 0
    with_website: int = 0
    without_website: int = 0
    ambiguous: int = 0
    qualified: int = 0
    rejected: int = 0
    tier_a: int = 0
    tier_b: int = 0
    tier_c: int = 0
    llm_calls: int = 0
    llm_tokens: int = 0
    tavily_credits: float = 0
    http_requests: int = 0
    started_at: str = Field(default_factory=utcnow)
    finished_at: str | None = None
