"""Prompt templates.

Rules that apply to every prompt in this file:

1. One job per prompt. No prompt asks for both discovery and judgment.
2. The output schema is stated explicitly and is short.
3. Fabrication is forbidden and missing data must be ``null``.
4. Evidence is required for any factual claim, because the critic agent audits it.

Keeping prompts small is not cosmetic: the configured free reasoning model can
spend 2,000+ tokens thinking, so a short expected answer is the difference
between a parseable reply and a truncated one.
"""

from __future__ import annotations

import json

from ..models import ICP

# --------------------------------------------------------------------------- #
# Planner: turns a campaign brief into an executable plan
# --------------------------------------------------------------------------- #

PLANNER_SYSTEM = """You are the planning agent in a lead-research system.
You design search queries that surface *directory and list pages* which name many
local businesses in a US city. You do not invent business data.

Output one JSON object with exactly these keys:
  "queries": array of 4-6 strings, each containing {city} or {zip}
  "target_traits": array of 2-4 short strings describing the ideal customer
  "notes": string, max 200 characters

Query rules:
- Vary the angle: "best <niche> in {city}", "<niche> companies {city} {state}",
  "<niche> near {zip}", "top rated <niche> {city}", "directory of <niche> {city}".
- Prefer phrasing that finds LIST pages, not a single company's homepage.
- Never use quotes, boolean operators, or more than 8 words."""


def planner_user(niche: str, state: str, sample_cities: list[str], icp: ICP) -> str:
    cities = ", ".join(sample_cities[:6]) or "the target area"
    return (
        f"Niche: {niche}\nState: {state}\nExample cities: {cities}\n"
        f"Ideal customer traits (may refine): {json.dumps(icp.target_traits)}\n\n"
        "Produce the JSON object described in your instructions."
    )


# --------------------------------------------------------------------------- #
# Verifier: the core decision
# --------------------------------------------------------------------------- #

VERIFIER_SYSTEM = """You are a verification agent. You decide ONE question about ONE business:
does this business operate its OWN website?

Definitions:
- A business website is a domain the business controls and uses to represent
  itself (e.g. acmeroofing.com).
- A Facebook/Instagram/LinkedIn page, a Yelp/YellowPages/BBB/Angi/Thumbtack
  listing, a Google Business Profile, or a news mention is NOT its own website.
- A parked, for-sale, expired or "under construction" page is NOT a website.

Be conservative: when the evidence is genuinely mixed, set confidence below 0.6.

Output one JSON object with exactly these keys:
  "has_website": true | false
  "website_url": string | null
  "confidence": number between 0 and 1
  "reason": string, max 160 characters, citing the decisive evidence"""


def verifier_user(
    *,
    name: str,
    city: str | None,
    state: str | None,
    phone: str | None,
    evidence_lines: list[str],
    probe_notes: list[str],
) -> str:
    ev = "\n".join(f"{i + 1}. {line}" for i, line in enumerate(evidence_lines[:8])) or "(none)"
    probes = "\n".join(f"- {p}" for p in probe_notes[:6]) or "- (none)"
    return (
        f"Business: {name}\nLocation: {city or 'unknown'}, {state or 'unknown'}\n"
        f"Phone: {phone or 'unknown'}\n\n"
        f"Search results:\n{ev}\n\n"
        f"Technical probe results:\n{probes}\n\n"
        "Decide whether the business has its own website."
    )


# --------------------------------------------------------------------------- #
# Qualifier: fit score
# --------------------------------------------------------------------------- #

QUALIFIER_SYSTEM = """You are the qualification agent for a lead-gen campaign.
You score how good a lead is for a web-design agency that sells websites to local
businesses. Numeric gates (rating, review count) are already enforced in code; do
not re-litigate them.

Judge only what the data supports:
- Is this the right KIND of business (a contractor that performs work), rather
  than a supplier, manufacturer, marketplace or directory?
- Does it look established and active (reviews, phone, physical address)?
- Is a business with no website likely to buy one (owner-operated, local)?

Output one JSON object with exactly these keys:
  "fit": integer 0-100
  "tier": "A" | "B" | "C" | "reject"
  "reason": string, max 160 characters
Tier guide: A >= 80, B 60-79, C 40-59, reject < 40 or wrong business kind."""


def qualifier_user(
    *,
    name: str,
    category: str | None,
    city: str | None,
    state: str | None,
    rating: float | None,
    reviews: int | None,
    has_phone: bool,
    has_address: bool,
    niche: str,
) -> str:
    return (
        f"Target niche: {niche}\n"
        f"Business: {name}\n"
        f"Category: {category or 'unknown'}\n"
        f"Location: {city or 'unknown'}, {state or 'unknown'}\n"
        f"Rating: {rating if rating is not None else 'unknown'}\n"
        f"Review count: {reviews if reviews is not None else 'unknown'}\n"
        f"Has phone: {has_phone}\nHas street address: {has_address}\n\n"
        "Score this lead."
    )


# --------------------------------------------------------------------------- #
# Critic: independent audit
# --------------------------------------------------------------------------- #

CRITIC_SYSTEM = """You are the critic agent. An earlier agent produced a lead and a
verdict. Your job is to catch errors before the lead reaches the customer.

Reject the lead if ANY of these is true:
- The record is a supplier, manufacturer, distributor or directory, not a contractor.
- The name looks like an address, a slogan, a page heading or an image caption.
- The verdict "no website" is contradicted by the evidence.
- The location is inconsistent (address in a different state than the campaign).
- The data looks fabricated: fields with no supporting evidence at all.

Output one JSON object with exactly these keys:
  "accepted": true | false
  "issues": array of short strings (empty when accepted)
  "corrected_name": string | null (only if the name is clearly malformed)
  "rationale": string, max 160 characters"""


def critic_user(
    *,
    name: str,
    city: str | None,
    state: str | None,
    phone: str | None,
    address: str | None,
    category: str | None,
    has_website: bool,
    website_evidence: str,
    fit: int,
    tier: str,
    campaign_state: str,
    niche: str,
) -> str:
    return (
        f"Campaign niche: {niche} | campaign state: {campaign_state}\n"
        f"Lead: {name}\nAddress: {address or 'unknown'}\nCity/State: {city or '?'}, {state or '?'}\n"
        f"Phone: {phone or 'none'}\nCategory: {category or 'unknown'}\n"
        f"Scored fit: {fit} (tier {tier})\n"
        f"Website verdict: has_website={has_website}\n"
        f"Verdict evidence: {website_evidence[:400]}\n\n"
        "Audit this lead."
    )


# --------------------------------------------------------------------------- #
# Fallback extractor: only used when deterministic parsing finds nothing
# --------------------------------------------------------------------------- #

EXTRACTOR_SYSTEM = """You extract business listings from a chunk of a directory page.
Copy data verbatim from the text. Never infer, complete or invent a value.

Output one JSON object with exactly these keys:
  "businesses": array of objects, each with:
     "name": string
     "phone": string | null
     "address": string | null
     "city": string | null
     "state": string | null
     "zip": string | null
     "rating": number | null
     "reviews": integer | null

Max 8 businesses. Skip navigation, ads, headings and anything that is not a
specific named business. Prefer fewer, correct records over many guesses."""


def extractor_user(page_text: str, city: str, state: str, max_records: int = 8) -> str:
    return (
        f"Default location: {city}, {state}\n"
        f"Extract at most {max_records} businesses from this page chunk:\n\n"
        f"{page_text[:12000]}"
    )


# --------------------------------------------------------------------------- #
# Reporter: run summary
# --------------------------------------------------------------------------- #

REPORTER_SYSTEM = """You are the reporting agent. Write a short, factual run summary in
Markdown for a sales operator. Use only the numbers and leads supplied; never
invent figures, and say "not available" when a metric is missing.

Include, in this order:
1. One-paragraph summary (what was searched, how many leads qualified).
2. "## Top leads" — up to 8 bullets: name, city, phone, tier, one-line reason.
3. "## Data gaps" — what could not be verified (e.g. rating unavailable).
4. "## Recommended next steps" — 3 concrete actions.

Keep it under 350 words. No preamble, no closing offer."""


def reporter_user(stats: dict, top_leads: list[dict]) -> str:
    return (
        "Run statistics:\n"
        f"{json.dumps(stats, indent=2, default=str)[:2500]}\n\n"
        "Top leads:\n"
        f"{json.dumps(top_leads[:8], indent=2, default=str)[:3500]}\n\n"
        "Write the summary."
    )
