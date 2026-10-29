"""Deterministic text -> lead extraction.

The LLM never produces lead data here. Directory pages, snippets and OSM tags
are parsed by rules, so every field is traceable to a source. The reason is
practical: a free reasoning model truncates long JSON, and hallucinated phone
numbers are worse than missing ones.

Every extractor returns ``Candidate`` objects with ``source_url`` and a verbatim
``evidence`` snippet for audit.
"""

from __future__ import annotations

import re
import unicodedata
from urllib.parse import urlparse

from ..models import Candidate

# --------------------------------------------------------------------------- #
# Vocabularies
# --------------------------------------------------------------------------- #

LEGAL_SUFFIXES = (
    "llc", "l.l.c.", "inc", "inc.", "incorporated", "co", "co.", "corp", "corp.",
    "corporation", "ltd", "ltd.", "limited", "lp", "llp", "pllc", "pc", "dba",
    "company", "group", "holdings",
)

BUSINESS_WORDS = (
    "roof", "roofing", "construction", "contractor", "builders", "builder",
    "exteriors", "siding", "gutters", "remodel", "restoration", "homes",
    "home", "services", "supply", "supplies", "materials", "shingles",
    "metal", "commercial", "residential", "install", "repair", "solutions",
)

# Lines that are never a business name.
NAME_STOPWORDS = {
    "read more", "learn more", "see more", "view all", "show more", "contact us",
    "call now", "get quote", "free estimate", "write a review", "reviews",
    "rating", "ratings", "directions", "website", "phone", "hours", "open now",
    "closed", "sponsored", "advertisement", "ad", "verified", "claim this listing",
    "add photo", "share", "save", "menu", "home", "services", "about us",
    "privacy policy", "terms of service", "sitemap", "blog", "results",
    "sort by", "filter", "nearby", "more info", "read less", "see less",
    "active contractor", "add filters", "add filters to refine the results",
    "refine the results", "clear filters", "apply filters", "load more",
    "show more results", "see all results", "view profile", "call business",
    "visit website", "get directions", "sponsored listing", "featured",
    "top rated", "highly rated", "our team", "our services", "our work",
    "reviews of", "leave a review", "overview", "photos", "gallery",
    "federal government", "government", "military", "verified business",
}

# Pure navigation/UI strings that appear in a name position.
UI_NAMES = {
    "home", "about", "contact", "services", "reviews", "menu", "results",
    "photos", "gallery", "overview", "blog", "sitemap", "login", "sign up",
    "search", "filter", "sort", "next", "previous", "back to top", "share",
}

ADDRESS_RE = re.compile(
    r"(?<!\d)(\d{1,6}\s+[A-Za-z0-9][A-Za-z0-9 .,'#\-]{2,70})\b"
    r"\s*[,]?\s*"
    r"([A-Za-z][A-Za-z .'\-]{1,25}?)"
    r"\s*[,]?\s*"
    r"([A-Z]{2})"
    r"\s*,?\s*"
    r"(\d{5})(?:-\d{4})?\b"
)

PHONE_RE = re.compile(
    r"(?<![\d.])(?:\+?1[\s.\-]?)?"
    r"\(?([2-9]\d{2})\)?[\s.\-]"
    r"([2-9]\d{2})[\s.\-]"
    r"(\d{4})(?![\d.])"
)

RATING_RE = re.compile(
    r"(?<![\d.])([0-5](?:\.\d)?)\s*(?:/\s*5|stars?|★)?\s*[\(\-–—|·]?\s*"
    r"(\d[\d,]{0,6})\s*(?:\+)?\s*(?:reviews?|ratings?|votes?)\b",
    re.IGNORECASE,
)

ZIP_RE = re.compile(r"(?<!\d)(\d{5})(?:-\d{4})?(?!\d)")

# Domains that are aggregators, social networks or marketplaces. A listing on
# one of these is *not* the business owning a website.
BLOCKED_DOMAINS = {
    "yelp.com", "yellowpages.com", "yellowbook.com", "bbb.org", "angi.com",
    "angieslist.com", "homeadvisor.com", "houzz.com", "thumbtack.com",
    "nextdoor.com", "manta.com", "buildzoom.com", "porch.com", "networx.com",
    "facebook.com", "fb.com", "instagram.com", "linkedin.com", "twitter.com",
    "x.com", "tiktok.com", "pinterest.com", "youtube.com", "reddit.com",
    "google.com", "goo.gl", "bing.com", "yahoo.com", "duckduckgo.com",
    "mapquest.com", "maps.google.com", "foursquare.com", "yellowpages.ca",
    "superpages.com", "citysearch.com", "kudzu.com", "bizapedia.com",
    "opencorporates.com", "dnb.com", "zoominfo.com", "crunchbase.com",
    "indeed.com", "glassdoor.com", "craigslist.org", "ebay.com", "amazon.com",
    "expertise.com", "birdeye.com", "chamberofcommerce.com", "patch.com",
    "prnewswire.com", "businesswire.com", "forbes.com", "wsj.com", "bloomberg.com",
    "downtobid.com", "builderspace.com", "construction.com", "buildforce.com",
    "houzz.co.uk", "trustpilot.com", "nocowboys.com", "bark.com", "bark.us",
    "wikipedia.org", "wikimedia.org", "tripadvisor.com", "schema.org",
    "sites.google.com", "business.site", "godaddysites.com", "wixsite.com",
    "weebly.com", "wordpress.com", "blogspot.com", "square.site",
}

PARKED_HINTS = (
    "domain is for sale", "buy this domain", "this domain may be for sale",
    "domain parking", "parked free", "account suspended", "site not found",
    "coming soon", "under construction", "index of /", "default web page",
    "this site can’t be reached", "this site can't be reached",
)

STREET_SUFFIX_RE = re.compile(
    r"\b(st|street|ave|avenue|rd|road|dr|drive|blvd|boulevard|ln|lane|way|ct|court|"
    r"pl|place|plaza|pkwy|parkway|hwy|highway|fwy|freeway|trl|trail|cir|circle|"
    r"suite|ste|unit|apt|building|bldg|floor|fl)\b",
    re.IGNORECASE,
)

# "Athens, TX 75751" — a location, not a company.
CITY_STATE_ZIP_RE = re.compile(r"^[A-Za-z][A-Za-z .'\-]*,\s*[A-Z]{2}\s*\d{5}$")
ANY_PHONE_IN_NAME_RE = re.compile(r"\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}")
SLOGAN_RE = re.compile(
    r"^(call|contact|text|book|schedule|get|request|free|serving|trusted|licensed|"
    r"insured|family|locally|award|need|looking|welcome|we|our)\b",
    re.IGNORECASE,
)

# "15 Best Roofing contractors in Dallas, TX" — listicle headings, not companies.
LISTICLE_RE = re.compile(
    r"^\d+\s+(best|top|greatest|leading)\b|\b(best|top)\b.*\b(in|near)\b|"
    r"\bcontractors\s+in\b|\bcompanies\s+in\b|\bnear\s+me\b",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------- #
# Normalisation helpers
# --------------------------------------------------------------------------- #


def normalize_ws(text: str) -> str:
    return re.sub(r"[ \t\u00a0]+", " ", text).strip()


def percent_decode(text: str) -> str:
    """Tavily/HTML sometimes hands back URL-encoded fragments (%20, %2C)."""
    if "%" not in text:
        return text
    return re.sub(
        r"%([0-9A-Fa-f]{2})",
        lambda m: chr(int(m.group(1), 16)),
        text,
    ).replace("+", " ")


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def clean_name(raw: str) -> str:
    """Normalise a business name without destroying it."""
    name = strip_accents(raw)
    name = percent_decode(name)
    name = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", name)                        # images
    name = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", name)                     # links
    name = re.sub(r"[*_`]+", "", name)                                      # emphasis
    # Strip markdown/list decoration, repeatedly ("### 1. Name" -> "Name").
    for _ in range(4):
        stripped = re.sub(
            r"^\s*(?:#{1,6}\s*|[-*•·]\s+|\d{1,2}[.)]\s+|>\s*)", "", name
        )
        if stripped == name:
            break
        name = stripped
    name = normalize_ws(name)
    # Trailing location suffix: "Kidd Roofing - Dallas, TX 75201" / "Acme, TX".
    name = re.sub(r"[,\s]+[A-Z]{2}\s*\d{5}$", "", name)
    name = re.sub(r",\s*[A-Z]{2}$", "", name)
    name = name.strip(" -|,.;:•·—–")
    # Drop trailing marketing tails: "Joe's Roofing - Dallas, TX"
    name = re.split(r"\s+[|—–]\s+", name)[0]
    return normalize_ws(name)[:120]


def name_core(name: str) -> str:
    """Name minus legal suffixes, for matching and domain guessing."""
    n = strip_accents(name).lower()
    n = re.sub(r"[^a-z0-9&\s]", " ", n)
    tokens = [t for t in n.split() if t]
    while tokens and tokens[-1] in {s.strip(".") for s in LEGAL_SUFFIXES}:
        tokens.pop()
    if tokens and tokens[0] == "the":
        tokens = tokens[1:]
    return " ".join(tokens)


def dedupe_key(name: str, city: str | None = None, zip_code: str | None = None) -> str:
    """Stable identity for a business across sources and ZIP cells.

    City is preferred over ZIP: the same business discovered via two adjacent
    ZIP codes in one city must collapse to one lead.
    """
    core = re.sub(r"[^a-z0-9]+", "", name_core(name))
    place = (city or zip_code or "").strip().lower()
    place = re.sub(r"[^a-z0-9]+", "", place)
    return f"{core}|{place}" if place else core


def name_tokens(name: str) -> list[str]:
    """Distinctive tokens used to confirm a page is about this business."""
    generic = {
        "roofing", "roof", "construction", "contractor", "contractors", "company",
        "services", "service", "group", "builders", "builder", "exteriors",
        "solutions", "home", "homes", "the", "and", "llc", "inc", "co", "corp",
        "supply", "supplies", "materials", "metal", "residential", "commercial",
        "repair", "install", "restoration", "remodel", "siding", "gutters",
    }
    return [t for t in name_core(name).split() if len(t) >= 3 and t not in generic]


def canonical_phone(raw: str | None) -> str | None:
    """Return E.164 (+1XXXXXXXXXX) for a valid US number, else None."""
    if not raw:
        return None
    digits = re.sub(r"\D", "", str(raw))
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) != 10:
        return None
    if digits[0] in "01" or digits[3] in "01":
        return None
    if len(set(digits)) <= 2:  # 5555555555 etc.
        return None
    return f"+1{digits}"


def is_blocklisted(url: str) -> bool:
    """True when a URL is a directory, social page or search engine."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return True
    if not host:
        return True
    host = host[4:] if host.startswith("www.") else host
    return any(host == d or host.endswith("." + d) for d in BLOCKED_DOMAINS)


def registrable_domain(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def guess_domains(name: str, niche: str = "roofing", limit: int = 4) -> list[str]:
    """Cheap candidate domains for a business, most likely first.

    Used to answer "do they have a website?" without spending a search credit.
    """
    core = name_core(name)
    if not core:
        return []
    words = core.split()
    base = re.sub(r"[^a-z0-9]", "", core)
    if len(base) < 4:
        return []
    joined2 = "".join(words[:2]) if len(words) >= 2 else base
    niche_slug = re.sub(r"[^a-z]", "", niche.lower())[:12] or "roofing"

    variants = [f"{base}.com", f"{joined2}.com"]
    # Only append the niche when the name does not already describe the trade.
    if niche_slug[:4] not in base:
        variants.append(f"{base}{niche_slug}.com")
    if len(words) >= 3:
        variants.append(f"{''.join(words[:3])}.com")

    out: list[str] = []
    for v in variants:
        if v not in out and re.fullmatch(r"[a-z0-9.\-]+", v):
            out.append(v)
    return [f"https://{d}" for d in out[:limit]]


def page_looks_like_listing(url: str, title: str, extracted_count: int) -> bool:
    """Distinguish a directory/listicle page from a single business's own site.

    We only mine candidates out of listing pages: a company's own homepage
    yields fragments ("CALL US TODAY!"), not leads.
    """
    if is_blocklisted(url):
        return True
    # A company's own homepage must not be treated as a directory. Requiring a
    # real list (5+ parsed businesses) is a blunt but effective filter.
    if extracted_count >= 5:
        return True
    low = (title or "").lower()
    listing_signals = (
        "best", "top ", "companies", "contractors", "directory", "list of",
        "near me", "reviews of", "find a", "roofers in", "local",
    )
    return extracted_count >= 3 and sum(1 for s in listing_signals if s in low) >= 2


def find_address(text: str) -> tuple[str, str, str, str] | None:
    """First *valid* US address in ``text`` -> (full, city, state, zip).

    Invalid regex splits are skipped rather than accepted: a mis-parsed address is
    worse than no address, because it silently corrupts the lead's location.
    """
    for m in ADDRESS_RE.finditer(text or ""):
        street, city, state, zip_code = (normalize_ws(g) for g in m.groups())
        street = street.rstrip(" ,;:")
        city = city.strip(" ,;:")
        if len(city) < 3 or STREET_SUFFIX_RE.search(city):
            continue
        if city.lower() in {"united states", "usa", "suite", "ste"}:
            continue
        if not STREET_SUFFIX_RE.search(street) and not re.search(r"\d{2,}", street):
            continue
        if len(street) < 5:
            continue
        return f"{street}, {city}, {state} {zip_code}", city, state, zip_code
    return None


def is_plausible_business_name(name: str) -> bool:
    """Strict gate for anything we treat as a company name.

    Directory pages are full of strings that look like names to a regex but are
    addresses, headings, slogans, image captions or link text. Those poison the
    whole run, so they are rejected as early as possible.
    """
    s = normalize_ws(name or "")
    if not (2 <= len(s) <= 70):
        return False
    if any(ch in s for ch in "[]{}<>|\"\\"):
        return False
    if re.match(r"^step\s*\d", s, re.IGNORECASE):
        return False
    if re.search(r"\.(com|net|org|io|us|co|biz)\b", s, re.IGNORECASE):
        return False
    if re.match(r"^(website|phone|tel|email|address|hours|rating|reviews?|call)\s*:", s, re.IGNORECASE):
        return False
    if re.search(r"\bunited states\b", s, re.IGNORECASE):
        return False
    if CITY_STATE_ZIP_RE.match(s):
        return False
    if ANY_PHONE_IN_NAME_RE.search(s):
        return False
    if s.endswith((",", ";", ":")):
        return False
    if ADDRESS_RE.match(s) or (re.match(r"^\d", s) and STREET_SUFFIX_RE.search(s)):
        return False
    low = s.lower().strip(" .")
    if low in NAME_STOPWORDS or low in UI_NAMES:
        return False
    if LISTICLE_RE.search(s):
        return False
    if SLOGAN_RE.match(s) and (("!" in s) or len(s.split()) <= 4):
        return False

    words = re.findall(r"[A-Za-z][A-Za-z'&.\-]*", s)
    if not words or len(words) > 6:
        return False
    if len(words) == 1:
        only = words[0].lower()
        if only not in BUSINESS_WORDS and len(only) < 4:
            return False
    # Multiple sentences ("North Texas Roots. Here for Your Home").
    if s.count(".") >= 1 and len(words) > 3 and not re.search(
        r"\b(inc|llc|l\.l\.c|co|corp|ltd|jr|sr|st|assoc)\.$", low
    ):
        return False
    # Heading text ("Roofing Contractors in Dallas", "Roofing Near Me").
    if re.search(r"\b(in|near|serving)\s+[A-Za-z]+$", s) and re.search(
        r"\b(contractors?|companies|services?|roofers?|roofing|construction|repair)\b", low
    ):
        return False
    # A slogan: four or more words with no trade vocabulary at all.
    if len(words) >= 4 and not any(w in low for w in BUSINESS_WORDS):
        return False
    # Property/building names, not businesses.
    property_terms = (
        "office building", "apartment", "condo", "townhome", "plaza",
        "shopping center", "strip mall", "warehouse", "facility",
        "medical building", "commercial property", "real estate",
    )
    if any(t in low for t in property_terms):
        return False
    # Service descriptions, not company names.
    service_terms = (
        "roof replacement", "roof repair", "roof installation", "roof inspection",
        "siding repair", "gutter cleaning", "storm damage", "emergency repair",
        "free estimate", "inspection service", "workmanship warranty",
        "owner involved", "sales process", "leak repair", "insurance claim",
        "roofing plus", "windows and", "siding and", "and nearby",
        "in canton", "in longview", "in austin", "in dallas", "nearby towns",
        "our company", "our business", "family owned", "locally owned",
    )
    if any(t in low for t in service_terms):
        return False
    # Comparison/marketplace names.
    marketplace_terms = ("comparison", "directory", "marketplace", "aggregator", "platform")
    if any(t in low for t in marketplace_terms):
        return False
    return True


def normalize_for_match(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — for name matching."""
    t = strip_accents(text or "").lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def page_mentions_business(
    title: str, text: str, name: str, city: str | None = None, phone: str | None = None
) -> tuple[bool, float]:
    """Decide whether a fetched page is really about this business.

    Returns (mentioned, score 0..1). The bar is deliberately high: a page must
    contain the business's full core name, or most of its distinctive tokens
    *plus* a corroborating signal (city or phone). Without this, "T-Rock Roofing"
    matches a random "dallasroofer.com" landing page and the lead is silently
    discarded as "already has a website".
    """
    blob = normalize_for_match(f"{title} \n {text}")
    if not blob:
        return False, 0.0

    core = name_core(name)
    tokens = name_tokens(name)

    phone_digits = re.sub(r"\D", "", phone or "")[-10:]
    phone_match = bool(phone_digits) and phone_digits in re.sub(r"\D", "", text or "")
    city_match = bool(city) and normalize_for_match(city) in blob

    core_match = bool(core) and core in blob

    hits = [t for t in tokens if t in blob]
    coverage = (len(hits) / len(tokens)) if tokens else 0.0

    if core_match:
        score = 0.75
    elif coverage >= 0.75 and (phone_match or city_match):
        score = 0.6
    else:
        return False, 0.0

    if city_match:
        score += 0.1
    if phone_match:
        score += 0.15
    if coverage >= 0.99 and tokens:
        score += 0.05

    return True, round(min(1.0, score), 2)


def looks_parked(title: str, text: str) -> bool:
    blob = f"{title}\n{text[:1500]}".lower()
    return any(h in blob for h in PARKED_HINTS)


def parse_rating_reviews(text: str) -> tuple[float | None, int | None]:
    """Pull a rating and review count out of free text."""
    best: tuple[float | None, int | None] = (None, None)
    for m in RATING_RE.finditer(text or ""):
        try:
            rating = float(m.group(1))
            reviews = int(m.group(2).replace(",", ""))
        except ValueError:
            continue
        if not 0.0 <= rating <= 5.0:
            continue
        if best[1] is None or reviews > best[1]:
            best = (rating, reviews)
    return best


# --------------------------------------------------------------------------- #
# Name detection
# --------------------------------------------------------------------------- #


def _is_nameish(line: str) -> bool:
    s = normalize_ws(line).strip()
    if not (2 <= len(s) <= 90):
        return False
    low = s.lower().strip(" .:•-|")
    if low in NAME_STOPWORDS:
        return False
    if "%" in s:
        return False
    if low.startswith(("http", "www.", "tel:", "mailto:")):
        return False
    if "@" in s or "http" in low:
        return False
    # A street address is a place, never a company name.
    if ADDRESS_RE.match(s) or (re.match(r"^\d", s) and STREET_SUFFIX_RE.search(s)):
        return False
    if re.match(r"^\d", s) and not re.search(r"[A-Za-z]{3,}", s):
        return False
    # Listicle headings ("15 Best Roofing Contractors in Dallas").
    if LISTICLE_RE.search(s):
        return False
    # Advertising calls to action.
    if SLOGAN_RE.match(s) and (("!" in s) or len(s.split()) <= 4):
        return False
    if low.startswith(("write a review", "read ", "more ", "see ", "view ")):
        return False
    if PHONE_RE.search(s) and len(re.sub(r"\d|\W", "", s)) < 4:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'&.\-]*", s)
    if len(words) > 8:  # a sentence, not a name
        return False
    if len(words) < 2 and not any(w in BUSINESS_WORDS for w in (w.lower() for w in words)):
        return False
    if re.search(r"\b(reviews?|ratings?|stars?|hours|open now|closed)\b", low):
        return False
    # Image alt-text fragments ("DTX Construction & Roofing logo").
    if re.search(r"\b(logo|icon|photo|image|banner|headshot|avatar)\s*$", low):
        return False
    return is_plausible_business_name(s)


def _score_name(line: str, distance: int) -> float:
    """Higher is better. Closer lines and business keywords win."""
    score = 10.0 - distance * 1.5
    low = line.lower()
    score += sum(1.6 for w in BUSINESS_WORDS if w in low)
    if line.strip().startswith(("**", "###", "#")) or line.strip().endswith("**"):
        score += 2.0
    if any(c.isupper() for c in line):  # Title Case / CAPS business names
        score += 0.4
    if len(line) > 70:
        score -= 1.5
    return score


def extract_candidates(
    text: str,
    *,
    source_url: str = "",
    source: str = "text",
    default_city: str | None = None,
    default_state: str | None = None,
    default_zip: str | None = None,
    expected_state: str | None = None,
    max_records: int = 25,
) -> list[Candidate]:
    """Parse business records out of a directory page or SERP snippet blob.

    Strategy: anchor on phone numbers (the least ambiguous signal), then look
    backwards for the closest plausible name and forwards for an address.
    Listings without a phone are recovered via ``NAME/address`` line pairs.
    """
    if not text or not text.strip():
        return []

    text = percent_decode(text)
    lines = [normalize_ws(ln) for ln in text.splitlines()]
    lines = [ln for ln in lines if ln not in {"", "-", "|", "·"}]
    joined = "\n".join(lines)

    found: dict[str, Candidate] = {}

    def add(cand: Candidate) -> None:
        if not is_plausible_business_name(cand.name):
            return
        key = dedupe_key(cand.name, cand.city or default_city, cand.zip or default_zip)
        if not key:
            return
        existing = found.get(key)
        if existing is None:
            found[key] = cand
            return
        # Merge: keep the richer record.
        for field in ("phone", "address", "city", "state", "zip", "rating", "reviews", "category"):
            if getattr(existing, field) in (None, "", 0) and getattr(cand, field) not in (None, "", 0):
                setattr(existing, field, getattr(cand, field))
        if len(cand.evidence) > len(existing.evidence):
            existing.evidence = cand.evidence

    # ---- pass 1: phone-anchored
    for idx, line in enumerate(lines):
        for pm in PHONE_RE.finditer(line):
            phone = canonical_phone(pm.group(0))
            if not phone:
                continue

            # Name: same line before the phone, else the closest preceding line.
            name = ""
            prefix = line[: pm.start()].strip(" -|,;:·—–")
            if len(prefix) >= 3 and _is_nameish(prefix):
                name = prefix
            if not name:
                for back in range(1, 5):
                    j = idx - back
                    if j < 0:
                        break
                    if _is_nameish(lines[j]):
                        name = lines[j]
                        break
            if not name:
                continue

            name = clean_name(name)
            if not name or not _is_nameish(name):
                continue

            window = "\n".join(lines[max(0, idx - 3): idx + 5])
            city = default_city
            state = default_state
            zip_code = default_zip
            address = None
            parsed = find_address(window)
            if parsed:
                address, city, state, zip_code = parsed
                # If the parsed state doesn't match the expected state (from the ZIP),
                # the page is likely listing businesses from multiple states.
                # Fall back to the default_state to avoid cross-state pollution.
                if expected_state and state.upper() != expected_state.upper():
                    state = default_state
                    city = default_city
                    zip_code = default_zip

            rating, reviews = parse_rating_reviews(window)
            add(
                Candidate(
                    name=name,
                    city=city,
                    state=state,
                    zip=zip_code,
                    phone=phone,
                    address=address,
                    rating=rating,
                    reviews=reviews,
                    source=source,
                    source_url=source_url,
                    evidence=normalize_ws(window)[:400],
                )
            )

    # ---- pass 2: heading + address blocks (no phone present)
    if len(found) < max_records:
        for idx, line in enumerate(lines):
            heading = line.strip()
            is_heading = heading.startswith(("###", "##", "**")) or (
                _is_nameish(heading) and len(heading) < 70
            )
            if not is_heading or not _is_nameish(clean_name(heading)):
                continue
            window = "\n".join(lines[idx: idx + 6])
            parsed = find_address(window)
            if not parsed:
                continue
            address, city, state, zip_code = parsed
            # Validate parsed state against expected state (from the ZIP).
            if expected_state and state.upper() != expected_state.upper():
                state = default_state
                city = default_city
                zip_code = default_zip
            name = clean_name(heading)
            if not is_plausible_business_name(name):
                continue
            rating, reviews = parse_rating_reviews(window)
            pm = PHONE_RE.search(window)
            add(
                Candidate(
                    name=name,
                    city=city,
                    state=state,
                    zip=zip_code,
                    phone=canonical_phone(pm.group(0)) if pm else None,
                    address=address,
                    rating=rating,
                    reviews=reviews,
                    source=source,
                    source_url=source_url,
                    evidence=normalize_ws(window)[:400],
                )
            )

    # Discard obvious non-businesses and the source page itself.
    out: list[Candidate] = []
    for cand in found.values():
        low = cand.name.lower()
        if any(w in low for w in ("privacy policy", "terms of", "cookie", "all rights reserved")):
            continue
        if cand.name.lower() in {"home", "about", "contact", "services", "reviews"}:
            continue
        out.append(cand)

    out.sort(key=lambda c: (c.phone is None, c.address is None, c.name))
    return out[:max_records]


def extract_city_state_zip(text: str) -> tuple[str | None, str | None, str | None]:
    """Best-effort city/state/zip from a free-form address or snippet."""
    if not text:
        return None, None, None
    m = ADDRESS_RE.search(text)
    if m:
        return normalize_ws(m.group(2)), m.group(3), m.group(4)
    m2 = re.search(r"([A-Za-z][A-Za-z .'\-]{1,25}),\s*([A-Z]{2})\s*(\d{5})?", text)
    if m2:
        return normalize_ws(m2.group(1)), m2.group(2), m2.group(3)
    return None, None, None
