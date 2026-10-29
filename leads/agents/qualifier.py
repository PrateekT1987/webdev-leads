"""Qualifier agent: score a verified website-less business.

Hard numeric gates are enforced in Python (rating/review thresholds, obvious
wrong business kinds), because a model should not be trusted to do arithmetic on
data that decides whether a salesperson calls someone. The LLM only judges the
qualitative question — is this the kind of business that would buy a website?

When the LLM is unavailable the deterministic score still produces a usable
tier, so the pipeline degrades instead of stopping.
"""

from __future__ import annotations

from ..errors import JSONParseError, LLMUnavailable
from ..llm import prompts
from ..models import ICP, Qualification, Tier
from ..tools.extract import is_plausible_business_name
from .base import BaseAgent

# Categories that are almost never the buyer we want.
NON_CONTRACTOR_HINTS = (
    "supply", "supplies", "supplier", "materials", "manufactur", "distribut",
    "wholesale", "retail", "store", "hardware", "lumber", "rental", "equipment",
    "directory", "marketplace", "association", "school", "training",
)


class QualifierAgent(BaseAgent):
    name = "qualifier"

    # --------------------------------------------------------------- gates
    def prefilter(
        self, *, name: str, category: str | None, rating: float | None,
        reviews: int | None, icp: ICP,
    ) -> Qualification | None:
        """Deterministic rejection. Returns None when the lead may proceed."""
        if not is_plausible_business_name(name):
            return Qualification(
                fit=0, tier=Tier.REJECT,
                reason="not a plausible business name",
                signals={"gate": "name"},
            )
        cat = (category or "").lower()
        if cat and any(hint in cat for hint in NON_CONTRACTOR_HINTS):
            return Qualification(
                fit=10, tier=Tier.REJECT,
                reason=f"not a contractor (category: {category})",
                signals={"gate": "category"},
            )
        if rating is not None and rating < icp.min_rating:
            return Qualification(
                fit=15, tier=Tier.REJECT,
                reason=f"rating {rating} below minimum {icp.min_rating}",
                signals={"gate": "rating"},
            )
        if reviews is not None and reviews < icp.min_reviews:
            return Qualification(
                fit=20, tier=Tier.REJECT,
                reason=f"{reviews} reviews below minimum {icp.min_reviews}",
                signals={"gate": "reviews"},
            )
        return None

    # -------------------------------------------------------------- scoring
    def _deterministic_score(
        self, *, rating: float | None, reviews: int | None, has_phone: bool,
        has_address: bool, category: str | None,
    ) -> Qualification:
        score = 45
        signals: dict[str, object] = {}

        if has_phone:
            score += 12
            signals["phone"] = True
        if has_address:
            score += 8
            signals["address"] = True
        if rating is not None:
            if rating >= 4.8:
                score += 15
            elif rating >= 4.5:
                score += 10
            signals["rating"] = rating
        if reviews is not None:
            if reviews >= 100:
                score += 15
            elif reviews >= 30:
                score += 10
            elif reviews >= 10:
                score += 5
            signals["reviews"] = reviews
        if category and "roof" in category.lower():
            score += 10
            signals["category_match"] = True

        score = max(0, min(100, score))
        tier = Tier.A if score >= 80 else Tier.B if score >= 60 else Tier.C
        return Qualification(
            fit=score,
            tier=tier,
            reason="deterministic score (LLM unavailable)",
            signals=signals,
            llm_generated=False,
        )

    # ------------------------------------------------------------------ run
    def qualify(
        self,
        *,
        name: str,
        category: str | None,
        city: str | None,
        state: str | None,
        rating: float | None,
        reviews: int | None,
        has_phone: bool,
        has_address: bool,
        icp: ICP,
    ) -> Qualification:
        gate = self.prefilter(
            name=name, category=category, rating=rating, reviews=reviews, icp=icp
        )
        if gate is not None:
            return gate

        deterministic = self._deterministic_score(
            rating=rating, reviews=reviews, has_phone=has_phone,
            has_address=has_address, category=category,
        )

        if not self.llm_available:
            return deterministic

        self.ctx.bump("llm_qualify_calls")
        try:
            data = self.ask_llm_json(
                system=prompts.QUALIFIER_SYSTEM,
                user=prompts.qualifier_user(
                    name=name, category=category, city=city, state=state,
                    rating=rating, reviews=reviews, has_phone=has_phone,
                    has_address=has_address, niche=icp.niche,
                ),
                max_tokens=800,
            )
        except (LLMUnavailable, JSONParseError) as exc:
            self.event("warning", f"qualifier: LLM unavailable ({exc}); using deterministic score")
            return deterministic
        except Exception as exc:
            self.event("warning", f"qualifier: LLM error ({exc}); using deterministic score")
            return deterministic

        fit = data.get("fit")
        try:
            fit_int = max(0, min(100, int(float(fit))))
        except (TypeError, ValueError):
            self.event("warning", f"qualifier: unusable fit value {fit!r}; using deterministic")
            return deterministic

        tier = Tier.parse(data.get("tier"), default=Tier.C)
        # Keep tier and score consistent — a model can otherwise say "A" for 30.
        expected = Tier.A if fit_int >= 80 else Tier.B if fit_int >= 60 else Tier.C
        if fit_int < 40:
            expected = Tier.REJECT
        if tier != expected:
            tier = expected

        return Qualification(
            fit=fit_int,
            tier=tier,
            reason=str(data.get("reason", ""))[:200],
            signals={"deterministic_fit": deterministic.fit, **deterministic.signals},
            llm_generated=True,
        )
