"""Critic agent: independent audit before a lead is exported.

Two layers:

* **Deterministic checks** that are always correct and always run: address-shaped
  names, UI text, out-of-state addresses, invalid phones, and a website verdict
  that contradicts itself.
* **LLM audit** for the judgment calls: wrong business kind, fabricated-looking
  fields, contradictory evidence.

The critic is a separate agent from the qualifier on purpose — one agent scoring
its own work is how bad leads get through. Rejections are recorded, never
silently dropped, so the operator can inspect what was filtered and why.
"""

from __future__ import annotations

from ..errors import BudgetExceeded, JSONParseError, LLMUnavailable
from ..llm import prompts
from ..tools.extract import (
    ADDRESS_RE,
    NAME_STOPWORDS,
    UI_NAMES,
    canonical_phone,
    clean_name,
    is_plausible_business_name,
)
from .base import BaseAgent

UI_NAMES_LEGACY: set[str] = set()


class CriticAgent(BaseAgent):
    name = "critic"

    # -------------------------------------------------------- deterministic
    def static_checks(
        self,
        *,
        name: str,
        address: str | None,
        city: str | None,
        state: str | None,
        phone: str | None,
        has_website: bool,
        campaign_state: str,
    ) -> list[str]:
        issues: list[str] = []

        cleaned = clean_name(name)
        if cleaned != name:
            issues.append(f"malformed name ({name!r} -> {cleaned!r})")

        low = name.lower().strip()
        if low in UI_NAMES or low in NAME_STOPWORDS:
            issues.append("name is page UI text, not a business")

        if not is_plausible_business_name(name):
            issues.append("name is not a plausible business name")

        if ADDRESS_RE.match(name):
            issues.append("name is a street address")

        if phone and canonical_phone(phone) is None:
            issues.append("invalid US phone number")

        blob = f"{address or ''} {state or ''}"
        if state and len(state) == 2 and state.upper() != campaign_state.upper():
            issues.append(f"state {state.upper()} outside campaign state {campaign_state}")

        if has_website:
            issues.append("lead has a website; should not be in the export")

        if not city and not address:
            issues.append("no location evidence")

        return issues

    # ------------------------------------------------------------------ run
    def review(
        self,
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
    ) -> tuple[bool, list[str], str]:
        """Return (accepted, issues, rationale). Never raises."""
        issues = self.static_checks(
            name=name, address=address, city=city, state=state, phone=phone,
            has_website=has_website, campaign_state=campaign_state,
        )

        # Hard rejects need no model opinion.
        hard = [
            i for i in issues
            if i.startswith(
                (
                    "name is",
                    "state ",
                    "lead has a website",
                    "invalid US phone",
                )
            )
        ]
        if hard:
            return False, issues, "rejected by deterministic checks: " + "; ".join(hard)

        if not self.llm_available:
            return (not issues), issues, "deterministic checks only (LLM unavailable)"

        self.ctx.bump("llm_critic_calls")
        try:
            data = self.ask_llm_json(
                system=prompts.CRITIC_SYSTEM,
                user=prompts.critic_user(
                    name=name, city=city, state=state, phone=phone, address=address,
                    category=category, has_website=has_website,
                    website_evidence=website_evidence, fit=fit, tier=tier,
                    campaign_state=campaign_state, niche=niche,
                ),
                max_tokens=900,
            )
        except (LLMUnavailable, JSONParseError) as exc:
            self.event("warning", f"critic: LLM unavailable ({exc}); deterministic only")
            return (not issues), issues, "deterministic checks only (LLM unavailable)"
        except BudgetExceeded as exc:
            self.event("warning", f"critic: budget reached ({exc}); deterministic only")
            return (not issues), issues, "deterministic checks only (budget reached)"
        except Exception as exc:
            self.event("warning", f"critic: LLM error ({exc}); deterministic only")
            return (not issues), issues, "deterministic checks only (LLM error)"

        accepted = bool(data.get("accepted", True))
        llm_issues = [str(i)[:160] for i in (data.get("issues") or []) if i]
        all_issues = issues + llm_issues
        rationale = str(data.get("rationale", ""))[:240]

        if not accepted and not llm_issues:
            llm_issues.append("rejected by critic without stated reason")
            all_issues = issues + llm_issues

        return accepted, all_issues, rationale or "LLM audit passed"
