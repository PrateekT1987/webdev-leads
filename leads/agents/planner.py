"""Planner agent.

Turns a campaign brief into an executable :class:`RunPlan` — the query set, the
ICP and the source order. The LLM proposes; ``RunPlan`` validation disposes. If
the model is unavailable or returns something unusable, a deterministic plan is
used, so a run never fails just because the planner did.
"""

from __future__ import annotations

from ..errors import JSONParseError, LLMUnavailable
from ..llm import prompts
from ..models import Campaign, ICP, RunPlan
from .base import BaseAgent

DEFAULT_QUERIES: tuple[str, ...] = (
    "best {niche} in {city} {state}",
    "{niche} companies {city} {state}",
    "top rated {niche} {city} {state}",
    "{niche} near {zip}",
    "list of {niche} in {city}",
)


def default_plan(campaign: Campaign) -> RunPlan:
    """Deterministic fallback plan. Kept boring on purpose."""
    queries = [q.replace("{niche}", campaign.niche.lower()) for q in DEFAULT_QUERIES]
    return RunPlan(
        queries=queries,
        icp=campaign.icp,
        source_priority=["overpass", "tavily"],
        notes="deterministic default plan",
        llm_generated=False,
    )


class PlannerAgent(BaseAgent):
    name = "planner"

    def build_plan(self, campaign: Campaign, sample_cities: list[str]) -> RunPlan:
        fallback = default_plan(campaign)

        if not self.llm_available:
            self.event("info", "planner: LLM unavailable, using deterministic plan")
            return fallback

        try:
            data = self.ask_llm_json(
                system=prompts.PLANNER_SYSTEM,
                user=prompts.planner_user(
                    campaign.niche, campaign.state, sample_cities, campaign.icp
                ),
                max_tokens=700,
            )
        except (JSONParseError, LLMUnavailable) as exc:
            self.event("warning", f"planner: LLM plan rejected ({exc}); using default")
            return fallback
        except Exception as exc:  # network/budget — never fatal for planning
            self.event("warning", f"planner: LLM error ({exc}); using default")
            return fallback

        raw_queries = data.get("queries") or []
        queries: list[str] = []
        for q in raw_queries:
            if not isinstance(q, str):
                continue
            q = " ".join(q.split())
            if "{city}" in q or "{zip}" in q:
                queries.append(q)

        if not queries:
            self.event("warning", "planner: no usable queries in LLM plan; using default")
            return fallback

        icp_data = dict(campaign.icp.model_dump())
        traits = data.get("target_traits")
        if isinstance(traits, list) and traits:
            icp_data["target_traits"] = [str(t)[:120] for t in traits[:4]]
        try:
            icp = ICP(**icp_data)
        except Exception:
            icp = campaign.icp

        try:
            plan = RunPlan(
                queries=queries[:6],
                icp=icp,
                source_priority=["overpass", "tavily"],
                notes=str(data.get("notes", ""))[:200],
                llm_generated=True,
            )
        except Exception as exc:
            self.event("warning", f"planner: plan validation failed ({exc}); using default")
            return fallback

        self.event(
            "info",
            "planner: plan ready",
            {"queries": plan.queries, "llm": True, "notes": plan.notes},
        )
        return plan
