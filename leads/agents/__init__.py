"""Agent registry.

The pipeline is a supervisor plus six specialists:

    planner -> discovery -> verifier -> qualifier -> critic -> reporter

Each agent has one job, one output type, and a deterministic fallback. The
supervisor (``orchestrator.py``) decides order, concurrency and retries; agents
never call each other.
"""

from .base import Agent, AgentContext, BaseAgent
from .critic import CriticAgent
from .discovery import DiscoveryAgent
from .planner import PlannerAgent
from .qualifier import QualifierAgent
from .reporter import ReporterAgent
from .verifier import VerifierAgent

AGENT_GRAPH: dict[str, str] = {
    "planner": "LLM + rules — turns the campaign brief into queries and an ICP",
    "discovery": "OSM + Tavily + rules — finds candidate businesses per ZIP/city",
    "verifier": "probes + Tavily + LLM — decides whether a business owns a website",
    "qualifier": "rules + LLM — gates on rating/reviews, scores fit, assigns tier",
    "critic": "rules + LLM — independent audit; can reject a qualified lead",
    "reporter": "LLM + template — writes the run brief",
}

__all__ = [
    "Agent",
    "AgentContext",
    "BaseAgent",
    "PlannerAgent",
    "DiscoveryAgent",
    "VerifierAgent",
    "QualifierAgent",
    "CriticAgent",
    "ReporterAgent",
    "AGENT_GRAPH",
]
