"""Reporter agent: turns a finished run into a short, factual brief.

The model writes prose; the numbers come from SQLite. If the model is down, a
deterministic Markdown report is produced instead — a run always ends with an
artifact the operator can read.
"""

from __future__ import annotations

from ..llm import prompts
from .base import BaseAgent

TEMPLATE = """# Lead run {run_id}

**Niche:** {niche} · **State:** {state} · **Finished:** {finished}

## Summary

{summary}

## Top leads

{leads}

## Data gaps

{gaps}

## Recommended next steps

1. Call the tier-A leads first; they have no website and enough signal to qualify.
2. Confirm website status by phone before pitching — the automated verdict is
   evidence-based, not a guarantee.
3. Re-run the ZIPs with zero results against a second discovery source.
"""


class ReporterAgent(BaseAgent):
    name = "reporter"

    def build_report(
        self,
        *,
        run_id: str,
        niche: str,
        state: str,
        finished: str,
        stats: dict,
        top_leads: list[dict],
    ) -> str:
        if self.llm_available:
            try:
                self.ctx.bump("llm_report_calls")
                text = self.ctx.llm.chat_text(
                    system=prompts.REPORTER_SYSTEM,
                    user=prompts.reporter_user(stats, top_leads),
                    agent=self.name,
                    max_tokens=1200,
                )
                if text and "##" in text:
                    return f"# Lead run {run_id}\n\n{text.strip()}\n"
                self.event("warning", "reporter: LLM output lacked structure; using template")
            except Exception as exc:
                self.event("warning", f"reporter: LLM unavailable ({exc}); using template")

        return self._template(
            run_id=run_id, niche=niche, state=state, finished=finished,
            stats=stats, top_leads=top_leads,
        )

    # ------------------------------------------------------------- fallback
    def _template(
        self, *, run_id: str, niche: str, state: str, finished: str,
        stats: dict, top_leads: list[dict],
    ) -> str:
        summary = (
            f"Searched {stats.get('cells_done', 0)} of {stats.get('cells_total', 0)} ZIP/city "
            f"cells and found {stats.get('candidates', 0)} candidate businesses. "
            f"{stats.get('without_website', 0)} had no website, "
            f"{stats.get('with_website', 0)} did. "
            f"{stats.get('qualified', 0)} passed qualification "
            f"({stats.get('tier_a', 0)} tier A, {stats.get('tier_b', 0)} tier B)."
        )

        if top_leads:
            lines = []
            for lead in top_leads[:8]:
                phone = lead.get("phone") or "no phone"
                lines.append(
                    f"- **{lead.get('name', '?')}** — {lead.get('city') or '?'}, "
                    f"{lead.get('state') or '?'} · {phone} · tier {lead.get('tier', '?')} "
                    f"(fit {lead.get('fit', '?')})"
                )
            leads_md = "\n".join(lines)
        else:
            leads_md = "_No qualified leads in this run._"

        gaps = []
        if stats.get("ambiguous"):
            gaps.append(f"- {stats['ambiguous']} leads had an inconclusive website verdict.")
        if not stats.get("with_website") and not stats.get("without_website"):
            gaps.append("- No website verdicts were produced; check discovery sources.")
        gaps.append("- Ratings and review counts are unavailable from most directory pages.")
        gaps_md = "\n".join(gaps)

        return TEMPLATE.format(
            run_id=run_id, niche=niche, state=state, finished=finished,
            summary=summary, leads=leads_md, gaps=gaps_md,
        )
