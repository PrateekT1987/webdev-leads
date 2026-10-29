"""Supervisor: runs the agent graph, owns retries, concurrency and checkpoints.

    plan ──▶ discover ──▶ verify ──▶ qualify ──▶ critic ──▶ report ──▶ export

Design guarantees:

* **Checkpointed.** Every cell and every stage is a row in ``tasks``. ``--resume``
  skips completed work instead of paying for it twice.
* **Budgeted.** LLM calls and Tavily credits have hard caps; hitting one stops
  new spending but still produces a report from what was gathered.
* **Degradable.** With no LLM key every agent falls back to deterministic rules,
  so the pipeline still produces leads.
* **Bounded.** Thread pools sized by config; a single cell failure never kills
  the run.
* **Auditable.** Every candidate keeps its source URL and evidence; rejections
  keep their reason.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import export as export_mod
from .agents import (
    AgentContext,
    CriticAgent,
    DiscoveryAgent,
    PlannerAgent,
    QualifierAgent,
    ReporterAgent,
    VerifierAgent,
)
from .config import Settings, get_settings
from .errors import BudgetExceeded, LeadsError, ProviderError, RateLimited
from .llm.client import LLMBudget, LLMClient
from .logging_setup import configure_logging
from .models import (
    Campaign,
    Cell,
    CriticReview,
    LeadStatus,
    Qualification,
    RunPlan,
    RunStats,
    Tier,
    utcnow,
)
from .store import Database, Store
from .tools.cache import HttpCache
from .tools.overpass import OverpassClient
from .tools.tavily import TavilyClient
from .tools.webprobe import WebProbe
from .zipcodes import ZipIndex, state_from_zip

log = logging.getLogger(__name__)

ZIP_SOURCE_FILES = (
    "us_zip_codes.txt",
    "zipcodes/us_zip_codes.txt",
)

ZIP_DATASETS = (
    "simplemaps_uszips_basicv1.95.1/uszips.csv",
    "USZipsWithLatLon_20231227.csv",
    "zipcodes/USZipsWithLatLon_20231227.csv",
)


@dataclass
class RunOptions:
    use_osm: bool = True
    verify_search: bool = True
    critic_sample: float = 1.0
    include_review: bool = False
    resume: bool = False
    export_formats: tuple[str, ...] = ("xlsx", "csv")
    max_cells_in_parallel: int | None = None


@dataclass
class RunResult:
    run_id: str
    stats: dict[str, Any]
    report_path: Path | None = None
    export_paths: list[Path] = field(default_factory=list)
    leads: list[dict[str, Any]] = field(default_factory=list)
    status: str = "completed"


class Orchestrator:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        run_id: str | None = None,
        options: RunOptions | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.options = options or RunOptions()
        self.settings.ensure_dirs()
        configure_logging(self.settings.log_level, self.settings.logs_dir)

        self.run_id = run_id or time.strftime("run-%Y%m%d-%H%M%S")
        self.db = Database(self.settings.db_path)
        self.store = Store(self.db)
        self.cache = HttpCache(self.store, self.settings.http_cache_ttl_hours)

        self.tavily = (
            TavilyClient(
                self.settings.tavily_api_key,
                cache=self.cache,
                timeout=self.settings.request_timeout,
                user_agent=self.settings.user_agent,
                credit_budget=self.settings.max_tavily_credits,
            )
            if self.settings.tavily_ready
            else None
        )
        self.overpass = OverpassClient(
            cache=self.cache,
            timeout=max(30.0, self.settings.request_timeout),
            user_agent=self.settings.user_agent,
        )
        self.probe = WebProbe(
            timeout=min(15.0, self.settings.request_timeout),
            user_agent=self.settings.user_agent,
        )

        self.budget = LLMBudget(max_calls=self.settings.max_llm_calls)
        self.llm = LLMClient(
            api_key=self.settings.openrouter_api_key,
            model=self.settings.model,
            base_url=self.settings.openrouter_base_url,
            timeout=self.settings.llm_timeout,
            max_tokens=self.settings.llm_max_tokens,
            budget=self.budget,
            telemetry=self._llm_telemetry,
            failure_threshold=self.settings.llm_max_consecutive_failures,
            enabled=self.settings.llm_ready,
        )

        self.ctx = AgentContext(
            run_id=self.run_id,
            settings=self.settings,
            store=self.store,
            llm=self.llm,
            tavily=self.tavily,
            overpass=self.overpass,
            probe=self.probe,
            cache=self.cache,
        )

        self._stop = threading.Event()
        self._progress_lock = threading.Lock()
        self._done_counter = 0

    # ----------------------------------------------------------- telemetry
    def _llm_telemetry(self, **kwargs: Any) -> None:
        try:
            self.store.record_llm_call(self.run_id, **kwargs)
        except Exception:  # telemetry must never break a run
            log.debug("failed to record llm telemetry", exc_info=True)

    # ---------------------------------------------------------------- public
    def run(self, campaign: Campaign) -> RunResult:
        from .logging_setup import run_id_var

        run_id_var.set(self.run_id)
        self.store.create_run(
            self.run_id, campaign.niche, campaign.state, campaign.model_dump()
        )
        self.store.log_event(
            self.run_id, "info", "orchestrator", "run started",
            {"niche": campaign.niche, "state": campaign.state, "options": self.options.__dict__},
        )
        started = time.monotonic()

        try:
            cells = self._resolve_cells(campaign)
            if not cells:
                return self._finish(
                    campaign, RunStats(), started, status="failed",
                    note=f"no ZIP codes found for {campaign.state}",
                )

            campaign = campaign.model_copy(update={"max_zips": len(cells)})
            plan = self._plan(campaign, cells)
            self._discover(cells, plan, campaign)
            self._verify(campaign)
            self._qualify_and_critic(campaign)
            return self._finish(campaign, started=started, status="completed")

        except BudgetExceeded as exc:
            self.store.log_event(self.run_id, "error", "orchestrator", str(exc))
            return self._finish(campaign, started=started, status="budget_exhausted")
        except KeyboardInterrupt:
            self.store.log_event(self.run_id, "warning", "orchestrator", "interrupted by user")
            self._stop.set()
            return self._finish(campaign, started=started, status="interrupted")
        except LeadsError as exc:
            self.store.log_event(self.run_id, "error", "orchestrator", str(exc))
            return self._finish(campaign, started=started, status="failed")
        except Exception as exc:
            self.store.log_event(
                self.run_id, "error", "orchestrator", f"unexpected error: {exc}"
            )
            log.exception("run failed")
            return self._finish(campaign, started=started, status="failed")

    # ----------------------------------------------------------------- cells
    def _resolve_cells(self, campaign: Campaign) -> list[Cell]:
        index = ZipIndex.from_candidates(
            [self.settings.home.parent / p for p in ZIP_DATASETS]
            + [Path(p) for p in ZIP_DATASETS]
        )
        log.info("ZIP index ready: %d codes", len(index))

        zips = list(campaign.zips)

        if not zips:
            # Fall back to a ZIP list file that matches the campaign state.
            for candidate in ZIP_SOURCE_FILES:
                path = Path(candidate)
                if not path.exists():
                    continue
                raw = [
                    ln.strip()
                    for ln in path.read_text(encoding="utf-8").splitlines()
                    if ln.strip().isdigit()
                ]
                matching = [
                    z.zfill(5) for z in raw
                    if state_from_zip(z.zfill(5)) == campaign.state
                ]
                if matching:
                    log.info("using %d ZIPs for %s from %s", len(matching), campaign.state, path)
                    zips = matching
                    break

        if not zips:
            # Last resort: derive from the ZIP index.
            all_zips = sorted(
                z for z, loc in index._data.items() if loc.state == campaign.state
            )
            zips = all_zips[: campaign.max_zips]

        zips = zips[: campaign.max_zips]
        cells: list[Cell] = []
        seen: set[str] = set()
        for z in zips:
            if z in seen:
                continue
            seen.add(z)
            loc = index.lookup(z)
            cells.append(
                Cell(
                    zip=z,
                    city=loc.city,
                    state=loc.state or campaign.state,
                    lat=loc.lat,
                    lon=loc.lon,
                )
            )
        return cells

    # ------------------------------------------------------------------ plan
    def _plan(self, campaign: Campaign, cells: list[Cell]) -> RunPlan:
        planner = PlannerAgent(self.ctx)
        cities = [c.city for c in cells if c.city]
        plan = planner.build_plan(campaign, cities)
        self.store.log_event(
            self.run_id, "info", "planner", "plan selected",
            {"queries": plan.queries, "llm_generated": plan.llm_generated},
        )
        return plan

    # -------------------------------------------------------------- discover
    def _discover(self, cells: list[Cell], plan: RunPlan, campaign: Campaign) -> None:
        discovery = DiscoveryAgent(self.ctx)
        for cell in cells:
            self.store.seed_task(self.run_id, "discover", cell.key)

        pending = [
            c for c in cells
            if not (self.options.resume and self.store.task_state(self.run_id, "discover", c.key) == "done")
        ]
        if not pending:
            log.info("discovery: all cells already complete (resume)")
            return

        workers = min(
            self.options.max_cells_in_parallel or self.settings.concurrency,
            max(1, len(pending)),
        )
        log.info("discovery: %d cells, %d workers", len(pending), workers)

        def work(cell: Cell) -> tuple[Cell, int]:
            if self._stop.is_set():
                return cell, 0
            self.store.start_task(self.run_id, "discover", cell.key)
            try:
                found = discovery.discover(cell, plan, use_osm=self.options.use_osm)
                for cand in found:
                    self.store.upsert_candidate(self.run_id, cand)
                self.store.complete_task(self.run_id, "discover", cell.key)
                return cell, len(found)
            except Exception as exc:
                self.store.fail_task(self.run_id, "discover", cell.key, str(exc))
                log.warning("discovery failed for %s: %s", cell.label, exc)
                return cell, 0

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(work, c): c for c in pending}
            for fut in as_completed(futures):
                try:
                    cell, count = fut.result()
                except Exception as exc:
                    log.warning("discovery worker crashed: %s", exc)
                    continue
                self._tick(f"discovery {cell.label}: +{count}")

        # Post-discovery filter: drop candidates outside campaign state.
        # Directory pages often list businesses from neighboring states; we
        # reject them before spending Tavily credits on verification.
        self._filter_by_state(campaign.state)

    def _filter_by_state(self, campaign_state: str) -> None:
        """Remove candidates whose state doesn't match the campaign (or is unknown)."""
        rows = self.store.candidates(self.run_id)
        dropped = 0
        for row in rows:
            state = (row.get("state") or "").upper()
            if state and state != campaign_state.upper():
                self.store.suppress(
                    row.get("dedupe_key", f"{row['name']}|{row['zip']}"),
                    row.get("phone"),
                    f"state {state} outside campaign state {campaign_state}",
                )
                self.store.upsert_lead(
                    self.run_id, row["id"], LeadStatus.REJECTED, None,
                    CriticReview(accepted=False, issues=[f"state {state} outside campaign state {campaign_state}"]),
                )
                dropped += 1
        if dropped:
            log.info("discovery: filtered %d out-of-state candidates", dropped)

    # ----------------------------------------------------------------- verify
    def _verify(self, campaign: Campaign) -> None:
        rows = self.store.candidates_pending_verdict(self.run_id)
        if not rows:
            log.info("verify: no candidates to verify")
            return

        verifier = VerifierAgent(self.ctx, use_search=self.options.verify_search)
        workers = min(self.settings.concurrency, max(1, len(rows)))
        log.info("verify: %d candidates, %d workers", len(rows), workers)

        def work(row: dict[str, Any]) -> None:
            if self._stop.is_set():
                return
            try:
                verdict = verifier.verify(
                    name=row["name"],
                    city=row["city"],
                    state=row["state"],
                    phone=row["phone"],
                    known_website=row.get("website"),
                    niche=campaign.niche,
                )
            except BudgetExceeded:
                self._stop.set()
                return
            except (ProviderError, RateLimited) as exc:
                log.warning("verify: provider issue for %s: %s", row["name"], exc)
                from .models import VerdictMethod, WebsiteVerdict

                verdict = WebsiteVerdict(
                    has_website=False, confidence=0.2, method=VerdictMethod.NONE,
                    evidence=f"verification skipped: {exc}", ambiguous=True,
                )
            except Exception as exc:
                log.warning("verify: failed for %s: %s", row["name"], exc)
                return

            self.store.set_verdict(row["id"], verdict)

            if verdict.has_website:
                status = LeadStatus.WEBSITE_FOUND
            elif verdict.ambiguous:
                status = LeadStatus.NEEDS_REVIEW
            else:
                status = LeadStatus.NO_WEBSITE
            self.store.upsert_lead(self.run_id, row["id"], status)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(work, r) for r in rows]
            for _ in as_completed(futures):
                self._tick("verification")

        self._dedupe_by_domain()

    def _dedupe_by_domain(self) -> None:
        """Collapse candidates that share one verified website domain.

        Two different names resolving to the same site means at least one is a
        mis-parsed heading. Keep the richest record; reject the rest with a
        reason, rather than exporting the same company twice.
        """
        from collections import defaultdict

        from .models import LeadStatus
        from .tools.extract import registrable_domain

        rows = self.store.db.query(
            """
            SELECT c.id, c.name, c.phone, c.address, v.website_url
            FROM candidates c JOIN verdicts v ON v.candidate_id = c.id
            WHERE c.run_id = ? AND v.has_website = 1 AND v.website_url IS NOT NULL
            """,
            (self.run_id,),
        )
        groups: dict[str, list] = defaultdict(list)
        for row in rows:
            domain = registrable_domain(row["website_url"] or "")
            if domain:
                groups[domain].append(row)

        for domain, items in groups.items():
            if len(items) < 2:
                continue
            keep = max(
                items,
                key=lambda r: (bool(r["phone"]), bool(r["address"]), len(r["name"] or "")),
            )
            for row in items:
                if row["id"] == keep["id"]:
                    continue
                self.store.upsert_lead(self.run_id, row["id"], LeadStatus.REJECTED)
                self.store.log_event(
                    self.run_id, "info", "verifier",
                    f"duplicate domain {domain}: kept {keep['name']!r}, dropped {row['name']!r}",
                    {"domain": domain, "kept": keep["name"], "dropped": row["name"]},
                )

    # ------------------------------------------------------- qualify + critic
    def _qualify_and_critic(self, campaign: Campaign) -> None:
        leads = self.store.leads(self.run_id)
        targets = [
            l for l in leads
            if l["status"] in (LeadStatus.NO_WEBSITE.value, LeadStatus.NEEDS_REVIEW.value)
        ]
        if not targets:
            log.info("qualify: nothing to score")
            return

        qualifier = QualifierAgent(self.ctx)
        critic = CriticAgent(self.ctx)
        workers = min(self.settings.concurrency, max(1, len(targets)))
        log.info("qualify: %d leads, %d workers", len(targets), workers)

        def work(lead: dict[str, Any]) -> None:
            if self._stop.is_set():
                return
            qualification: Qualification = qualifier.qualify(
                name=lead["name"],
                category=lead["category"],
                city=lead["city"],
                state=lead["state"],
                rating=lead["rating"],
                reviews=lead["reviews"],
                has_phone=bool(lead["phone"]),
                has_address=bool(lead["address"]),
                icp=campaign.icp,
            )

            skipped = (
                qualification.tier == Tier.C
                and self.options.critic_sample < 1.0
                and (hash(lead["name"]) % 1000) / 1000.0 > self.options.critic_sample
            )
            if skipped:
                status = (
                    LeadStatus.QUALIFIED
                    if qualification.tier != Tier.REJECT
                    else LeadStatus.REJECTED
                )
                self.store.upsert_lead(self.run_id, lead["candidate_id"], status, qualification)
                return

            accepted, issues, rationale = critic.review(
                name=lead["name"],
                city=lead["city"],
                state=lead["state"],
                phone=lead["phone"],
                address=lead["address"],
                category=lead["category"],
                has_website=bool(lead["has_website"]),
                website_evidence=str(lead.get("website_method") or ""),
                fit=qualification.fit,
                tier=qualification.tier.value,
                campaign_state=campaign.state,
                niche=campaign.niche,
            )

            from .models import CriticReview

            review = CriticReview(accepted=accepted, issues=issues, rationale=rationale)

            if not accepted or qualification.tier == Tier.REJECT:
                status = LeadStatus.REJECTED
            elif lead["status"] == LeadStatus.NEEDS_REVIEW.value:
                status = LeadStatus.NEEDS_REVIEW
            else:
                status = LeadStatus.QUALIFIED

            self.store.upsert_lead(
                self.run_id, lead["candidate_id"], status, qualification, review
            )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(work, l) for l in targets]
            for _ in as_completed(futures):
                self._tick("qualification")

    # ---------------------------------------------------------------- finish
    def _finish(
        self,
        campaign: Campaign,
        started: float,
        *,
        status: str,
        stats: RunStats | None = None,
        note: str = "",
    ) -> RunResult:
        counts = self.store.counts(self.run_id)
        task_counts = self.store.task_counts(self.run_id)

        all_leads = self.store.leads(self.run_id)
        exportable = [
            l for l in all_leads
            if l["status"] == LeadStatus.QUALIFIED.value
            or (self.options.include_review and l["status"] == LeadStatus.NEEDS_REVIEW.value)
        ]

        stats = stats or RunStats()
        stats.cells_total = sum(task_counts.get("discover", {}).values()) or campaign.max_zips
        stats.cells_done = task_counts.get("discover", {}).get("done", 0)
        stats.cells_failed = task_counts.get("discover", {}).get("failed", 0)
        stats.candidates = int(counts["candidates"])
        stats.with_website = int(counts["with_website"])
        stats.without_website = int(counts["without_website"])
        stats.qualified = sum(1 for l in all_leads if l["status"] == LeadStatus.QUALIFIED.value)
        stats.rejected = sum(1 for l in all_leads if l["status"] == LeadStatus.REJECTED.value)
        stats.ambiguous = sum(
            1 for l in all_leads if l["status"] == LeadStatus.NEEDS_REVIEW.value
        )
        stats.tier_a = sum(1 for l in exportable if l["tier"] == "A")
        stats.tier_b = sum(1 for l in exportable if l["tier"] == "B")
        stats.tier_c = sum(1 for l in exportable if l["tier"] == "C")
        stats.llm_calls = self.budget.calls
        stats.llm_tokens = self.budget.total_tokens
        stats.tavily_credits = float(self.tavily.credits_used) if self.tavily else 0.0
        stats.http_requests = self.probe.request_count if self.probe else 0
        stats.finished_at = utcnow()

        run_dir = self.settings.runs_dir / self.run_id
        report_path: Path | None = None
        try:
            reporter = ReporterAgent(self.ctx)
            report_md = reporter.build_report(
                run_id=self.run_id,
                niche=campaign.niche,
                state=campaign.state,
                finished=stats.finished_at or utcnow(),
                stats=stats.model_dump(),
                top_leads=exportable,
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            report_path = run_dir / "report.md"
            report_path.write_text(report_md, encoding="utf-8")
        except Exception as exc:
            log.warning("report generation failed: %s", exc)

        export_paths: list[Path] = []
        try:
            summary = {
                "run_id": self.run_id,
                "niche": campaign.niche,
                "state": campaign.state,
                "finished": stats.finished_at,
                "elapsed_seconds": round(time.monotonic() - started, 1),
                **stats.model_dump(),
            }
            export_paths = export_mod.export_all(
                exportable, run_dir, state=campaign.state,
                formats=self.options.export_formats, summary=summary,
            )
        except Exception as exc:
            log.warning("export failed: %s", exc)

        manifest = {
            "run_id": self.run_id,
            "status": status,
            "campaign": campaign.model_dump(),
            "options": self.options.__dict__,
            "stats": stats.model_dump(),
            "llm": self.llm.stats,
            "tavily": self.tavily.stats if self.tavily else None,
            "probe": self.probe.stats if self.probe else None,
            "cache": self.cache.stats,
            "artifacts": {
                "report": str(report_path) if report_path else None,
                "exports": [str(p) for p in export_paths],
            },
            "note": note,
        }
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, default=str), encoding="utf-8"
            )
        except OSError:
            log.warning("could not write manifest", exc_info=True)

        self.store.finish_run(self.run_id, status, stats, str(report_path) if report_path else None)
        self.store.log_event(self.run_id, "info", "orchestrator", f"run {status}", manifest["stats"])

        log.info(
            "run %s %s: %d candidates, %d without website, %d qualified (%d A / %d B)",
            self.run_id, status, stats.candidates, stats.without_website,
            stats.qualified, stats.tier_a, stats.tier_b,
        )

        return RunResult(
            run_id=self.run_id,
            stats=stats.model_dump(),
            report_path=report_path,
            export_paths=export_paths,
            leads=exportable,
            status=status,
        )

    # ------------------------------------------------------------- progress
    def _tick(self, label: str) -> None:
        with self._progress_lock:
            self._done_counter += 1
            n = self._done_counter
        if n % 10 == 0:
            log.info(
                "progress: %d steps | llm=%d tavily=%.1f credits",
                n, self.budget.calls,
                self.tavily.credits_used if self.tavily else 0.0,
            )

    # ---------------------------------------------------------------- close
    def close(self) -> None:
        self.db.close()
