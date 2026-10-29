"""Command-line interface.

    leads doctor                 validate config, files and connectivity
    leads agents                 show the agent graph
    leads plan --state TX        preview the plan without touching the network
    leads run --state TX --zips 5
    leads status                 latest run summary
    leads runs                   recent runs
    leads export RUN_ID          re-export a finished run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .agents import AGENT_GRAPH
from .config import Settings, get_settings
from .errors import ConfigError
from .models import Campaign, ICP
from .orchestrator import RunOptions, Orchestrator
from .store import Database, Store
from .zipcodes import ZipIndex


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _settings(args: argparse.Namespace) -> Settings:
    s = get_settings()
    if getattr(args, "concurrency", None):
        s = s.model_copy(update={"concurrency": args.concurrency})
    return s


def _campaign(args: argparse.Namespace) -> Campaign:
    zips: list[str] = []
    if getattr(args, "zips", None):
        raw = str(args.zips)
        if raw.endswith(".txt") or Path(raw).exists():
            zips = [ln.strip() for ln in Path(raw).read_text(encoding="utf-8").splitlines() if ln.strip()]
        else:
            zips = [z.strip() for z in raw.split(",") if z.strip()]

    icp = ICP(
        niche=args.niche,
        min_rating=args.min_rating,
        min_reviews=args.min_reviews,
    )
    return Campaign(
        niche=args.niche,
        state=args.state,
        zips=zips,
        max_zips=args.max_zips,
        icp=icp,
    )


def _fmt_table(rows: list[dict], columns: list[tuple[str, str]], limit: int = 25) -> str:
    if not rows:
        return "  (none)"
    header = "  " + " | ".join(title for _, title in columns)
    sep = "  " + "-+-".join("-" * len(title) for _, title in columns)
    lines = [header, sep]
    for row in rows[:limit]:
        cells = []
        for key, _ in columns:
            value = row.get(key)
            text = "" if value is None else str(value)
            cells.append(text[:28])
        lines.append("  " + " | ".join(cells))
    if len(rows) > limit:
        lines.append(f"  ... {len(rows) - limit} more")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

def cmd_doctor(args: argparse.Namespace) -> int:
    s = _settings(args)
    ok = True
    print(f"leads v{__version__} | python {sys.version.split()[0]}")
    print(f"home: {s.home}")

    print("\n[config]")
    for label, present, hint in (
        ("OPENROUTER_API_KEY", bool(s.openrouter_api_key), "required for LLM agents"),
        ("MODEL", bool(s.model), "e.g. nvidia/nemotron-3-ultra-550b-a55b:free"),
        ("TAVILY_API_KEY", bool(s.tavily_api_key), "free tier: 1,000 credits/month"),
    ):
        mark = "ok  " if present else "MISS"
        print(f"  {mark} {label:<20} {hint}")
        if not present:
            ok = False
    print(f"  ok   model                {s.model or '-'}")
    print(f"  ok   concurrency          {s.concurrency}")
    print(f"  ok   llm call budget      {s.max_llm_calls}")
    print(f"  ok   tavily credit budget {s.max_tavily_credits}")

    print("\n[data]")
    datasets = [
        "simplemaps_uszips_basicv1.95.1/uszips.csv",
        "USZipsWithLatLon_20231227.csv",
        "zipcodes/USZipsWithLatLon_20231227.csv",
    ]
    index = ZipIndex.from_candidates(datasets)
    if len(index) > 1000:
        print(f"  ok   ZIP index            {len(index)} codes")
    else:
        print("  warn ZIP index            not found; state falls back to ZIP prefix table")

    print("\n[storage]")
    try:
        s.ensure_dirs()
        db = Database(s.db_path)
        Store(db).list_runs(1)
        print(f"  ok   database             {s.db_path}")
        print(f"  ok   writable             {s.home}")
    except Exception as exc:
        print(f"  FAIL database: {exc}")
        ok = False

    if args.network:
        print("\n[connectivity]")
        ok = _probe_network(s) and ok
    else:
        print("\n[connectivity] skipped (pass --network to ping providers)")

    print("\n" + ("READY" if ok else "NOT READY — fix the MISS/FAIL lines above"))
    return 0 if ok else 1


def _probe_network(s: Settings) -> bool:
    import requests

    ok = True
    if s.llm_ready:
        try:
            r = requests.post(
                f"{s.openrouter_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {s.openrouter_api_key}"},
                json={
                    "model": s.model,
                    "messages": [{"role": "user", "content": 'Reply with {"ok":true}'}],
                    "max_tokens": 40,
                    "response_format": {"type": "json_object"},
                },
                timeout=90,
            )
            print(f"  {'ok  ' if r.ok else 'FAIL'} LLM ({s.model}) HTTP {r.status_code}")
            ok = ok and r.ok
        except requests.RequestException as exc:
            print(f"  FAIL LLM: {exc}")
            ok = False
    if s.tavily_ready:
        try:
            r = requests.post(
                "https://api.tavily.com/search",
                headers={"Authorization": f"Bearer {s.tavily_api_key}"},
                json={"query": "roofing contractor", "max_results": 1, "search_depth": "basic"},
                timeout=30,
            )
            print(f"  {'ok  ' if r.ok else 'FAIL'} Tavily search HTTP {r.status_code} (costs 1 credit)")
            ok = ok and r.ok
        except requests.RequestException as exc:
            print(f"  FAIL Tavily: {exc}")
            ok = False
    return ok


def cmd_agents(args: argparse.Namespace) -> int:
    print("agent graph: planner -> discovery -> verifier -> qualifier -> critic -> reporter\n")
    for name, desc in AGENT_GRAPH.items():
        print(f"  {name:<11} {desc}")
    print(
        "\nThe supervisor (orchestrator.py) owns ordering, concurrency, retries and\n"
        "checkpoints. Agents never call each other and each has a deterministic fallback."
    )
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    from .logging_setup import configure_logging

    s = _settings(args)
    configure_logging("WARNING", s.logs_dir)
    campaign = _campaign(args)

    orch = Orchestrator(s, run_id="plan-preview", options=RunOptions())
    try:
        cells = orch._resolve_cells(campaign)
        if not cells:
            print(f"No ZIPs resolved for {campaign.state}. Pass --zips or check your ZIP files.")
            return 1
        plan = orch._plan(campaign, cells)

        print(f"campaign : {campaign.niche} | {campaign.state} | {len(cells)} cells")
        print(f"planner  : {'LLM' if plan.llm_generated else 'deterministic'}")
        if plan.notes:
            print(f"notes    : {plan.notes}")
        print("\ncells:")
        for cell in cells[:15]:
            print(f"  {cell.zip}  {cell.label}")
        if len(cells) > 15:
            print(f"  ... {len(cells) - 15} more")
        print("\nqueries:")
        for q in plan.queries:
            print(f"  - {q}")
        print("\nicp:")
        print(f"  min rating {plan.icp.min_rating} | min reviews {plan.icp.min_reviews}")
        print(f"  traits: {', '.join(plan.icp.target_traits)}")
        est = len(cells) * min(3, len(plan.queries)) + len(cells) * 2
        print(f"\nestimated Tavily credits: ~{est} (free tier allows 1,000/month)")
        return 0
    finally:
        orch.close()


def cmd_run(args: argparse.Namespace) -> int:
    s = _settings(args)
    try:
        s.require_llm()
    except ConfigError as exc:
        print(f"[config] {exc}")
        if not args.allow_no_llm:
            print("        Deterministic mode is available: pass --allow-no-llm")
            return 2
        print("        Continuing in deterministic mode (no LLM agents).")

    campaign = _campaign(args)
    options = RunOptions(
        use_osm=not args.no_osm,
        verify_search=not args.no_verify_search,
        critic_sample=args.critic_sample,
        include_review=args.include_review,
        resume=args.resume,
        export_formats=tuple(args.format),
        max_cells_in_parallel=args.concurrency,
    )

    orch = Orchestrator(s, run_id=args.run_id, options=options)
    try:
        result = orch.run(campaign)
    except KeyboardInterrupt:
        print("\nInterrupted. Partial work is checkpointed; rerun with --resume.")
        return 130
    finally:
        orch.close()

    stats = result.stats
    print("\n" + "=" * 66)
    print(f"run {result.run_id} — {result.status}")
    print("=" * 66)
    print(
        _fmt_table(
            [
                {"metric": "cells searched", "value": stats.get("cells_done")},
                {"metric": "candidates found", "value": stats.get("candidates")},
                {"metric": "no website", "value": stats.get("without_website")},
                {"metric": "has website", "value": stats.get("with_website")},
                {"metric": "qualified", "value": stats.get("qualified")},
                {"metric": "tier A / B / C", "value": f"{stats.get('tier_a')} / {stats.get('tier_b')} / {stats.get('tier_c')}"},
                {"metric": "rejected", "value": stats.get("rejected")},
                {"metric": "needs review", "value": stats.get("ambiguous")},
                {"metric": "LLM calls", "value": stats.get("llm_calls")},
                {"metric": "Tavily credits", "value": stats.get("tavily_credits")},
            ],
            [("metric", "metric"), ("value", "value")],
        )
    )
    if result.leads:
        print("\nleads:")
        print(
            _fmt_table(
                result.leads,
                [("name", "name"), ("city", "city"), ("phone", "phone"),
                 ("tier", "tier"), ("fit", "fit")],
                limit=12,
            )
        )
    if result.report_path:
        print(f"\nreport  : {result.report_path}")
    for p in result.export_paths:
        print(f"export  : {p}")
    return 0 if result.status == "completed" else 1


def cmd_status(args: argparse.Namespace) -> int:
    s = _settings(args)
    db = Database(s.db_path)
    store = Store(db)
    run = store.get_run(args.run_id) if args.run_id else store.latest_run()
    if not run:
        print("no runs found")
        return 1

    print(f"run      : {run['run_id']}")
    print(f"status   : {run['status']}")
    print(f"created  : {run['created_at']}")
    print(f"finished : {run['finished_at'] or '-'}")
    print(f"campaign : {run['niche']} | {run['state']}")
    if run.get("stats_json"):
        stats = json.loads(run["stats_json"])
        for key in (
            "cells_done", "candidates", "without_website", "with_website",
            "qualified", "tier_a", "tier_b", "rejected", "ambiguous",
            "llm_calls", "tavily_credits",
        ):
            print(f"  {key:<16} {stats.get(key)}")
    if run.get("report_path"):
        print(f"report   : {run['report_path']}")

    counts = store.task_counts(run["run_id"])
    if counts:
        print("\nstages:")
        for stage, by_status in counts.items():
            parts = ", ".join(f"{k}={v}" for k, v in sorted(by_status.items()))
            print(f"  {stage:<10} {parts}")
    db.close()
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    s = _settings(args)
    db = Database(s.db_path)
    runs = Store(db).list_runs(args.limit)
    db.close()
    print(
        _fmt_table(
            runs,
            [("run_id", "run_id"), ("status", "status"), ("state", "state"),
             ("niche", "niche"), ("created_at", "created")],
        )
    )
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from . import export as export_mod

    s = _settings(args)
    db = Database(s.db_path)
    store = Store(db)
    run = store.get_run(args.run_id) if args.run_id else store.latest_run()
    if not run:
        print("no runs found")
        return 1

    leads = store.leads(run["run_id"])
    if not args.all:
        from .models import LeadStatus

        wanted = {LeadStatus.QUALIFIED.value}
        if args.include_review:
            wanted.add(LeadStatus.NEEDS_REVIEW.value)
        leads = [l for l in leads if l["status"] in wanted]

    if args.tier:
        wanted_tiers = {t.upper() for t in args.tier.split(",")}
        leads = [l for l in leads if (l.get("tier") or "").upper() in wanted_tiers]

    out_dir = Path(args.out) if args.out else s.runs_dir / run["run_id"]
    paths = export_mod.export_all(
        leads, out_dir, state=run["state"] or "us", formats=args.format
    )
    for p in paths:
        print(f"wrote {p}  ({len(leads)} leads)")
    db.close()
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="leads",
        description="Multi-agent discovery of high-rated local businesses without websites.",
    )
    p.add_argument("--version", action="version", version=f"leads {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--concurrency", type=int, default=None, help="parallel workers")

    d = sub.add_parser("doctor", parents=[common], help="validate config and connectivity")
    d.add_argument("--network", action="store_true", help="ping providers (costs 1 Tavily credit)")
    d.set_defaults(func=cmd_doctor)

    a = sub.add_parser("agents", help="show the agent graph")
    a.set_defaults(func=cmd_agents)

    def add_campaign_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--niche", default="Roofing Contractor", help="business type")
        parser.add_argument("--state", default="TX", help="two-letter US state")
        parser.add_argument("--zips", default="", help="comma list, or path to a .txt of ZIPs")
        parser.add_argument("--max-zips", dest="max_zips", type=int, default=10)
        parser.add_argument("--min-rating", dest="min_rating", type=float, default=4.5)
        parser.add_argument("--min-reviews", dest="min_reviews", type=int, default=10)

    pl = sub.add_parser("plan", parents=[common], help="preview the plan (no writes)")
    add_campaign_args(pl)
    pl.set_defaults(func=cmd_plan)

    r = sub.add_parser("run", parents=[common], help="execute a lead-generation run")
    add_campaign_args(r)
    r.add_argument("--run-id", default=None)
    r.add_argument("--resume", action="store_true", help="skip already-completed work")
    r.add_argument("--no-osm", action="store_true", help="disable OpenStreetMap discovery")
    r.add_argument("--no-verify-search", action="store_true", help="skip paid site verification")
    r.add_argument("--critic-sample", type=float, default=1.0,
                   help="fraction of tier-C leads audited by the critic (0-1)")
    r.add_argument("--include-review", action="store_true", help="export needs_review leads too")
    r.add_argument("--format", nargs="+", default=["xlsx", "csv"], choices=["xlsx", "csv", "jsonl"])
    r.add_argument("--allow-no-llm", action="store_true",
                   help="run deterministic agents when no LLM key is configured")
    r.set_defaults(func=cmd_run)

    st = sub.add_parser("status", help="show a run")
    st.add_argument("--run-id", default=None)
    st.set_defaults(func=cmd_status)

    rs = sub.add_parser("runs", help="list recent runs")
    rs.add_argument("--limit", type=int, default=15)
    rs.set_defaults(func=cmd_runs)

    ex = sub.add_parser("export", help="re-export a run")
    ex.add_argument("run_id", nargs="?", default=None)
    ex.add_argument("--out", default=None)
    ex.add_argument("--tier", default=None, help="e.g. A,B")
    ex.add_argument("--format", nargs="+", default=["xlsx", "csv"], choices=["xlsx", "csv", "jsonl"])
    ex.add_argument("--all", action="store_true", help="include every status")
    ex.add_argument("--include-review", action="store_true")
    ex.set_defaults(func=cmd_export)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except ConfigError as exc:
        print(f"[config] {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
